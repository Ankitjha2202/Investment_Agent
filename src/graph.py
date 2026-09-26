"""
Investment agent pipeline (LangGraph) — self-healing RAG with improve retries.

┌─────────────────────────────────────────────────────────────────────────┐
│  START                                                                  │
│    │                                                                    │
│    ▼                                                                    │
│  assess ────────── brain → sql | meetings | both + top_k chunks         │
│    │                                                                    │
│    ├─ route=sql ──────────► sql ──────────────────────► synthesize      │
│    ├─ route=meetings ─────► meetings(top_k) ──────────► synthesize      │
│    └─ route=both ─────────► sql ──► meetings(top_k) ──► synthesize      │
│                                                          │              │
│                                                          ▼              │
│                                                       improve           │
│                                                    (critique + heal)    │
│                                                          │              │
│                          ┌──── accept / retries exhausted ──► END       │
│                          │                                              │
│                          └──── retry (≤ 2) ──► replan route/top_k/      │
│                                query/feedback ──► sql | meetings ───────┘
└─────────────────────────────────────────────────────────────────────────┘

Nodes own one job (SRP). Tools live in sibling modules (DIP):
  assess     → src.assess.assess_question   (brain: route + top_k)
  sql        → src.text_to_sql.answer_with_sql   (+ guardrails)
  meetings   → src.retrieve.retrieve_meetings
  synthesize → src.synthesize.synthesize
  improve    → src.improve.critique_and_improve  (self-heal, max 2 retries)

Edges (after_assess / after_sql / after_improve) decide the next node —
open/closed for new sources later without rewriting the UI.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from src.assess import AssessDecision, assess_question
from src.improve import critique_and_improve
from src.retrieve import MeetingResult, retrieve_meetings
from src.synthesize import synthesize
from src.text_to_sql import SqlResult, answer_with_sql

StepCallback = Callable[[str, str, str], None]

# Graph-level self-heal budget (distinct from SQL's internal repair attempt).
MAX_RETRIES = 2

# Human-readable node map (UI labels + plan copy). Single source of truth.
NODES: dict[str, dict[str, str]] = {
    "assess": {
        "title": "Assess & plan",
        "description": "Brain: choose SQL vs meetings vs both, and how many note chunks to retrieve.",
    },
    "sql": {
        "title": "Query structured data",
        "description": "Generate safe SELECT over investments / performance, then execute.",
    },
    "meetings": {
        "title": "Retrieve meeting notes",
        "description": "Embed the question and search meeting notes via pgvector.",
    },
    "synthesize": {
        "title": "Synthesize answer",
        "description": "Ground the final reply only in the evidence gathered above.",
    },
    "improve": {
        "title": "Critique & heal",
        "description": "Grade evidence/answer; if weak, replan and retry (up to 2 times).",
    },
}

PIPELINE_ORDER = ("assess", "sql", "meetings", "synthesize", "improve")


class AgentState(TypedDict, total=False):
    question: str
    route: str
    route_reason: str
    top_k: int
    client_id: str | None
    plan: list[dict[str, Any]]
    sql_result: SqlResult | None
    meeting_result: MeetingResult | None
    answer: str
    # Self-healing RAG
    retry_count: int
    max_retries: int
    retrieval_query: str | None
    improve_feedback: str | None
    improve_history: list[str]
    quality_ok: bool
    heal_action: Literal["accept", "retry"]


def _emit(config: RunnableConfig | None, step_id: str, status: str, detail: str = "") -> None:
    cfg = ((config or {}).get("configurable") or {}) if config is not None else {}
    cb = cfg.get("on_step")
    if cb:
        cb(step_id, status, detail)


def _plan_for(decision: AssessDecision, *, retry_count: int = 0) -> list[dict[str, Any]]:
    steps = [{"id": "assess", **NODES["assess"]}]
    if decision.route in {"sql", "both"}:
        steps.append({"id": "sql", **NODES["sql"]})
    if decision.route in {"meetings", "both"}:
        meet = dict(NODES["meetings"])
        meet["description"] += f" Retrieving top_k={decision.top_k} chunks."
        if decision.client_id:
            meet["description"] += f" Filtered to client {decision.client_id}."
        steps.append({"id": "meetings", **meet})
    steps.append({"id": "synthesize", **NODES["synthesize"]})
    improve = dict(NODES["improve"])
    if retry_count:
        improve["description"] += f" Retry {retry_count}/{MAX_RETRIES}."
    steps.append({"id": "improve", **improve})
    return steps


def _emit_pending_tools(
    config: RunnableConfig,
    *,
    route: str,
    top_k: int,
    client_id: str | None,
) -> None:
    """Re-arm timeline chips when the heal loop re-enters tools."""
    if route in {"sql", "both"}:
        _emit(config, "sql", "pending", NODES["sql"]["description"])
    else:
        _emit(config, "sql", "skipped", "Not needed for this route.")
    if route in {"meetings", "both"}:
        detail = NODES["meetings"]["description"] + f" Retrieving top_k={top_k} chunks."
        if client_id:
            detail += f" Filtered to client {client_id}."
        _emit(config, "meetings", "pending", detail)
    else:
        _emit(config, "meetings", "skipped", "Not needed for this route.")
    _emit(config, "synthesize", "pending", NODES["synthesize"]["description"])
    _emit(config, "improve", "pending", NODES["improve"]["description"])


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def assess_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    retry_count = int(state.get("retry_count") or 0)
    feedback = state.get("improve_feedback")
    running = (
        f"Replanning after heal feedback (retry {retry_count}/{MAX_RETRIES})…"
        if feedback
        else "Planning tools and retrieval depth…"
    )
    _emit(config, "assess", "running", running)

    decision = assess_question(
        state["question"],
        feedback=feedback,
        prior_route=state.get("route"),
        prior_top_k=state.get("top_k"),
    )
    # Prefer critic-raised top_k when assess under-shoots on a retry.
    if feedback and state.get("top_k"):
        decision.top_k = max(decision.top_k, int(state["top_k"]))
    plan = _plan_for(decision, retry_count=retry_count)

    detail = f"Route → **{decision.route}**"
    if decision.route in {"meetings", "both"}:
        detail += f" · **top_k={decision.top_k}** chunks"
    if decision.client_id:
        detail += f" · client `{decision.client_id}`"
    if retry_count:
        detail += f" · heal retry **{retry_count}/{MAX_RETRIES}**"
    if decision.reason:
        detail += f"\n{decision.reason}"
    _emit(config, "assess", "done", detail)

    for step in plan:
        if step["id"] != "assess":
            _emit(config, step["id"], "pending", step["description"])

    if decision.route == "sql":
        _emit(config, "meetings", "skipped", "Not needed for this route.")
    elif decision.route == "meetings":
        _emit(config, "sql", "skipped", "Not needed for this route.")

    out: dict[str, Any] = {
        "route": decision.route,
        "route_reason": decision.reason,
        "top_k": decision.top_k,
        "client_id": decision.client_id,
        "plan": plan,
        "sql_result": None,
        "meeting_result": None,
        "answer": "",
        "quality_ok": False,
        "max_retries": int(state.get("max_retries") or MAX_RETRIES),
        "retry_count": retry_count,
    }
    # Fresh assess without critic feedback clears retrieval rewrite.
    if not feedback:
        out["retrieval_query"] = None
        out["improve_feedback"] = None
        out["improve_history"] = list(state.get("improve_history") or [])
    return out


def sql_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    feedback = state.get("improve_feedback")
    detail = "Generating and executing text-to-SQL…"
    if feedback:
        detail = "Re-running text-to-SQL with critic feedback…"
    _emit(config, "sql", "running", detail)
    result = answer_with_sql(state["question"], feedback=feedback)

    if result.error:
        err_detail = result.error
        if result.rationale:
            err_detail += f"\nRationale: {result.rationale}"
        _emit(config, "sql", "error", err_detail)
    else:
        n = len(result.rows)
        lines = []
        if result.rationale:
            lines.append(result.rationale)
        lines.append(f"Returned **{n}** row{'s' if n != 1 else ''}.")
        if result.sql:
            lines.append(f"```sql\n{result.sql}\n```")
        _emit(config, "sql", "done", "\n\n".join(lines))

    return {"sql_result": result}


def meetings_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    client_id = state.get("client_id")
    top_k = int(state.get("top_k") or 6)
    query = (state.get("retrieval_query") or state["question"]).strip() or state["question"]
    filt = f" for client `{client_id}`" if client_id else ""
    rewritten = (
        f" Rewritten query: `{query}`."
        if state.get("retrieval_query") and query != state["question"]
        else ""
    )
    _emit(
        config,
        "meetings",
        "running",
        f"Embedding query and searching notes{filt} (top_k={top_k})…{rewritten}",
    )
    result = retrieve_meetings(query, top_k=top_k, client_id=client_id)

    if result.error:
        _emit(config, "meetings", "error", result.error)
    else:
        n = len(result.hits)
        tops = ", ".join(
            f"{h.company or 'Unknown'} ({h.score:.2f})" for h in result.hits[:3]
        )
        detail = f"Found **{n}** / {top_k} requested note{'s' if n != 1 else ''}."
        if tops:
            detail += f" Top matches: {tops}."
        _emit(config, "meetings", "done", detail)

    return {"meeting_result": result}


def synthesize_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    attempt = int(state.get("retry_count") or 0) + 1
    _emit(
        config,
        "synthesize",
        "running",
        f"Composing a grounded answer from evidence (attempt {attempt})…",
    )
    answer = synthesize(
        state["question"],
        route=state.get("route") or "both",
        sql_result=state.get("sql_result"),
        meeting_result=state.get("meeting_result"),
    )
    _emit(config, "synthesize", "done", "Answer ready.")
    return {"answer": answer}


def improve_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Critique evidence + answer; heal plan and loop back up to MAX_RETRIES."""
    retry_count = int(state.get("retry_count") or 0)
    max_retries = int(state.get("max_retries") or MAX_RETRIES)
    _emit(
        config,
        "improve",
        "running",
        f"Critiquing evidence & answer (retry budget {retry_count}/{max_retries})…",
    )

    decision = critique_and_improve(
        state["question"],
        route=state.get("route") or "both",
        top_k=int(state.get("top_k") or 6),
        client_id=state.get("client_id"),
        sql_result=state.get("sql_result"),
        meeting_result=state.get("meeting_result"),
        answer=state.get("answer") or "",
        retry_count=retry_count,
        max_retries=max_retries,
        prior_feedback=state.get("improve_feedback"),
    )

    history = list(state.get("improve_history") or [])
    history.append(decision.reason)

    if decision.action == "accept" or decision.ok:
        detail = f"**Accept** — {decision.reason}"
        if retry_count:
            detail += f"\nHeal attempts used: **{retry_count}/{max_retries}**."
        _emit(config, "improve", "done", detail)
        return {
            "quality_ok": bool(decision.ok),
            "heal_action": "accept",
            "improve_history": history,
            "improve_feedback": state.get("improve_feedback"),
        }

    # --- Heal: apply improvements and schedule another tool pass ---
    new_retry = retry_count + 1
    route = decision.route or state.get("route") or "both"
    top_k = decision.top_k if decision.top_k is not None else int(state.get("top_k") or 6)
    client_id = None if decision.clear_client_id else state.get("client_id")
    retrieval_query = decision.retrieval_query or state.get("retrieval_query")
    feedback = decision.feedback or decision.reason

    heal_bits = [f"Route → **{route}**", f"**top_k={top_k}**"]
    if decision.clear_client_id:
        heal_bits.append("cleared client filter")
    elif client_id:
        heal_bits.append(f"client `{client_id}`")
    if retrieval_query:
        heal_bits.append(f"rewritten query `{retrieval_query[:80]}`")
    detail = (
        f"**Retry {new_retry}/{max_retries}** — {decision.reason}\n"
        + " · ".join(heal_bits)
        + (f"\nFeedback: {feedback}" if feedback else "")
    )
    _emit(config, "improve", "done", detail)

    _emit_pending_tools(
        config,
        route=route,
        top_k=top_k,
        client_id=client_id,
    )

    # Refresh plan metadata for debug / UI consumers.
    fake = AssessDecision(
        route=route,  # type: ignore[arg-type]
        reason=f"Self-heal replan: {feedback}",
        top_k=top_k,
        client_id=client_id,
    )
    plan = _plan_for(fake, retry_count=new_retry)

    return {
        "quality_ok": False,
        "heal_action": "retry",
        "retry_count": new_retry,
        "route": route,
        "route_reason": f"Self-heal replan: {decision.reason}",
        "top_k": top_k,
        "client_id": client_id,
        "retrieval_query": retrieval_query,
        "improve_feedback": feedback,
        "improve_history": history,
        "plan": plan,
        "sql_result": None,
        "meeting_result": None,
        "answer": state.get("answer") or "",
    }


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------


def after_assess(state: AgentState) -> Literal["sql", "meetings"]:
    return "meetings" if state.get("route") == "meetings" else "sql"


def after_sql(state: AgentState) -> Literal["meetings", "synthesize"]:
    return "meetings" if state.get("route") == "both" else "synthesize"


def after_improve(state: AgentState) -> Literal["sql", "meetings", "__end__"]:
    """Loop back into tools when the critic scheduled a heal retry."""
    if state.get("heal_action") != "retry":
        return "__end__"
    return "meetings" if state.get("route") == "meetings" else "sql"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("assess", assess_node)
    g.add_node("sql", sql_node)
    g.add_node("meetings", meetings_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("improve", improve_node)

    g.add_edge(START, "assess")
    g.add_conditional_edges(
        "assess", after_assess, {"sql": "sql", "meetings": "meetings"}
    )
    g.add_conditional_edges(
        "sql", after_sql, {"meetings": "meetings", "synthesize": "synthesize"}
    )
    g.add_edge("meetings", "synthesize")
    g.add_edge("synthesize", "improve")
    g.add_conditional_edges(
        "improve",
        after_improve,
        {"sql": "sql", "meetings": "meetings", "__end__": END},
    )
    return g.compile()


agent_graph = build_graph()
