"""Self-improving critique node — grade evidence/answer and propose heals."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from src.assess import DEFAULT_TOP_K, MAX_TOP_K, MIN_TOP_K, Route
from src.llm import chat_json
from src.retrieve import MeetingResult
from src.text_to_sql import SqlResult

Action = Literal["accept", "retry"]


def _clamp_top_k(value: object) -> int:
    try:
        k = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_TOP_K
    return max(MIN_TOP_K, min(MAX_TOP_K, k))

IMPROVE_SYSTEM = """
You are the self-healing critic for an investment analytics RAG agent.

Judge whether the gathered evidence and draft answer adequately answer the user.
If not, propose concrete fixes for the NEXT pipeline attempt.

Failure modes to watch for:
- Wrong route (needed SQL numbers but only meetings, or vice versa)
- SQL error / 0 rows when structured data should exist
- Meeting retrieval empty, off-topic, or too shallow (raise top_k)
- Over-filtering by client_id that wiped results
- Weak semantic query — rewrite retrieval_query for better embedding match
- Draft answer admits missing evidence or hallucinates around gaps

Return JSON only:
{
  "ok": true|false,
  "reason": "short critique",
  "route": "sql"|"meetings"|"both"|null,
  "top_k": <integer 3-20 or null>,
  "clear_client_id": true|false,
  "retrieval_query": "<rewritten search query or null>",
  "feedback": "instructions for the next SQL/retrieval pass"
}

Rules:
- ok=true only if evidence supports a solid answer to the question.
- When ok=false, always set feedback with actionable fixes.
- Prefer expanding route toward "both" when unsure.
- Raise top_k when meetings look thin for a broad question (cap 20).
- clear_client_id=true if a client filter likely caused empty/wrong meetings.
- retrieval_query should be a focused semantic search string (entities, themes),
  not a full chatbot reply. null if meetings are unused or query is fine.
- Do not invent that data exists; only improve the search/plan strategy.
""".strip()


@dataclass
class ImproveDecision:
    """Critique outcome + optional healing plan for a retry."""

    ok: bool
    reason: str
    action: Action
    route: Route | None = None
    top_k: int | None = None
    clear_client_id: bool = False
    retrieval_query: str | None = None
    feedback: str = ""


def _sql_summary(sql: SqlResult | None) -> dict[str, Any] | None:
    if sql is None:
        return None
    return {
        "query": sql.sql,
        "rationale": sql.rationale,
        "error": sql.error,
        "row_count": len(sql.rows),
        "sample_rows": sql.rows[:5],
    }


def _meetings_summary(meetings: MeetingResult | None) -> dict[str, Any] | None:
    if meetings is None:
        return None
    tops = [
        {
            "company": h.company,
            "score": round(h.score, 3),
            "date": h.meeting_date,
            "summary": (h.summary or "")[:180],
        }
        for h in meetings.hits[:5]
    ]
    return {
        "error": meetings.error,
        "hit_count": len(meetings.hits),
        "top_hits": tops,
        "avg_score": (
            round(sum(h.score for h in meetings.hits) / len(meetings.hits), 3)
            if meetings.hits
            else None
        ),
    }


def _heuristic_heal(
    question: str,
    *,
    route: str,
    top_k: int,
    client_id: str | None,
    sql_result: SqlResult | None,
    meeting_result: MeetingResult | None,
    answer: str,
) -> ImproveDecision | None:
    """Fast deterministic heals for obvious pipeline failures. None = ask LLM."""
    needs_sql = route in {"sql", "both"}
    needs_meet = route in {"meetings", "both"}
    sql_broken = bool(
        needs_sql
        and sql_result is not None
        and (sql_result.error or not sql_result.rows)
    )
    meet_broken = bool(
        needs_meet
        and meeting_result is not None
        and (meeting_result.error or not meeting_result.hits)
    )
    weak_meet = bool(
        needs_meet
        and meeting_result
        and meeting_result.hits
        and not meeting_result.error
        and (
            len(meeting_result.hits) < max(2, top_k // 3)
            or (
                sum(h.score for h in meeting_result.hits) / len(meeting_result.hits)
                < 0.25
            )
        )
    )
    answer_lacks = bool(
        re.search(
            r"\b(could(?:n'?t| not)|unable to|no (?:data|evidence|results|information)|"
            r"not (?:enough|sufficient)|don'?t have|cannot find)\b",
            answer or "",
            re.I,
        )
    )

    if not (sql_broken or meet_broken or weak_meet or answer_lacks):
        return None

    new_route: Route = route if route in {"sql", "meetings", "both"} else "both"  # type: ignore[assignment]
    if sql_broken and not needs_meet:
        new_route = "both"
    if meet_broken and not needs_sql:
        new_route = "both"
    if answer_lacks and new_route != "both":
        new_route = "both"

    new_top_k = top_k
    if meet_broken or weak_meet or (answer_lacks and needs_meet):
        new_top_k = min(MAX_TOP_K, max(top_k + 4, DEFAULT_TOP_K + 2))

    clear_client = bool(client_id and (meet_broken or weak_meet))

    parts: list[str] = []
    if sql_broken:
        err = (sql_result.error if sql_result else None) or "0 rows"
        parts.append(f"SQL path failed ({err}). Broaden filters / fix columns.")
    if meet_broken:
        err = (meeting_result.error if meeting_result else None) or "0 hits"
        parts.append(f"Meeting retrieval failed ({err}).")
    if weak_meet:
        parts.append("Meeting hits look thin/low-relevance; deepen retrieval.")
    if answer_lacks:
        parts.append("Draft answer reports missing evidence.")

    feedback = " ".join(parts)
    retrieval_query = None
    if new_route in {"meetings", "both"} and (meet_broken or weak_meet):
        # Light rewrite: strip chatter, keep entities — LLM may refine later.
        retrieval_query = re.sub(r"\s+", " ", question).strip()

    return ImproveDecision(
        ok=False,
        reason=feedback or "Evidence insufficient",
        action="retry",
        route=new_route,
        top_k=new_top_k,
        clear_client_id=clear_client,
        retrieval_query=retrieval_query,
        feedback=feedback,
    )


def critique_and_improve(
    question: str,
    *,
    route: str,
    top_k: int,
    client_id: str | None,
    sql_result: SqlResult | None,
    meeting_result: MeetingResult | None,
    answer: str,
    retry_count: int,
    max_retries: int,
    prior_feedback: str | None = None,
) -> ImproveDecision:
    """Grade the attempt; if weak and retries remain, propose a healing plan."""
    heuristic = _heuristic_heal(
        question,
        route=route,
        top_k=top_k,
        client_id=client_id,
        sql_result=sql_result,
        meeting_result=meeting_result,
        answer=answer,
    )

    payload = {
        "question": question,
        "attempt": retry_count + 1,
        "max_attempts": max_retries + 1,
        "plan": {
            "route": route,
            "top_k": top_k,
            "client_id": client_id,
            "prior_feedback": prior_feedback,
        },
        "sql": _sql_summary(sql_result),
        "meetings": _meetings_summary(meeting_result),
        "draft_answer": (answer or "")[:2500],
    }
    user = (
        "Critique this pipeline attempt and propose heals if needed.\n"
        + json.dumps(payload, default=str)[:12000]
    )

    try:
        raw = chat_json(IMPROVE_SYSTEM, user)
        data = json.loads(raw)
    except (json.JSONDecodeError, Exception):
        # Fall back to heuristic, or accept if nothing obvious is wrong.
        if heuristic is not None and retry_count < max_retries:
            return heuristic
        if heuristic is not None:
            return ImproveDecision(
                ok=False,
                reason=(
                    f"{heuristic.reason} Max retries ({max_retries}) reached; "
                    "accepting best-effort answer."
                ),
                action="accept",
                feedback=heuristic.feedback,
            )
        return ImproveDecision(
            ok=True,
            reason="Critic unavailable; accepting current answer.",
            action="accept",
        )

    ok = bool(data.get("ok"))
    reason = str(data.get("reason") or "").strip() or (
        "Evidence looks sufficient" if ok else "Evidence insufficient"
    )

    if ok and heuristic is None:
        return ImproveDecision(ok=True, reason=reason, action="accept")

    # Prefer retry when either the LLM or heuristics say the answer is weak.
    if ok and heuristic is not None:
        # LLM said ok but deterministic checks failed — trust heuristics.
        data_ok = False
        reason = heuristic.reason
    else:
        data_ok = ok

    if data_ok:
        return ImproveDecision(ok=True, reason=reason, action="accept")

    # Evidence is weak — only heal if retry budget remains.
    if retry_count >= max_retries:
        return ImproveDecision(
            ok=False,
            reason=f"{reason} Max retries ({max_retries}) reached; accepting best-effort.",
            action="accept",
            feedback=prior_feedback or "",
        )

    route_raw = data.get("route")
    new_route: Route | None = None
    if isinstance(route_raw, str) and route_raw.lower().strip() in {
        "sql",
        "meetings",
        "both",
    }:
        new_route = route_raw.lower().strip()  # type: ignore[assignment]
    elif heuristic and heuristic.route:
        new_route = heuristic.route

    top_k_val = data.get("top_k")
    new_top_k: int | None
    if top_k_val is None and heuristic is not None:
        new_top_k = heuristic.top_k
    elif top_k_val is None:
        new_top_k = None
    else:
        new_top_k = _clamp_top_k(top_k_val)
        if new_top_k < top_k and heuristic and heuristic.top_k:
            new_top_k = heuristic.top_k

    clear_client = bool(data.get("clear_client_id"))
    if heuristic and heuristic.clear_client_id:
        clear_client = True

    retrieval_query = data.get("retrieval_query")
    if retrieval_query is not None:
        retrieval_query = str(retrieval_query).strip() or None
    if not retrieval_query and heuristic and heuristic.retrieval_query:
        retrieval_query = heuristic.retrieval_query

    feedback = str(data.get("feedback") or "").strip()
    if heuristic and heuristic.feedback:
        if feedback:
            feedback = f"{feedback} | {heuristic.feedback}"
        else:
            feedback = heuristic.feedback

    return ImproveDecision(
        ok=False,
        reason=reason,
        action="retry",
        route=new_route,
        top_k=new_top_k,
        clear_client_id=clear_client,
        retrieval_query=retrieval_query,
        feedback=feedback or reason,
    )
