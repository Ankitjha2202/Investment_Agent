"""Public chat API — invoke the LangGraph agent and shape the UI response."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.graph import agent_graph
from src.retrieve import MeetingResult
from src.text_to_sql import SqlResult

StepCallback = Callable[[str, str, str], None]


@dataclass
class ChatResponse:
    answer: str
    route: str
    route_reason: str
    sql: SqlResult | None = None
    meetings: MeetingResult | None = None
    debug: dict[str, Any] = field(default_factory=dict)
    plan: list[dict[str, Any]] = field(default_factory=list)


def _serialize_sql(result: SqlResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "query": result.sql,
        "rationale": result.rationale,
        "error": result.error,
        "row_count": len(result.rows),
        "columns": result.columns,
        "sample_rows": result.rows[:10],
    }


def _serialize_meetings(result: MeetingResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "error": result.error,
        "hit_count": len(result.hits),
        "hits": [
            {
                "meeting_id": h.meeting_id,
                "score": round(h.score, 4),
                "meeting_date": h.meeting_date,
                "company": h.company,
                "sector": h.sector,
                "region": h.region,
                "client_id": h.client_id,
                "summary": (h.summary or "")[:280],
                "action_items": (h.action_items or "")[:180],
            }
            for h in result.hits
        ],
    }


def ask(
    question: str,
    *,
    show_debug: bool = True,
    on_step: StepCallback | None = None,
) -> ChatResponse:
    """Run the LangGraph agent (route → tools → synthesize → improve/heal)."""
    final = agent_graph.invoke(
        {
            "question": question,
            "retry_count": 0,
            "max_retries": 2,
            "improve_history": [],
            "quality_ok": False,
            "heal_action": "accept",
        },
        config={"configurable": {"on_step": on_step}},
    )

    sql_result = final.get("sql_result")
    meeting_result = final.get("meeting_result")
    route = final.get("route") or "both"
    route_reason = final.get("route_reason") or ""
    plan = final.get("plan") or []

    debug: dict[str, Any] = {}
    if show_debug:
        debug = {
            "route": route,
            "route_reason": route_reason,
            "top_k": final.get("top_k"),
            "client_id": final.get("client_id"),
            "plan": plan,
            "retry_count": final.get("retry_count") or 0,
            "max_retries": final.get("max_retries") or 2,
            "quality_ok": final.get("quality_ok"),
            "retrieval_query": final.get("retrieval_query"),
            "improve_feedback": final.get("improve_feedback"),
            "improve_history": final.get("improve_history") or [],
            "sql": _serialize_sql(sql_result),
            "meetings": _serialize_meetings(meeting_result),
        }

    return ChatResponse(
        answer=final.get("answer") or "",
        route=route,
        route_reason=route_reason,
        sql=sql_result,
        meetings=meeting_result,
        debug=debug,
        plan=plan,
    )
