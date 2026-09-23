"""End-to-end chat orchestration: route → retrieve/SQL → synthesize."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.retrieve import MeetingResult, retrieve_meetings
from src.router import RouteDecision, route_question
from src.synthesize import synthesize
from src.text_to_sql import SqlResult, answer_with_sql


@dataclass
class ChatResponse:
    answer: str
    route: str
    route_reason: str
    sql: SqlResult | None = None
    meetings: MeetingResult | None = None
    debug: dict[str, Any] = field(default_factory=dict)


def ask(question: str, *, show_debug: bool = True) -> ChatResponse:
    decision: RouteDecision = route_question(question)

    sql_result: SqlResult | None = None
    meeting_result: MeetingResult | None = None

    if decision.route in {"sql", "both"}:
        sql_result = answer_with_sql(question)

    if decision.route in {"meetings", "both"}:
        meeting_result = retrieve_meetings(
            question, top_k=6, client_id=decision.client_id
        )

    answer = synthesize(
        question,
        route=decision.route,
        sql_result=sql_result,
        meeting_result=meeting_result,
    )

    debug: dict[str, Any] = {}
    if show_debug:
        debug = {
            "route": decision.route,
            "route_reason": decision.reason,
            "client_id": decision.client_id,
            "sql": sql_result.sql if sql_result else None,
            "sql_error": sql_result.error if sql_result else None,
            "sql_rows": len(sql_result.rows) if sql_result else 0,
            "meeting_hits": len(meeting_result.hits) if meeting_result else 0,
            "meeting_error": meeting_result.error if meeting_result else None,
        }

    return ChatResponse(
        answer=answer,
        route=decision.route,
        route_reason=decision.reason,
        sql=sql_result,
        meetings=meeting_result,
        debug=debug,
    )
