"""Synthesize a final natural-language answer from tool outputs."""

from __future__ import annotations

import json
from typing import Any

from src.llm import chat_text
from src.retrieve import MeetingResult
from src.text_to_sql import SqlResult

SYNTHESIS_SYSTEM = """
You are an investment analytics assistant. Answer the user's question using ONLY
the provided evidence (SQL results and/or meeting notes). Be concise and specific.

Rules:
- Cite figures clearly (amounts, IRR, dates, client/deal/RM names).
- If meeting notes are used, mention company / date when relevant.
- If evidence is missing or empty, say what you couldn't find — do not invent numbers.
- If SQL errored, acknowledge it briefly and use whatever else is available.
- Prefer short paragraphs or bullet points over long essays.
""".strip()


def synthesize(
    question: str,
    *,
    route: str,
    sql_result: SqlResult | None,
    meeting_result: MeetingResult | None,
) -> str:
    payload: dict[str, Any] = {"question": question, "route": route}

    if sql_result is not None:
        payload["sql"] = {
            "query": sql_result.sql,
            "rationale": sql_result.rationale,
            "error": sql_result.error,
            "row_count": len(sql_result.rows),
            "rows": sql_result.rows[:50],  # cap payload size
        }

    if meeting_result is not None:
        payload["meetings"] = {
            "error": meeting_result.error,
            "hit_count": len(meeting_result.hits),
            "context": meeting_result.as_context(),
        }

    user = (
        "Evidence (JSON):\n"
        + json.dumps(payload, default=str)[:14000]
        + "\n\nWrite the final answer for the user."
    )
    return chat_text(SYNTHESIS_SYSTEM, user)
