"""Route a question to SQL, meeting RAG, or both."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from src.llm import chat_json

Route = Literal["sql", "meetings", "both"]
_CLIENT_ID_RE = re.compile(r"^[A-Za-z]\d{4,}$")


@dataclass
class RouteDecision:
    route: Route
    reason: str
    client_id: str | None = None


ROUTER_SYSTEM = """
You route natural-language questions about investment / client data.

Sources:
- sql: structured metrics & transactions in tables investments + performance
  (amounts, AUM, IRR, MOIC, deal counts, RM portfolios, dates, statuses)
- meetings: unstructured meeting notes (diligence discussion, action items,
  qualitative concerns, what was said in meetings)
- both: question needs numbers AND meeting narrative

Return JSON only:
{"route": "sql"|"meetings"|"both", "reason": "...", "client_id": "A12345"|null}

client_id rules:
- Only set client_id when the question contains an ID like A12345 / B12463 / C12355 (letter + digits).
- NEVER put a company name, RM name, or sector into client_id — use null instead.
- If unsure between sql and meetings, prefer "both".
""".strip()


def route_question(question: str) -> RouteDecision:
    raw = chat_json(ROUTER_SYSTEM, f"Question: {question}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return RouteDecision(route="both", reason="Router parse failed; defaulting to both")

    route = str(data.get("route", "both")).lower().strip()
    if route not in {"sql", "meetings", "both"}:
        route = "both"

    client_id = data.get("client_id")
    if client_id is not None:
        client_id = str(client_id).strip() or None
    # Reject non-ID values the model sometimes invents (company names, etc.)
    if client_id and not _CLIENT_ID_RE.match(client_id):
        client_id = None
    if not client_id:
        m = re.search(r"\b([A-Za-z]\d{4,})\b", question)
        if m:
            client_id = m.group(1)

    return RouteDecision(
        route=route,  # type: ignore[arg-type]
        reason=str(data.get("reason") or ""),
        client_id=client_id,
    )
