"""Assessor brain — decide route, retrieval depth, and optional client filter."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from src.llm import chat_json

Route = Literal["sql", "meetings", "both"]
_CLIENT_ID_RE = re.compile(r"^[A-Za-z]\d{4,}$")

MIN_TOP_K = 3
MAX_TOP_K = 20
DEFAULT_TOP_K = 6


@dataclass
class AssessDecision:
    """Downstream plan produced by the assessor."""

    route: Route
    reason: str
    top_k: int = DEFAULT_TOP_K
    client_id: str | None = None


ASSESS_SYSTEM = """
You are the planning brain for an investment analytics agent.

Decide what evidence to gather BEFORE tools run. Downstream nodes follow you.

Sources:
- sql: structured metrics & transactions in tables investments + performance
  (amounts, AUM, IRR, MOIC, deal counts, RM portfolios, dates, statuses)
- meetings: unstructured meeting notes (diligence discussion, action items,
  qualitative concerns, who met whom, meeting dates, narrative)
- both: question needs numbers AND meeting narrative

Also choose top_k = how many meeting-note chunks to retrieve (ignored for sql-only).

top_k guidance:
- 3–5: narrow / single-meeting / one company or one event
- 6–8: typical thematic question (“fintech diligence in EMEA”)
- 10–15: one person/RM/client with many counterparties, “list all meetings”,
  “everyone they met”, “details and dates” across a portfolio of notes
- 16–20: exhaustive listing across many people/meetings when the user clearly
  wants breadth (“all”, “every”, “complete list”, “a lot of people”)

Return JSON only:
{
  "route": "sql"|"meetings"|"both",
  "top_k": <integer 3-20>,
  "reason": "short justification covering route AND retrieval depth",
  "client_id": "A12345"|null
}

client_id rules:
- Only set client_id when the question contains an ID like A12345 / B12463 (letter + digits).
- NEVER put a company name, RM name, or sector into client_id — use null instead.
- If unsure between sql and meetings, prefer "both".
- For sql-only, still return a sensible top_k (e.g. 6); it will be unused.
""".strip()


def _clamp_top_k(value: object) -> int:
    try:
        k = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_TOP_K
    return max(MIN_TOP_K, min(MAX_TOP_K, k))


def assess_question(
    question: str,
    *,
    feedback: str | None = None,
    prior_route: str | None = None,
    prior_top_k: int | None = None,
) -> AssessDecision:
    """Plan route + meeting retrieval depth for the agent pipeline.

    Optional feedback / prior plan are used on self-heal retries so the brain
    can replan instead of repeating a failed strategy.
    """
    user = f"Question: {question}"
    if feedback:
        user += (
            "\n\nPrevious attempt was insufficient. Critic feedback:\n"
            f"{feedback}\n"
            f"Prior route={prior_route!r}, prior top_k={prior_top_k!r}.\n"
            "Replan: prefer expanding sources, raising top_k, and fixing filters."
        )
    raw = chat_json(ASSESS_SYSTEM, user)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # On heal retries, bias toward the safer hybrid plan.
        return AssessDecision(
            route="both",
            reason="Assessor parse failed; defaulting to both with top_k=6",
            top_k=max(prior_top_k or DEFAULT_TOP_K, DEFAULT_TOP_K),
        )

    route = str(data.get("route", "both")).lower().strip()
    if route not in {"sql", "meetings", "both"}:
        route = "both"

    top_k = _clamp_top_k(data.get("top_k", DEFAULT_TOP_K))
    # SQL-only never needs notes; keep a default for state consistency.
    if route == "sql":
        top_k = DEFAULT_TOP_K

    client_id = data.get("client_id")
    if client_id is not None:
        client_id = str(client_id).strip() or None
    if client_id and not _CLIENT_ID_RE.match(client_id):
        client_id = None
    if not client_id:
        m = re.search(r"\b([A-Za-z]\d{4,})\b", question)
        if m:
            client_id = m.group(1)

    return AssessDecision(
        route=route,  # type: ignore[arg-type]
        reason=str(data.get("reason") or ""),
        top_k=top_k,
        client_id=client_id,
    )
