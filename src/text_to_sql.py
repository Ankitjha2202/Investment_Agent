"""Text-to-SQL over investments + performance tables."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.db import connect_ro
from src.llm import chat_json
from src.sql_guardrails import validate_sql

SCHEMA_PROMPT = """
You are a Postgres SQL expert for an investment analytics database.

Tables (snake_case columns):

1) investments — one row per capital call / investment event (~50k rows)
   client_id TEXT, client_name TEXT, client_group_id INT, deal_id TEXT,
   capital_call_id INT, investment_amount_usd FLOAT, investment_amount_natural FLOAT,
   natural_currency_code TEXT, investment_exchange_rate FLOAT, invested_date DATE,
   deal_name TEXT, lob_code TEXT, lob_name TEXT, realised BOOLEAN,
   client_status TEXT, account_name TEXT, account_rm TEXT, rm_alias TEXT,
   account_rm_email TEXT, account_owner_id TEXT
   Notes: Use investment_amount_usd for USD aggregates. RM = account_rm or rm_alias.
   LoB codes include COP, RE, etc. realised indicates realised investments.

2) performance — client / group performance snapshots (~1.6k rows)
   client_group_id INT, client_id TEXT, client_name TEXT, is_group BOOLEAN,
   total_aum_amount FLOAT, ci_aum_amount FLOAT, re_aum_amount FLOAT, hf_aum_amount FLOAT,
   ci_current_irr FLOAT, ci_total_irr FLOAT, ci_realised_irr FLOAT,
   ci_current_moic TEXT, ci_total_moic TEXT,  -- MOIC stored as text like '2.3x'
   re_current_irr FLOAT, re_total_irr FLOAT, re_core_irr FLOAT,
   cop_current_irr FLOAT, cop_total_irr FLOAT,
   as_of_date DATE, client_last_met_date DATE,
   investment_status_name TEXT, ci_status_name TEXT, re_status_name TEXT,
   last_ci_investment_name TEXT, last_ci_investment_amount FLOAT,
   last_re_investment_name TEXT, product_count_number INT
   Notes: Filter is_group = false for client-level rows, is_group = true for group rollups.
   Multiple snapshot rows may exist per client — prefer latest as_of_date when aggregating.

Rules:
- Output ONLY a JSON object: {"sql": "<single SELECT>", "rationale": "<brief>"}
- SELECT / WITH…SELECT only. Never modify data.
- Only query investments and/or performance.
- Prefer aggregates (SUM/AVG/COUNT) over dumping raw rows.
- Always include a LIMIT (<= 100) for non-aggregate detail queries.
- Use ILIKE for fuzzy name matching when helpful.
""".strip()


@dataclass
class SqlResult:
    sql: str | None = None
    rationale: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    error: str | None = None


def _parse_sql_response(raw: str) -> tuple[str | None, str | None, str | None]:
    """Parse model JSON into (sql, rationale, error)."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, None, f"Model returned non-JSON: {raw[:200]}"
    sql = data.get("sql")
    rationale = data.get("rationale")
    if not sql:
        return None, rationale, "No sql field in model response"
    return str(sql), (str(rationale) if rationale is not None else None), None


def generate_sql(
    question: str,
    *,
    feedback: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Returns (sql, rationale, error)."""
    user = f"Question: {question}"
    if feedback:
        user += (
            "\n\nPipeline critic feedback from a prior attempt — apply these fixes:\n"
            f"{feedback}"
        )
    raw = chat_json(SCHEMA_PROMPT, user)
    return _parse_sql_response(raw)


def repair_sql(
    question: str,
    *,
    failed_sql: str | None,
    error: str,
) -> tuple[str | None, str | None, str | None]:
    """Ask the model to fix a failed / empty query. Returns (sql, rationale, error)."""
    user = (
        f"Question: {question}\n\n"
        f"Previous SQL:\n{failed_sql or '(none generated)'}\n\n"
        f"Problem: {error}\n\n"
        "Fix the query. Keep the same JSON shape: "
        '{"sql": "<single SELECT>", "rationale": "<what you changed>"}'
    )
    raw = chat_json(SCHEMA_PROMPT, user)
    return _parse_sql_response(raw)


def execute_sql(sql: str) -> SqlResult:
    guard = validate_sql(sql)
    if not guard.ok:
        return SqlResult(sql=sql, error=f"Guardrail rejected SQL: {guard.error}")

    safe_sql = guard.sql
    assert safe_sql is not None
    try:
        with connect_ro() as conn:
            with conn.cursor() as cur:
                cur.execute(safe_sql)
                cols = [d.name for d in cur.description] if cur.description else []
                fetched = cur.fetchall()
                rows = [dict(zip(cols, row)) for row in fetched]
        return SqlResult(sql=safe_sql, rows=rows, columns=cols)
    except Exception as e:
        return SqlResult(sql=safe_sql, error=f"Execution error: {e}")


def _needs_repair(result: SqlResult) -> str | None:
    """Return a repair hint if the result should be retried, else None."""
    if result.error:
        return result.error
    if not result.rows:
        return (
            "Query returned 0 rows. If the question implies matching data exists, "
            "fix filters, joins, column names, or date/status predicates."
        )
    return None


def answer_with_sql(
    question: str,
    *,
    max_attempts: int = 2,
    feedback: str | None = None,
) -> SqlResult:
    """Generate + execute SQL, with one self-correction retry on failure/empty.

    ``feedback`` is optional critic guidance from the graph-level heal loop.
    ``max_attempts`` caps generate+repair cycles (2 ⇒ one repair pass).
    """
    sql, rationale, err = generate_sql(question, feedback=feedback)
    if err or not sql:
        if max_attempts < 2:
            return SqlResult(rationale=rationale, error=err or "SQL generation failed")
        repair_err = err or "SQL generation failed"
        if feedback:
            repair_err = f"{repair_err}\nCritic feedback: {feedback}"
        sql, rationale, err = repair_sql(
            question,
            failed_sql=None,
            error=repair_err,
        )
        if err or not sql:
            return SqlResult(rationale=rationale, error=err or "SQL generation failed")

    result = execute_sql(sql)
    result.rationale = rationale

    hint = _needs_repair(result)
    if not hint or max_attempts < 2:
        return result

    if feedback:
        hint = f"{hint}\nCritic feedback: {feedback}"

    sql2, rationale2, err2 = repair_sql(
        question,
        failed_sql=result.sql or sql,
        error=hint,
    )
    if err2 or not sql2:
        return result

    repaired = execute_sql(sql2)
    note = rationale2 or rationale or ""
    if note and "repair" not in note.lower():
        note = f"{note} (repaired after: {hint[:120]})"
    elif not note:
        note = f"Repaired after: {hint[:120]}"
    repaired.rationale = note

    # Prefer a successful repair. If repair fails but the first attempt only
    # returned empty rows (no error), keep the first valid empty result.
    if not repaired.error:
        return repaired
    if result.error:
        return repaired
    return result
