"""
SQL guardrails for LLM-generated queries.

Rules:
  1. Parse with sqlglot — must be a single SELECT (or WITH…SELECT).
  2. Reject any write/DDL/admin statements.
  3. Only allow whitelisted tables.
  4. Inject/enforce a LIMIT if missing.
  5. Execution happens on DATABASE_URL_READONLY with read_only + statement_timeout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

ALLOWED_TABLES = {"investments", "performance"}
DEFAULT_LIMIT = 100
MAX_LIMIT = 500

# Keywords that must never appear even if parser is bypassed
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|"
    r"COPY|EXECUTE|CALL|MERGE|REPLACE|ATTACH|DETACH|VACUUM|ANALYZE|"
    r"SET\s+ROLE|SET\s+SESSION|pg_sleep|lo_import|dblink)\b",
    re.IGNORECASE,
)


@dataclass
class GuardResult:
    ok: bool
    sql: str | None = None
    error: str | None = None


def _collect_tables(parsed: exp.Expression) -> set[str]:
    tables: set[str] = set()
    for node in parsed.find_all(exp.Table):
        name = node.name
        if name:
            tables.add(name.lower())
    return tables


def _collect_cte_names(parsed: exp.Expression) -> set[str]:
    names: set[str] = set()
    for cte in parsed.find_all(exp.CTE):
        # CTE alias
        alias = cte.alias
        if alias:
            names.add(alias.lower())
    return names


def _ensure_limit(select: exp.Select) -> exp.Select:
    existing = select.args.get("limit")
    if existing is None:
        return select.limit(DEFAULT_LIMIT)
    # Cap absurd limits
    try:
        n = int(existing.expression.this) if existing.expression else DEFAULT_LIMIT
        if n > MAX_LIMIT:
            return select.limit(MAX_LIMIT)
    except Exception:
        return select.limit(DEFAULT_LIMIT)
    return select


def validate_sql(raw_sql: str) -> GuardResult:
    if not raw_sql or not raw_sql.strip():
        return GuardResult(False, error="Empty SQL")

    cleaned = raw_sql.strip().rstrip(";")
    # Strip markdown fences if model wraps output
    cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip().rstrip(";")

    if _FORBIDDEN.search(cleaned):
        return GuardResult(False, error="Forbidden keyword detected")

    if ";" in cleaned:
        return GuardResult(False, error="Multiple statements are not allowed")

    try:
        statements = sqlglot.parse(cleaned, read="postgres")
    except sqlglot.errors.ParseError as e:
        return GuardResult(False, error=f"Parse error: {e}")

    if len(statements) != 1 or statements[0] is None:
        return GuardResult(False, error="Exactly one SQL statement required")

    stmt = statements[0]

    # Allow SELECT or WITH…SELECT only
    if isinstance(stmt, exp.Select):
        select = stmt
    elif isinstance(stmt, exp.With) and isinstance(stmt.this, exp.Select):
        select = stmt
    else:
        # CTE wrapped differently
        if not isinstance(stmt, (exp.Select, exp.With)):
            return GuardResult(False, error=f"Only SELECT queries allowed (got {type(stmt).__name__})")
        select = stmt

    # Disallow SELECT INTO / FOR UPDATE etc.
    if select.find(exp.Into):
        return GuardResult(False, error="SELECT INTO is not allowed")

    tables = _collect_tables(stmt)
    cte_names = _collect_cte_names(stmt)
    physical = tables - cte_names
    if not physical:
        return GuardResult(False, error="No tables referenced")
    unknown = physical - ALLOWED_TABLES
    if unknown:
        return GuardResult(
            False,
            error=f"Table(s) not allowed: {sorted(unknown)}. Allowed: {sorted(ALLOWED_TABLES)}",
        )

    # Apply limit on the outermost select
    outer = stmt.this if isinstance(stmt, exp.With) else stmt
    if isinstance(outer, exp.Select):
        limited = _ensure_limit(outer)
        if isinstance(stmt, exp.With):
            stmt = stmt.set("this", limited)
        else:
            stmt = limited

    final_sql = stmt.sql(dialect="postgres")
    return GuardResult(True, sql=final_sql)
