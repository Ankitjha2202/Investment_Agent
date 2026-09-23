"""Shared configuration loaded from .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def database_url() -> str:
    """Read-write URL — use for ETL and migrations only."""
    return _require("DATABASE_URL")


def database_url_readonly() -> str:
    """
    Read-only URL — use for executing LLM-generated SQL.
    Falls back to DATABASE_URL if DATABASE_URL_READONLY is unset (dev only).
    """
    return os.getenv("DATABASE_URL_READONLY") or _require("DATABASE_URL")


def openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")


XLSX_PATH = ROOT / "Assignment_Data.xlsx"
SCHEMA_STRUCTURED_SQL = ROOT / "sql" / "schema_structured.sql"
SCHEMA_MEETINGS_SQL = ROOT / "sql" / "schema_meetings.sql"
