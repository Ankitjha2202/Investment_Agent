"""Database helpers (write + read-only connections)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector

from src.config import database_url, database_url_readonly


@contextmanager
def connect_rw() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(database_url(), connect_timeout=30)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


@contextmanager
def connect_ro() -> Iterator[psycopg.Connection]:
    """Connection intended for LLM-generated SQL only."""
    conn = psycopg.connect(database_url_readonly(), connect_timeout=30)
    try:
        # Session-level belt-and-suspenders (works even if role is not truly RO)
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = on")
            cur.execute("SET statement_timeout = '15000'")  # 15s
        register_vector(conn)
        yield conn
    finally:
        conn.close()
