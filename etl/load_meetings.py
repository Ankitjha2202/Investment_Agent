"""
Load meeting_notes from Excel and embed with text-embedding-3-small (fast path).

Speed strategy:
  - Large embedding batches (512)
  - Concurrent OpenAI requests (thread pool — I/O bound)
  - Embed fully in memory, then single bulk COPY into Postgres
  - Create HNSW index once at the end

Usage:
  python -m etl.load_meetings
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import XLSX_PATH, database_url  # noqa: E402
from src.llm import embed_texts, get_client  # noqa: E402

SCHEMA_SQL = ROOT / "sql" / "schema_meetings.sql"

# Tunables — stay under ~5M TPM while maximizing throughput
EMBED_BATCH_SIZE = 256
EMBED_WORKERS = 8


def build_embed_text(row: pd.Series) -> str:
    return (
        f"company: {row.get('company') or ''}\n"
        f"sector: {row.get('sector') or ''}\n"
        f"region: {row.get('region') or ''}\n"
        f"investment_stage: {row.get('investment_stage') or ''}\n"
        f"deal_size_estimate: {row.get('deal_size_estimate') or ''}\n"
        f"client_id: {row.get('client_id') or ''}\n"
        f"group_id: {row.get('group_id') or ''}\n"
        f"attendees: {row.get('attendees') or ''}\n"
        f"date: {row.get('meeting_date')}\n"
        f"summary: {row.get('summary') or ''}\n"
        f"action_items: {row.get('action_items') or ''}"
    )


def apply_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL.read_text())
    conn.commit()
    print(f"Applied {SCHEMA_SQL.name}")


def _embed_batch(args: tuple[int, list[str]]) -> tuple[int, list[list[float]]]:
    """Worker: embed one batch. Returns (batch_index, vectors)."""
    batch_idx, texts = args
    client = get_client()  # lightweight; each thread gets its own client
    vectors = embed_texts(texts, client=client)
    return batch_idx, vectors


def embed_all_parallel(texts: list[str]) -> list[list[float]]:
    batches: list[tuple[int, list[str]]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batches.append((i // EMBED_BATCH_SIZE, texts[i : i + EMBED_BATCH_SIZE]))

    print(
        f"Embedding {len(texts):,} notes — "
        f"{len(batches)} batches × up to {EMBED_BATCH_SIZE}, "
        f"{EMBED_WORKERS} parallel workers …"
    )

    results: dict[int, list[list[float]]] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        futures = [pool.submit(_embed_batch, b) for b in batches]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="embed"):
            idx, vectors = fut.result()
            results[idx] = vectors

    # Reassemble in order
    ordered: list[list[float]] = []
    for i in range(len(batches)):
        ordered.extend(results[i])

    elapsed = time.time() - t0
    print(f"  embeddings done in {elapsed:.1f}s ({len(texts) / elapsed:.0f} docs/s)")
    return ordered


def _csv_cell(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value)
    if any(ch in s for ch in [",", '"', "\n", "\r"]):
        s = '"' + s.replace('"', '""') + '"'
    return s


def _vector_literal(vec: list[float]) -> str:
    # pgvector accepts '[1,2,3]' — quote for CSV
    inner = ",".join(f"{x:.8f}" for x in vec)
    return '"' + f"[{inner}]" + '"'


def bulk_copy(conn: psycopg.Connection, df: pd.DataFrame, vectors: list[list[float]]) -> None:
    """One-shot COPY of all rows including embeddings."""
    assert len(df) == len(vectors)
    cols = [
        "meeting_id",
        "meeting_date",
        "attendees",
        "company",
        "sector",
        "region",
        "investment_stage",
        "deal_size_estimate",
        "summary",
        "action_items",
        "summary_char_length",
        "client_id",
        "group_id",
        "embed_text",
        "embedding",
    ]

    print(f"COPY {len(df):,} rows into meeting_notes …")
    t0 = time.time()
    lines: list[str] = []
    for (_, r), vec in zip(df.iterrows(), vectors):
        meeting_date = (
            r["meeting_date"].strftime("%Y-%m-%d")
            if pd.notna(r["meeting_date"])
            else ""
        )
        summary_len = (
            int(r["summary_char_length"])
            if pd.notna(r.get("summary_char_length"))
            else ""
        )
        group_id = int(r["group_id"]) if pd.notna(r.get("group_id")) else ""
        cells = [
            _csv_cell(int(r["meeting_id"])),
            _csv_cell(meeting_date),
            _csv_cell(r.get("attendees")),
            _csv_cell(r.get("company")),
            _csv_cell(r.get("sector")),
            _csv_cell(r.get("region")),
            _csv_cell(r.get("investment_stage")),
            _csv_cell(r.get("deal_size_estimate")),
            _csv_cell(r.get("summary")),
            _csv_cell(r.get("action_items")),
            _csv_cell(summary_len),
            _csv_cell(r.get("client_id")),
            _csv_cell(group_id),
            _csv_cell(r["embed_text"]),
            _vector_literal(vec),
        ]
        lines.append(",".join(cells))

    payload = ("\n".join(lines) + "\n").encode("utf-8")
    col_list = sql.SQL(", ").join(sql.Identifier(c) for c in cols)
    copy_sql = sql.SQL(
        "COPY {} ({}) FROM STDIN WITH (FORMAT CSV, NULL '')"
    ).format(sql.Identifier("meeting_notes"), col_list)

    with conn.cursor() as cur:
        with cur.copy(copy_sql) as copy:
            copy.write(payload)
    conn.commit()
    print(f"  COPY done in {time.time() - t0:.1f}s")


def create_hnsw_index(conn: psycopg.Connection) -> None:
    print("Creating HNSW index on embedding …")
    t0 = time.time()
    with conn.cursor() as cur:
        # Supabase pooler default timeout is too low for 20k-vector HNSW builds
        cur.execute("SET statement_timeout = '600000'")  # 10 min
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_meeting_notes_embedding_hnsw
            ON meeting_notes
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )
        cur.execute("SET statement_timeout = '60s'")
    conn.commit()
    print(f"  HNSW ready in {time.time() - t0:.1f}s")


def main() -> None:
    t_all = time.time()
    print("Reading meeting_notes_20k …")
    df = pd.read_excel(XLSX_PATH, sheet_name="meeting_notes_20k")
    df = df.rename(columns={"date": "meeting_date"})
    df["meeting_date"] = pd.to_datetime(df["meeting_date"], errors="coerce")
    print("Building embed_text …")
    df["embed_text"] = df.apply(build_embed_text, axis=1)
    texts = df["embed_text"].tolist()

    vectors = embed_all_parallel(texts)

    url = database_url()
    print(f"Connecting to {url.split('@')[-1].split('?')[0]} …")
    with psycopg.connect(url, connect_timeout=60) as conn:
        register_vector(conn)
        apply_schema(conn)
        bulk_copy(conn, df, vectors)
        create_hnsw_index(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM meeting_notes WHERE embedding IS NOT NULL")
            n = cur.fetchone()[0]
    print(f"Done — {n:,} meeting notes embedded in {time.time() - t_all:.1f}s total")


if __name__ == "__main__":
    main()
