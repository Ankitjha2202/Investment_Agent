"""Meeting-notes semantic retrieval via pgvector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.db import connect_ro
from src.llm import embed_texts


@dataclass
class MeetingHit:
    meeting_id: int
    score: float
    meeting_date: str | None
    company: str | None
    sector: str | None
    region: str | None
    client_id: str | None
    group_id: int | None
    summary: str | None
    action_items: str | None


@dataclass
class MeetingResult:
    hits: list[MeetingHit] = field(default_factory=list)
    error: str | None = None

    def as_context(self, max_chars: int = 6000) -> str:
        parts: list[str] = []
        used = 0
        for h in self.hits:
            block = (
                f"[meeting_id={h.meeting_id} | date={h.meeting_date} | "
                f"company={h.company} | sector={h.sector} | region={h.region} | "
                f"client_id={h.client_id} | score={h.score:.3f}]\n"
                f"Summary: {h.summary or ''}\n"
                f"Actions: {h.action_items or ''}\n"
            )
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n---\n".join(parts)


def retrieve_meetings(
    question: str,
    *,
    top_k: int = 6,
    client_id: str | None = None,
) -> MeetingResult:
    try:
        qvec = embed_texts([question])[0]
    except Exception as e:
        return MeetingResult(error=f"Embedding failed: {e}")

    # Cosine distance (<=>): lower is better. Expose similarity = 1 - distance.
    if client_id:
        params: list[Any] = [qvec, client_id, qvec, top_k]
        sql = """
            SELECT
                meeting_id,
                meeting_date::text,
                company,
                sector,
                region,
                client_id,
                group_id,
                summary,
                action_items,
                1 - (embedding <=> %s::vector) AS score
            FROM meeting_notes
            WHERE embedding IS NOT NULL AND client_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
    else:
        params = [qvec, qvec, top_k]
        sql = """
            SELECT
                meeting_id,
                meeting_date::text,
                company,
                sector,
                region,
                client_id,
                group_id,
                summary,
                action_items,
                1 - (embedding <=> %s::vector) AS score
            FROM meeting_notes
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

    try:
        with connect_ro() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        hits = [
            MeetingHit(
                meeting_id=r[0],
                meeting_date=r[1],
                company=r[2],
                sector=r[3],
                region=r[4],
                client_id=r[5],
                group_id=r[6],
                summary=r[7],
                action_items=r[8],
                score=float(r[9]) if r[9] is not None else 0.0,
            )
            for r in rows
        ]
        return MeetingResult(hits=hits)
    except Exception as e:
        return MeetingResult(error=f"Retrieval failed: {e}")
