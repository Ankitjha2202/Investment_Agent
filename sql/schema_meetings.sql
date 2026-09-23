-- Meeting notes + pgvector (run after structured schema)

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS meeting_notes CASCADE;
CREATE TABLE meeting_notes (
    meeting_id                  INTEGER PRIMARY KEY,
    meeting_date                DATE,
    attendees                   TEXT,
    company                     TEXT,
    sector                      TEXT,
    region                      TEXT,
    investment_stage            TEXT,
    deal_size_estimate          TEXT,
    summary                     TEXT,
    action_items                TEXT,
    summary_char_length         INTEGER,
    client_id                   TEXT,
    group_id                    INTEGER,
    embed_text                  TEXT NOT NULL,
    embedding                   vector(1536)
);

CREATE INDEX idx_meeting_notes_client_id ON meeting_notes (client_id);
CREATE INDEX idx_meeting_notes_group_id ON meeting_notes (group_id);
CREATE INDEX idx_meeting_notes_date ON meeting_notes (meeting_date);
CREATE INDEX idx_meeting_notes_sector ON meeting_notes (sector);
CREATE INDEX idx_meeting_notes_company ON meeting_notes (company);

-- HNSW index created after embeddings are loaded (see etl/load_meetings.py)
