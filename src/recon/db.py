"""Knowledge database for Recon.

SQLite-backed persistence layer for runs, claims, sources, events,
and token-usage metrics.  Uses FTS5 for keyword search over claims.

All functions accept a ``sqlite3.Connection`` to keep I/O explicit and
testable — no hidden global state.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------

SCHEMA_VERSION = 1

_SCHEMA_SQL = """\
-- schema version tracking
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ============================================================
-- RUNS & PIPELINE METRICS
-- ============================================================

CREATE TABLE IF NOT EXISTS runs (
    id                TEXT PRIMARY KEY,
    timestamp         TEXT NOT NULL,
    topic             TEXT NOT NULL,
    depth             TEXT NOT NULL,
    model             TEXT,
    provider          TEXT,
    search_provider   TEXT,
    verify            INTEGER NOT NULL DEFAULT 1,
    auto_questions    INTEGER NOT NULL DEFAULT 1,
    status            TEXT NOT NULL DEFAULT 'running',
    duration_seconds  REAL,
    config_json       TEXT
);

CREATE TABLE IF NOT EXISTS phase_metrics (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL REFERENCES runs(id),
    phase            TEXT NOT NULL,
    status           TEXT NOT NULL,
    started_at       TEXT,
    ended_at         TEXT,
    duration_seconds REAL,
    output_files     TEXT,
    metadata_json    TEXT
);

CREATE TABLE IF NOT EXISTS token_usage (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL REFERENCES runs(id),
    phase               TEXT NOT NULL,
    agent               TEXT,
    prompt_tokens       INTEGER NOT NULL DEFAULT 0,
    completion_tokens   INTEGER NOT NULL DEFAULT 0,
    total_tokens        INTEGER NOT NULL DEFAULT 0,
    cached_tokens       INTEGER NOT NULL DEFAULT 0,
    successful_requests INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd  REAL
);

-- ============================================================
-- KNOWLEDGE: CLAIMS + SOURCES + HISTORY
-- ============================================================

CREATE TABLE IF NOT EXISTS claims (
    id                  TEXT PRIMARY KEY,
    text                TEXT NOT NULL,
    claim_type          TEXT,
    source_document     TEXT,
    cited_source        TEXT,
    topic_tags          TEXT,
    entity_tags         TEXT,
    verification_status TEXT,
    confidence          REAL,
    evidence_excerpt    TEXT,
    first_seen_run      TEXT REFERENCES runs(id),
    first_seen_at       TEXT,
    last_verified_at    TEXT,
    last_verified_run   TEXT REFERENCES runs(id),
    times_seen          INTEGER NOT NULL DEFAULT 1,
    times_verified      INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS claim_sources (
    claim_id         TEXT NOT NULL REFERENCES claims(id),
    source_url       TEXT NOT NULL,
    domain           TEXT,
    support_type     TEXT,
    evidence_excerpt TEXT,
    retrieved_at     TEXT,
    http_status      INTEGER,
    content_hash     TEXT,
    PRIMARY KEY (claim_id, source_url)
);

CREATE TABLE IF NOT EXISTS claim_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id       TEXT NOT NULL REFERENCES claims(id),
    run_id         TEXT NOT NULL REFERENCES runs(id),
    verified_at    TEXT NOT NULL,
    old_status     TEXT,
    new_status     TEXT,
    old_confidence REAL,
    new_confidence REAL,
    method         TEXT NOT NULL
);

-- ============================================================
-- SOURCES (global, cross-run)
-- ============================================================

CREATE TABLE IF NOT EXISTS sources (
    url              TEXT PRIMARY KEY,
    domain           TEXT NOT NULL,
    title            TEXT,
    first_seen_run   TEXT REFERENCES runs(id),
    first_seen_at    TEXT,
    last_fetched_at  TEXT,
    times_cited      INTEGER NOT NULL DEFAULT 1,
    content_hash     TEXT,
    reliability_score REAL
);

-- ============================================================
-- EVENTS & TRACEABILITY
-- ============================================================

CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES runs(id),
    timestamp     TEXT NOT NULL,
    phase         TEXT,
    agent         TEXT,
    task          TEXT,
    action        TEXT NOT NULL,
    detail        TEXT,
    metadata_json TEXT,
    tokens_used   INTEGER
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_events_run    ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_phase  ON events(run_id, phase);
CREATE INDEX IF NOT EXISTS idx_events_action ON events(action);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(verification_status);
CREATE INDEX IF NOT EXISTS idx_claims_run    ON claims(first_seen_run);
CREATE INDEX IF NOT EXISTS idx_claims_conf   ON claims(confidence);
CREATE INDEX IF NOT EXISTS idx_phase_run     ON phase_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_token_run     ON token_usage(run_id);

-- FTS5 for keyword search over claims
CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
    text, topic_tags, entity_tags,
    content=claims, content_rowid=rowid
);

-- Triggers to keep FTS5 in sync with claims table
CREATE TRIGGER IF NOT EXISTS claims_ai AFTER INSERT ON claims BEGIN
    INSERT INTO claims_fts(rowid, text, topic_tags, entity_tags)
    VALUES (new.rowid, new.text, new.topic_tags, new.entity_tags);
END;

CREATE TRIGGER IF NOT EXISTS claims_ad AFTER DELETE ON claims BEGIN
    INSERT INTO claims_fts(claims_fts, rowid, text, topic_tags, entity_tags)
    VALUES ('delete', old.rowid, old.text, old.topic_tags, old.entity_tags);
END;

CREATE TRIGGER IF NOT EXISTS claims_au AFTER UPDATE ON claims BEGIN
    INSERT INTO claims_fts(claims_fts, rowid, text, topic_tags, entity_tags)
    VALUES ('delete', old.rowid, old.text, old.topic_tags, old.entity_tags);
    INSERT INTO claims_fts(rowid, text, topic_tags, entity_tags)
    VALUES (new.rowid, new.text, new.topic_tags, new.entity_tags);
END;
"""

# ------------------------------------------------------------------
# Connection helpers
# ------------------------------------------------------------------

DEFAULT_DB_PATH = "./knowledge.db"


def get_db(path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (or create) the knowledge database and return a connection.

    Enables WAL mode for concurrent reads and foreign-key enforcement.
    Calls :func:`init_db` automatically on first creation.
    """
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not db_path.exists()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    if is_new:
        init_db(conn)
    else:
        _maybe_migrate(conn)

    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables, indexes, triggers and set schema version."""
    conn.executescript(_SCHEMA_SQL)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )
    conn.commit()


def _maybe_migrate(conn: sqlite3.Connection) -> None:
    """Run forward migrations when the on-disk schema is older."""
    try:
        row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
        version = int(row["value"]) if row else 0
    except sqlite3.OperationalError:
        # schema_meta table doesn't exist → legacy DB, reinitialise
        version = 0

    if version < SCHEMA_VERSION:
        init_db(conn)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _json_dumps(obj: Any) -> str | None:
    """Serialize to JSON string, or *None* if *obj* is None."""
    return json.dumps(obj, default=str) if obj is not None else None


def _json_loads(raw: str | None) -> Any:
    """Deserialize from JSON string, or *None* if *raw* is None."""
    return json.loads(raw) if raw else None


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    """Convert a sqlite3.Row to a plain dict, or return *None*."""
    if row is None:
        return None
    return dict(row)


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


# ------------------------------------------------------------------
# CRUD — Runs
# ------------------------------------------------------------------


def insert_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    timestamp: str,
    topic: str,
    depth: str,
    model: str | None = None,
    provider: str | None = None,
    search_provider: str | None = None,
    verify: bool = True,
    auto_questions: bool = True,
    config_json: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        """INSERT INTO runs
           (id, timestamp, topic, depth, model, provider, search_provider,
            verify, auto_questions, config_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            timestamp,
            topic,
            depth,
            model,
            provider,
            search_provider,
            int(verify),
            int(auto_questions),
            _json_dumps(config_json),
        ),
    )
    conn.commit()


def update_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    status: str | None = None,
    duration_seconds: float | None = None,
) -> None:
    parts: list[str] = []
    vals: list[Any] = []
    if status is not None:
        parts.append("status = ?")
        vals.append(status)
    if duration_seconds is not None:
        parts.append("duration_seconds = ?")
        vals.append(duration_seconds)
    if not parts:
        return
    vals.append(run_id)
    conn.execute(f"UPDATE runs SET {', '.join(parts)} WHERE id = ?", vals)
    conn.commit()


def get_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return _row_to_dict(row)


def get_runs(
    conn: sqlite3.Connection,
    *,
    limit: int = 10,
    topic: str | None = None,
) -> list[dict[str, Any]]:
    if topic:
        rows = conn.execute(
            "SELECT * FROM runs WHERE topic LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{topic}%", limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return _rows_to_dicts(rows)


# ------------------------------------------------------------------
# CRUD — Phase Metrics
# ------------------------------------------------------------------


def insert_phase_metric(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    phase: str,
    status: str,
    started_at: str | None = None,
    ended_at: str | None = None,
    duration_seconds: float | None = None,
    output_files: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO phase_metrics
           (run_id, phase, status, started_at, ended_at, duration_seconds,
            output_files, metadata_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            phase,
            status,
            started_at,
            ended_at,
            duration_seconds,
            _json_dumps(output_files),
            _json_dumps(metadata),
        ),
    )
    conn.commit()
    return cur.lastrowid or 0


def update_phase_metric(
    conn: sqlite3.Connection,
    metric_id: int,
    *,
    status: str | None = None,
    ended_at: str | None = None,
    duration_seconds: float | None = None,
    output_files: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    parts: list[str] = []
    vals: list[Any] = []
    if status is not None:
        parts.append("status = ?")
        vals.append(status)
    if ended_at is not None:
        parts.append("ended_at = ?")
        vals.append(ended_at)
    if duration_seconds is not None:
        parts.append("duration_seconds = ?")
        vals.append(duration_seconds)
    if output_files is not None:
        parts.append("output_files = ?")
        vals.append(_json_dumps(output_files))
    if metadata is not None:
        parts.append("metadata_json = ?")
        vals.append(_json_dumps(metadata))
    if not parts:
        return
    vals.append(metric_id)
    conn.execute(f"UPDATE phase_metrics SET {', '.join(parts)} WHERE id = ?", vals)
    conn.commit()


# ------------------------------------------------------------------
# CRUD — Token Usage
# ------------------------------------------------------------------


def insert_token_usage(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    phase: str,
    agent: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cached_tokens: int = 0,
    successful_requests: int = 0,
    estimated_cost_usd: float | None = None,
) -> None:
    conn.execute(
        """INSERT INTO token_usage
           (run_id, phase, agent, prompt_tokens, completion_tokens,
            total_tokens, cached_tokens, successful_requests, estimated_cost_usd)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            phase,
            agent,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cached_tokens,
            successful_requests,
            estimated_cost_usd,
        ),
    )
    conn.commit()


def get_token_usage_by_run(conn: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM token_usage WHERE run_id = ? ORDER BY id", (run_id,)
    ).fetchall()
    return _rows_to_dicts(rows)


def get_total_token_usage(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return aggregate token usage across all runs."""
    row = conn.execute(
        """SELECT
             COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
             COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
             COALESCE(SUM(total_tokens), 0) AS total_tokens,
             COALESCE(SUM(cached_tokens), 0) AS cached_tokens,
             COALESCE(SUM(successful_requests), 0) AS successful_requests,
             COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
           FROM token_usage"""
    ).fetchone()
    return dict(row) if row else {}


# ------------------------------------------------------------------
# CRUD — Claims
# ------------------------------------------------------------------


def upsert_claim(
    conn: sqlite3.Connection,
    *,
    claim_id: str,
    text: str,
    claim_type: str | None = None,
    source_document: str | None = None,
    cited_source: str | None = None,
    topic_tags: list[str] | None = None,
    entity_tags: list[str] | None = None,
    verification_status: str | None = None,
    confidence: float | None = None,
    evidence_excerpt: str | None = None,
    run_id: str | None = None,
    timestamp: str | None = None,
) -> None:
    """Insert a new claim or update an existing one.

    On conflict (same *claim_id*), increments ``times_seen`` and updates
    verification fields if they are provided.
    """
    conn.execute(
        """INSERT INTO claims
           (id, text, claim_type, source_document, cited_source,
            topic_tags, entity_tags, verification_status, confidence,
            evidence_excerpt, first_seen_run, first_seen_at,
            last_verified_at, last_verified_run, times_seen, times_verified)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1)
           ON CONFLICT(id) DO UPDATE SET
              times_seen = times_seen + 1,
              verification_status = COALESCE(excluded.verification_status, verification_status),
              confidence = COALESCE(excluded.confidence, confidence),
              evidence_excerpt = COALESCE(excluded.evidence_excerpt, evidence_excerpt),
              last_verified_at = COALESCE(excluded.last_verified_at, last_verified_at),
              last_verified_run = COALESCE(excluded.last_verified_run, last_verified_run),
              times_verified = times_verified + 1
        """,
        (
            claim_id,
            text,
            claim_type,
            source_document,
            cited_source,
            _json_dumps(topic_tags),
            _json_dumps(entity_tags),
            verification_status,
            confidence,
            evidence_excerpt,
            run_id,
            timestamp,
            timestamp,
            run_id,
        ),
    )
    conn.commit()


def get_claim(conn: sqlite3.Connection, claim_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
    return _row_to_dict(row)


def get_claims(
    conn: sqlite3.Connection,
    *,
    run_id: str | None = None,
    status: str | None = None,
    min_confidence: float | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    conditions: list[str] = []
    params: list[Any] = []

    if run_id:
        conditions.append("first_seen_run = ?")
        params.append(run_id)
    if status:
        conditions.append("verification_status = ?")
        params.append(status)
    if min_confidence is not None:
        conditions.append("confidence >= ?")
        params.append(min_confidence)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = conn.execute(
        f"SELECT * FROM claims {where} ORDER BY confidence DESC LIMIT ?",
        [*params, limit],
    ).fetchall()
    return _rows_to_dicts(rows)


def search_claims_fts(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Full-text search over claims using FTS5."""
    rows = conn.execute(
        """SELECT c.*, rank
           FROM claims_fts fts
           JOIN claims c ON c.rowid = fts.rowid
           WHERE claims_fts MATCH ?
           ORDER BY rank
           LIMIT ?""",
        (query, limit),
    ).fetchall()
    return _rows_to_dicts(rows)


def get_stale_claims(
    conn: sqlite3.Connection,
    *,
    older_than_days: int = 30,
    topic: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return claims whose last verification is older than *older_than_days*."""
    conditions = [
        "julianday('now') - julianday(last_verified_at) > ?",
    ]
    params: list[Any] = [older_than_days]

    if topic:
        conditions.append("topic_tags LIKE ?")
        params.append(f"%{topic}%")

    where = f"WHERE {' AND '.join(conditions)}"
    rows = conn.execute(
        f"SELECT * FROM claims {where} ORDER BY last_verified_at ASC LIMIT ?",
        [*params, limit],
    ).fetchall()
    return _rows_to_dicts(rows)


# ------------------------------------------------------------------
# CRUD — Claim Sources
# ------------------------------------------------------------------


def upsert_claim_source(
    conn: sqlite3.Connection,
    *,
    claim_id: str,
    source_url: str,
    domain: str | None = None,
    support_type: str | None = None,
    evidence_excerpt: str | None = None,
    retrieved_at: str | None = None,
    http_status: int | None = None,
    content_hash: str | None = None,
) -> None:
    conn.execute(
        """INSERT INTO claim_sources
           (claim_id, source_url, domain, support_type, evidence_excerpt,
            retrieved_at, http_status, content_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(claim_id, source_url) DO UPDATE SET
              support_type = COALESCE(excluded.support_type, support_type),
              evidence_excerpt = COALESCE(excluded.evidence_excerpt, evidence_excerpt),
              retrieved_at = COALESCE(excluded.retrieved_at, retrieved_at),
              http_status = COALESCE(excluded.http_status, http_status),
              content_hash = COALESCE(excluded.content_hash, content_hash)
        """,
        (
            claim_id,
            source_url,
            domain,
            support_type,
            evidence_excerpt,
            retrieved_at,
            http_status,
            content_hash,
        ),
    )
    conn.commit()


# ------------------------------------------------------------------
# CRUD — Claim History
# ------------------------------------------------------------------


def insert_claim_history(
    conn: sqlite3.Connection,
    *,
    claim_id: str,
    run_id: str,
    verified_at: str,
    old_status: str | None = None,
    new_status: str | None = None,
    old_confidence: float | None = None,
    new_confidence: float | None = None,
    method: str = "initial",
) -> None:
    conn.execute(
        """INSERT INTO claim_history
           (claim_id, run_id, verified_at, old_status, new_status,
            old_confidence, new_confidence, method)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            claim_id,
            run_id,
            verified_at,
            old_status,
            new_status,
            old_confidence,
            new_confidence,
            method,
        ),
    )
    conn.commit()


def get_claim_history(conn: sqlite3.Connection, claim_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM claim_history WHERE claim_id = ? ORDER BY verified_at",
        (claim_id,),
    ).fetchall()
    return _rows_to_dicts(rows)


# ------------------------------------------------------------------
# CRUD — Sources (global)
# ------------------------------------------------------------------


def upsert_source(
    conn: sqlite3.Connection,
    *,
    url: str,
    domain: str,
    title: str | None = None,
    run_id: str | None = None,
    timestamp: str | None = None,
    content_hash: str | None = None,
) -> None:
    conn.execute(
        """INSERT INTO sources
           (url, domain, title, first_seen_run, first_seen_at,
            last_fetched_at, times_cited, content_hash)
           VALUES (?, ?, ?, ?, ?, ?, 1, ?)
           ON CONFLICT(url) DO UPDATE SET
              times_cited = times_cited + 1,
              last_fetched_at = COALESCE(excluded.last_fetched_at, last_fetched_at),
              content_hash = COALESCE(excluded.content_hash, content_hash),
              title = COALESCE(excluded.title, title)
        """,
        (url, domain, title, run_id, timestamp, timestamp, content_hash),
    )
    conn.commit()


def get_source(conn: sqlite3.Connection, url: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM sources WHERE url = ?", (url,)).fetchone()
    return _row_to_dict(row)


def get_top_sources(conn: sqlite3.Connection, *, limit: int = 20) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM sources ORDER BY times_cited DESC LIMIT ?", (limit,)
    ).fetchall()
    return _rows_to_dicts(rows)


def get_source_reliability(conn: sqlite3.Connection, domain: str) -> dict[str, Any]:
    """Calculate verification stats for all claims citing a domain."""
    row = conn.execute(
        """SELECT
             COUNT(*) AS total_claims,
             SUM(CASE WHEN c.verification_status = 'VERIFIED' THEN 1 ELSE 0 END) AS verified,
             SUM(CASE WHEN c.verification_status = 'CONTRADICTED'
                 THEN 1 ELSE 0 END) AS contradicted,
             AVG(c.confidence) AS avg_confidence
           FROM claim_sources cs
           JOIN claims c ON c.id = cs.claim_id
           WHERE cs.domain = ?""",
        (domain,),
    ).fetchone()
    return dict(row) if row else {}


# ------------------------------------------------------------------
# CRUD — Events
# ------------------------------------------------------------------


def insert_event(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    timestamp: str,
    action: str,
    phase: str | None = None,
    agent: str | None = None,
    task: str | None = None,
    detail: str | None = None,
    metadata: dict[str, Any] | None = None,
    tokens_used: int | None = None,
) -> None:
    conn.execute(
        """INSERT INTO events
           (run_id, timestamp, phase, agent, task, action, detail,
            metadata_json, tokens_used)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            timestamp,
            phase,
            agent,
            task,
            action,
            detail,
            _json_dumps(metadata),
            tokens_used,
        ),
    )
    conn.commit()


def get_events(
    conn: sqlite3.Connection,
    *,
    run_id: str | None = None,
    phase: str | None = None,
    action: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    conditions: list[str] = []
    params: list[Any] = []

    if run_id:
        conditions.append("run_id = ?")
        params.append(run_id)
    if phase:
        conditions.append("phase = ?")
        params.append(phase)
    if action:
        conditions.append("action = ?")
        params.append(action)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = conn.execute(
        f"SELECT * FROM events {where} ORDER BY id LIMIT ?",
        [*params, limit],
    ).fetchall()
    return _rows_to_dicts(rows)


# ------------------------------------------------------------------
# Aggregate queries
# ------------------------------------------------------------------


def get_run_stats(conn: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    """Return summary statistics for a single run."""
    claims_row = conn.execute(
        """SELECT
             COUNT(*) AS total_claims,
             SUM(CASE WHEN verification_status = 'VERIFIED' THEN 1 ELSE 0 END) AS verified,
             SUM(CASE WHEN verification_status = 'PARTIALLY_VERIFIED' THEN 1 ELSE 0 END) AS partial,
             SUM(CASE WHEN verification_status = 'UNVERIFIABLE' THEN 1 ELSE 0 END) AS unverifiable,
             SUM(CASE WHEN verification_status = 'CONTRADICTED' THEN 1 ELSE 0 END) AS contradicted,
             AVG(confidence) AS avg_confidence
           FROM claims WHERE first_seen_run = ?""",
        (run_id,),
    ).fetchone()

    tokens_row = conn.execute(
        """SELECT
             COALESCE(SUM(total_tokens), 0) AS total_tokens,
             COALESCE(SUM(estimated_cost_usd), 0) AS total_cost
           FROM token_usage WHERE run_id = ?""",
        (run_id,),
    ).fetchone()

    sources_row = conn.execute(
        """SELECT COUNT(DISTINCT cs.source_url) AS unique_sources,
                  COUNT(DISTINCT cs.domain) AS unique_domains
           FROM claim_sources cs
           JOIN claims c ON c.id = cs.claim_id
           WHERE c.first_seen_run = ?""",
        (run_id,),
    ).fetchone()

    return {
        "claims": dict(claims_row) if claims_row else {},
        "tokens": dict(tokens_row) if tokens_row else {},
        "sources": dict(sources_row) if sources_row else {},
    }


def get_global_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return summary statistics across all runs."""
    runs_row = conn.execute("SELECT COUNT(*) AS total_runs FROM runs").fetchone()

    claims_row = conn.execute(
        """SELECT
             COUNT(*) AS total_claims,
             SUM(CASE WHEN verification_status = 'VERIFIED' THEN 1 ELSE 0 END) AS verified,
             SUM(CASE WHEN verification_status = 'PARTIALLY_VERIFIED' THEN 1 ELSE 0 END) AS partial,
             SUM(CASE WHEN verification_status = 'UNVERIFIABLE' THEN 1 ELSE 0 END) AS unverifiable,
             SUM(CASE WHEN verification_status = 'CONTRADICTED' THEN 1 ELSE 0 END) AS contradicted,
             AVG(confidence) AS avg_confidence
           FROM claims"""
    ).fetchone()

    sources_row = conn.execute("SELECT COUNT(*) AS total_sources FROM sources").fetchone()

    tokens_row = conn.execute(
        """SELECT
             COALESCE(SUM(total_tokens), 0) AS total_tokens,
             COALESCE(SUM(estimated_cost_usd), 0) AS total_cost
           FROM token_usage"""
    ).fetchone()

    reliability_rows = conn.execute(
        """SELECT r.id, r.topic, r.timestamp,
                  COUNT(c.id) AS claims_count,
                  AVG(c.confidence) AS avg_confidence
           FROM runs r
           LEFT JOIN claims c ON c.first_seen_run = r.id
           GROUP BY r.id
           ORDER BY r.timestamp DESC
           LIMIT 10"""
    ).fetchall()

    return {
        "runs": dict(runs_row) if runs_row else {},
        "claims": dict(claims_row) if claims_row else {},
        "sources": dict(sources_row) if sources_row else {},
        "tokens": dict(tokens_row) if tokens_row else {},
        "reliability_trend": _rows_to_dicts(reliability_rows),
    }
