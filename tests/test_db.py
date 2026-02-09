"""Tests for recon.db — the SQLite knowledge database.

All tests use an in-memory database or a temp-file database to avoid
polluting the filesystem.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

from recon.db import (
    SCHEMA_VERSION,
    get_claim,
    get_claim_history,
    get_claims,
    get_db,
    get_events,
    get_global_stats,
    get_run,
    get_run_stats,
    get_runs,
    get_source,
    get_source_reliability,
    get_stale_claims,
    get_token_usage_by_run,
    get_top_sources,
    get_total_token_usage,
    insert_claim_history,
    insert_event,
    insert_phase_metric,
    insert_run,
    insert_token_usage,
    search_claims_fts,
    update_phase_metric,
    update_run,
    upsert_claim,
    upsert_claim_source,
    upsert_source,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def conn(tmp_path: Path) -> sqlite3.Connection:
    """Return a fresh knowledge database connection."""
    return get_db(tmp_path / "test.db")


@pytest.fixture()
def seeded_conn(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Return a connection with a sample run and claims pre-inserted."""
    insert_run(
        conn,
        run_id="run-001",
        timestamp="2026-01-15T10:00:00Z",
        topic="AI agents",
        depth="standard",
        model="kimi-k2.5",
        provider="openrouter",
        search_provider="tavily",
    )
    upsert_claim(
        conn,
        claim_id="c-001",
        text="CrewAI has 44K GitHub stars",
        claim_type="statistic",
        source_document="research/landscape.md",
        cited_source="https://github.com/crewAIInc/crewAI",
        topic_tags=["AI", "agents"],
        entity_tags=["CrewAI", "GitHub"],
        verification_status="VERIFIED",
        confidence=0.92,
        evidence_excerpt="As of 2026, CrewAI repo shows 44K stars.",
        run_id="run-001",
        timestamp="2026-01-15T10:05:00Z",
    )
    upsert_claim(
        conn,
        claim_id="c-002",
        text="LangGraph is growing fast",
        claim_type="qualitative",
        source_document="research/trends.md",
        verification_status="PARTIALLY_VERIFIED",
        confidence=0.65,
        run_id="run-001",
        timestamp="2026-01-15T10:06:00Z",
    )
    return conn


# ---------------------------------------------------------------------------
# Schema & connection
# ---------------------------------------------------------------------------


class TestSchema:
    def test_get_db_creates_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "knowledge.db"
        conn = get_db(db_path)
        assert db_path.exists()
        conn.close()

    def test_get_db_creates_parent_dirs(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nested" / "dirs" / "knowledge.db"
        conn = get_db(db_path)
        assert db_path.exists()
        conn.close()

    def test_schema_version_stored(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
        assert row is not None
        assert int(row["value"]) == SCHEMA_VERSION

    def test_tables_exist(self, conn: sqlite3.Connection) -> None:
        tables = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        expected_main = {
            "schema_meta",
            "runs",
            "phase_metrics",
            "token_usage",
            "claims",
            "claim_sources",
            "claim_history",
            "sources",
            "events",
            "claims_fts",
        }
        assert expected_main.issubset(tables)

    def test_wal_mode(self, conn: sqlite3.Connection) -> None:
        mode = conn.execute("PRAGMA journal_mode").fetchone()
        assert mode[0] == "wal"

    def test_foreign_keys_enabled(self, conn: sqlite3.Connection) -> None:
        fk = conn.execute("PRAGMA foreign_keys").fetchone()
        assert fk[0] == 1

    def test_reopen_existing_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "reopen.db"
        conn1 = get_db(db_path)
        insert_run(
            conn1,
            run_id="r1",
            timestamp="2026-01-01T00:00:00Z",
            topic="test",
            depth="quick",
        )
        conn1.close()

        conn2 = get_db(db_path)
        run = get_run(conn2, "r1")
        assert run is not None
        assert run["topic"] == "test"
        conn2.close()

    def test_migration_from_empty(self, tmp_path: Path) -> None:
        """A DB without schema_meta should be re-initialised."""
        db_path = tmp_path / "legacy.db"
        raw = sqlite3.connect(str(db_path))
        raw.execute("CREATE TABLE dummy (id TEXT)")
        raw.commit()
        raw.close()

        conn = get_db(db_path)
        row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
        assert row is not None
        conn.close()


# ---------------------------------------------------------------------------
# CRUD — Runs
# ---------------------------------------------------------------------------


class TestRuns:
    def test_insert_and_get_run(self, conn: sqlite3.Connection) -> None:
        insert_run(
            conn,
            run_id="run-abc",
            timestamp="2026-02-01T12:00:00Z",
            topic="Quantum computing",
            depth="deep",
            model="gpt-4o",
            provider="openai",
            verify=True,
            config_json={"custom": "data"},
        )
        run = get_run(conn, "run-abc")
        assert run is not None
        assert run["topic"] == "Quantum computing"
        assert run["depth"] == "deep"
        assert run["model"] == "gpt-4o"
        assert run["status"] == "running"

    def test_get_nonexistent_run(self, conn: sqlite3.Connection) -> None:
        assert get_run(conn, "nonexistent") is None

    def test_update_run(self, conn: sqlite3.Connection) -> None:
        insert_run(
            conn,
            run_id="run-upd",
            timestamp="2026-01-01T00:00:00Z",
            topic="test",
            depth="quick",
        )
        update_run(conn, "run-upd", status="done", duration_seconds=42.5)
        run = get_run(conn, "run-upd")
        assert run is not None
        assert run["status"] == "done"
        assert run["duration_seconds"] == 42.5

    def test_update_run_partial(self, conn: sqlite3.Connection) -> None:
        insert_run(
            conn,
            run_id="run-part",
            timestamp="2026-01-01T00:00:00Z",
            topic="test",
            depth="quick",
        )
        update_run(conn, "run-part", status="error")
        run = get_run(conn, "run-part")
        assert run is not None
        assert run["status"] == "error"
        assert run["duration_seconds"] is None

    def test_update_run_no_fields(self, conn: sqlite3.Connection) -> None:
        """update_run with no fields should be a no-op."""
        insert_run(
            conn,
            run_id="run-noop",
            timestamp="2026-01-01T00:00:00Z",
            topic="test",
            depth="quick",
        )
        update_run(conn, "run-noop")  # no changes
        run = get_run(conn, "run-noop")
        assert run is not None
        assert run["status"] == "running"

    def test_get_runs_ordering(self, conn: sqlite3.Connection) -> None:
        for i in range(3):
            insert_run(
                conn,
                run_id=f"run-{i}",
                timestamp=f"2026-01-0{i + 1}T00:00:00Z",
                topic="test",
                depth="quick",
            )
        runs = get_runs(conn, limit=10)
        assert len(runs) == 3
        # Most recent first
        assert runs[0]["id"] == "run-2"

    def test_get_runs_topic_filter(self, conn: sqlite3.Connection) -> None:
        insert_run(
            conn,
            run_id="r-ai",
            timestamp="2026-01-01T00:00:00Z",
            topic="AI agents",
            depth="quick",
        )
        insert_run(
            conn,
            run_id="r-bio",
            timestamp="2026-01-02T00:00:00Z",
            topic="Biotech trends",
            depth="quick",
        )
        results = get_runs(conn, topic="AI")
        assert len(results) == 1
        assert results[0]["id"] == "r-ai"


# ---------------------------------------------------------------------------
# CRUD — Phase Metrics
# ---------------------------------------------------------------------------


class TestPhaseMetrics:
    def test_insert_and_update(self, seeded_conn: sqlite3.Connection) -> None:
        metric_id = insert_phase_metric(
            seeded_conn,
            run_id="run-001",
            phase="investigation",
            status="running",
            started_at="2026-01-15T10:00:00Z",
        )
        assert metric_id > 0

        update_phase_metric(
            seeded_conn,
            metric_id,
            status="done",
            ended_at="2026-01-15T10:05:00Z",
            duration_seconds=300.0,
            output_files=["research/overview.md"],
            metadata={"agents": 3},
        )
        row = seeded_conn.execute(
            "SELECT * FROM phase_metrics WHERE id = ?", (metric_id,)
        ).fetchone()
        assert row["status"] == "done"
        assert row["duration_seconds"] == 300.0

    def test_update_no_fields(self, seeded_conn: sqlite3.Connection) -> None:
        metric_id = insert_phase_metric(
            seeded_conn,
            run_id="run-001",
            phase="test",
            status="running",
        )
        update_phase_metric(seeded_conn, metric_id)  # no-op
        row = seeded_conn.execute(
            "SELECT * FROM phase_metrics WHERE id = ?", (metric_id,)
        ).fetchone()
        assert row["status"] == "running"


# ---------------------------------------------------------------------------
# CRUD — Token Usage
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_insert_and_query(self, seeded_conn: sqlite3.Connection) -> None:
        insert_token_usage(
            seeded_conn,
            run_id="run-001",
            phase="investigation",
            agent="researcher-1",
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            estimated_cost_usd=0.003,
        )
        insert_token_usage(
            seeded_conn,
            run_id="run-001",
            phase="verification",
            prompt_tokens=800,
            completion_tokens=200,
            total_tokens=1000,
        )

        by_run = get_token_usage_by_run(seeded_conn, "run-001")
        assert len(by_run) == 2

        totals = get_total_token_usage(seeded_conn)
        assert totals["total_tokens"] == 2500
        assert totals["prompt_tokens"] == 1800


# ---------------------------------------------------------------------------
# CRUD — Claims
# ---------------------------------------------------------------------------


class TestClaims:
    def test_upsert_insert(self, seeded_conn: sqlite3.Connection) -> None:
        claim = get_claim(seeded_conn, "c-001")
        assert claim is not None
        assert claim["text"] == "CrewAI has 44K GitHub stars"
        assert claim["confidence"] == 0.92
        assert claim["times_seen"] == 1

    def test_upsert_dedup(self, seeded_conn: sqlite3.Connection) -> None:
        """Upserting the same claim_id should increment times_seen."""
        upsert_claim(
            seeded_conn,
            claim_id="c-001",
            text="CrewAI has 44K GitHub stars",
            confidence=0.95,
            verification_status="VERIFIED",
        )
        claim = get_claim(seeded_conn, "c-001")
        assert claim is not None
        assert claim["times_seen"] == 2
        assert claim["confidence"] == 0.95  # updated

    def test_get_claims_by_status(self, seeded_conn: sqlite3.Connection) -> None:
        verified = get_claims(seeded_conn, status="VERIFIED")
        assert len(verified) == 1
        assert verified[0]["id"] == "c-001"

    def test_get_claims_by_min_confidence(self, seeded_conn: sqlite3.Connection) -> None:
        high = get_claims(seeded_conn, min_confidence=0.9)
        assert len(high) == 1
        low = get_claims(seeded_conn, min_confidence=0.5)
        assert len(low) == 2

    def test_get_claims_by_run(self, seeded_conn: sqlite3.Connection) -> None:
        claims = get_claims(seeded_conn, run_id="run-001")
        assert len(claims) == 2

    def test_get_nonexistent_claim(self, conn: sqlite3.Connection) -> None:
        assert get_claim(conn, "nonexistent") is None

    def test_fts_search(self, seeded_conn: sqlite3.Connection) -> None:
        results = search_claims_fts(seeded_conn, "CrewAI")
        assert len(results) >= 1
        assert any("CrewAI" in r["text"] for r in results)

    def test_fts_search_no_results(self, seeded_conn: sqlite3.Connection) -> None:
        results = search_claims_fts(seeded_conn, "nonexistentterm12345")
        assert results == []

    def test_fts_search_by_tag(self, seeded_conn: sqlite3.Connection) -> None:
        results = search_claims_fts(seeded_conn, "agents")
        assert len(results) >= 1

    def test_stale_claims(self, seeded_conn: sqlite3.Connection) -> None:
        """Claims verified more than N days ago should appear as stale."""
        stale = get_stale_claims(seeded_conn, older_than_days=1)
        # Both claims were verified at a timestamp in 2026; depending on
        # current date they may or may not be stale.  Since we're running
        # in Feb 2026 and the claims are from Jan 2026, they are stale
        # with older_than_days=1.
        assert len(stale) >= 1


# ---------------------------------------------------------------------------
# CRUD — Claim Sources
# ---------------------------------------------------------------------------


class TestClaimSources:
    def test_upsert_and_query(self, seeded_conn: sqlite3.Connection) -> None:
        upsert_claim_source(
            seeded_conn,
            claim_id="c-001",
            source_url="https://github.com/crewAIInc/crewAI",
            domain="github.com",
            support_type="primary",
            evidence_excerpt="Stars badge shows 44K",
            retrieved_at="2026-01-15T10:05:00Z",
            http_status=200,
        )
        row = seeded_conn.execute(
            "SELECT * FROM claim_sources WHERE claim_id = ? AND source_url = ?",
            ("c-001", "https://github.com/crewAIInc/crewAI"),
        ).fetchone()
        assert row is not None
        assert row["domain"] == "github.com"
        assert row["http_status"] == 200

    def test_upsert_updates_existing(self, seeded_conn: sqlite3.Connection) -> None:
        upsert_claim_source(
            seeded_conn,
            claim_id="c-001",
            source_url="https://example.com",
            domain="example.com",
            support_type="secondary",
        )
        upsert_claim_source(
            seeded_conn,
            claim_id="c-001",
            source_url="https://example.com",
            domain="example.com",
            support_type="primary",
            http_status=200,
        )
        row = seeded_conn.execute(
            "SELECT * FROM claim_sources WHERE claim_id = ? AND source_url = ?",
            ("c-001", "https://example.com"),
        ).fetchone()
        assert row["support_type"] == "primary"
        assert row["http_status"] == 200


# ---------------------------------------------------------------------------
# CRUD — Claim History
# ---------------------------------------------------------------------------


class TestClaimHistory:
    def test_insert_and_get(self, seeded_conn: sqlite3.Connection) -> None:
        insert_claim_history(
            seeded_conn,
            claim_id="c-001",
            run_id="run-001",
            verified_at="2026-01-15T10:05:00Z",
            old_status=None,
            new_status="VERIFIED",
            old_confidence=None,
            new_confidence=0.92,
            method="initial",
        )
        insert_claim_history(
            seeded_conn,
            claim_id="c-001",
            run_id="run-001",
            verified_at="2026-01-15T11:00:00Z",
            old_status="VERIFIED",
            new_status="VERIFIED",
            old_confidence=0.92,
            new_confidence=0.95,
            method="re-verification",
        )
        history = get_claim_history(seeded_conn, "c-001")
        assert len(history) == 2
        assert history[0]["method"] == "initial"
        assert history[1]["new_confidence"] == 0.95

    def test_empty_history(self, seeded_conn: sqlite3.Connection) -> None:
        history = get_claim_history(seeded_conn, "c-002")
        assert history == []


# ---------------------------------------------------------------------------
# CRUD — Sources
# ---------------------------------------------------------------------------


class TestSources:
    def test_upsert_and_get(self, seeded_conn: sqlite3.Connection) -> None:
        upsert_source(
            seeded_conn,
            url="https://github.com/crewAIInc/crewAI",
            domain="github.com",
            title="CrewAI GitHub Repository",
            run_id="run-001",
            timestamp="2026-01-15T10:05:00Z",
        )
        source = get_source(seeded_conn, "https://github.com/crewAIInc/crewAI")
        assert source is not None
        assert source["domain"] == "github.com"
        assert source["times_cited"] == 1

    def test_upsert_increments_times_cited(self, seeded_conn: sqlite3.Connection) -> None:
        url = "https://example.com/article"
        upsert_source(seeded_conn, url=url, domain="example.com", run_id="run-001")
        upsert_source(seeded_conn, url=url, domain="example.com", run_id="run-001")
        source = get_source(seeded_conn, url)
        assert source is not None
        assert source["times_cited"] == 2

    def test_get_nonexistent_source(self, conn: sqlite3.Connection) -> None:
        assert get_source(conn, "https://nonexistent.example.com") is None

    def test_get_top_sources(self, seeded_conn: sqlite3.Connection) -> None:
        for i in range(5):
            upsert_source(
                seeded_conn,
                url=f"https://example.com/page-{i}",
                domain="example.com",
                run_id="run-001",
            )
        # Cite page-0 extra times
        for _ in range(3):
            upsert_source(
                seeded_conn,
                url="https://example.com/page-0",
                domain="example.com",
            )
        top = get_top_sources(seeded_conn, limit=3)
        assert len(top) == 3
        assert top[0]["url"] == "https://example.com/page-0"
        assert top[0]["times_cited"] == 4  # 1 initial + 3 extra

    def test_source_reliability(self, seeded_conn: sqlite3.Connection) -> None:
        upsert_claim_source(
            seeded_conn,
            claim_id="c-001",
            source_url="https://github.com/crewAIInc/crewAI",
            domain="github.com",
        )
        reliability = get_source_reliability(seeded_conn, "github.com")
        assert reliability["total_claims"] == 1
        assert reliability["verified"] == 1


# ---------------------------------------------------------------------------
# CRUD — Events
# ---------------------------------------------------------------------------


class TestEvents:
    def test_insert_and_get(self, seeded_conn: sqlite3.Connection) -> None:
        insert_event(
            seeded_conn,
            run_id="run-001",
            timestamp="2026-01-15T10:00:00Z",
            action="phase_start",
            phase="investigation",
            agent="pipeline",
            detail="Starting investigation phase",
            metadata={"agents": 3},
            tokens_used=100,
        )
        events = get_events(seeded_conn, run_id="run-001")
        assert len(events) == 1
        assert events[0]["action"] == "phase_start"

    def test_filter_by_action(self, seeded_conn: sqlite3.Connection) -> None:
        insert_event(
            seeded_conn,
            run_id="run-001",
            timestamp="2026-01-15T10:00:00Z",
            action="phase_start",
            phase="investigation",
        )
        insert_event(
            seeded_conn,
            run_id="run-001",
            timestamp="2026-01-15T10:01:00Z",
            action="tool_call",
            phase="investigation",
            agent="researcher-1",
        )
        starts = get_events(seeded_conn, action="phase_start")
        assert len(starts) == 1

    def test_filter_by_phase(self, seeded_conn: sqlite3.Connection) -> None:
        insert_event(
            seeded_conn,
            run_id="run-001",
            timestamp="2026-01-15T10:00:00Z",
            action="start",
            phase="investigation",
        )
        insert_event(
            seeded_conn,
            run_id="run-001",
            timestamp="2026-01-15T10:01:00Z",
            action="start",
            phase="verification",
        )
        inv = get_events(seeded_conn, phase="investigation")
        assert len(inv) == 1

    def test_empty_events(self, conn: sqlite3.Connection) -> None:
        events = get_events(conn)
        assert events == []


# ---------------------------------------------------------------------------
# Aggregate queries
# ---------------------------------------------------------------------------


class TestAggregates:
    def test_run_stats(self, seeded_conn: sqlite3.Connection) -> None:
        insert_token_usage(
            seeded_conn,
            run_id="run-001",
            phase="investigation",
            total_tokens=5000,
            estimated_cost_usd=0.01,
        )
        stats = get_run_stats(seeded_conn, "run-001")
        assert stats["claims"]["total_claims"] == 2
        assert stats["claims"]["verified"] == 1
        assert stats["tokens"]["total_tokens"] == 5000

    def test_global_stats_empty(self, conn: sqlite3.Connection) -> None:
        stats = get_global_stats(conn)
        assert stats["runs"]["total_runs"] == 0
        assert stats["claims"]["total_claims"] == 0

    def test_global_stats(self, seeded_conn: sqlite3.Connection) -> None:
        stats = get_global_stats(seeded_conn)
        assert stats["runs"]["total_runs"] == 1
        assert stats["claims"]["total_claims"] == 2
        assert stats["claims"]["verified"] == 1
        assert stats["claims"]["partial"] == 1
