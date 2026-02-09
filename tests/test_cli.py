"""Tests for the Recon CLI commands.

Uses typer.testing.CliRunner to invoke commands without subprocess overhead.
All external calls (LLM, search, crews) are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from recon.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Shared fixture: seed a temp knowledge DB for the v0.3 commands
# ---------------------------------------------------------------------------

@pytest.fixture()
def seeded_db(tmp_path: Path) -> Path:
    """Create and seed a knowledge DB, returning the path."""
    from recon.db import (
        get_db,
        insert_claim_history,
        insert_run,
        insert_token_usage,
        upsert_claim,
        upsert_source,
    )

    db_path = tmp_path / "knowledge.db"
    conn = get_db(str(db_path))

    insert_run(
        conn,
        run_id="run-001",
        timestamp="2026-01-15T10:00:00+00:00",
        topic="Quantum computing",
        depth="standard",
        model="kimi-k2.5",
        provider="openrouter",
        search_provider="tavily",
    )

    upsert_claim(
        conn,
        claim_id="clm-aaa",
        text="Quantum computers can factor large primes exponentially faster.",
        verification_status="VERIFIED",
        confidence=0.92,
        cited_source="https://example.com/quantum",
        run_id="run-001",
        timestamp="2026-01-15T10:05:00+00:00",
        topic_tags=["quantum", "computing"],
    )
    upsert_claim(
        conn,
        claim_id="clm-bbb",
        text="Classical computers are obsolete.",
        verification_status="CONTRADICTED",
        confidence=0.15,
        cited_source="https://example.com/hype",
        run_id="run-001",
        timestamp="2026-01-15T10:06:00+00:00",
        topic_tags=["computing"],
    )
    upsert_claim(
        conn,
        claim_id="clm-ccc",
        text="Error correction remains a major challenge.",
        verification_status="PARTIALLY_VERIFIED",
        confidence=0.70,
        run_id="run-001",
        timestamp="2026-01-15T10:07:00+00:00",
    )

    # Stale claim (verified long ago)
    upsert_claim(
        conn,
        claim_id="clm-old",
        text="Quantum supremacy was achieved in 2019.",
        verification_status="VERIFIED",
        confidence=0.88,
        run_id="run-001",
        timestamp="2024-06-01T00:00:00+00:00",
    )

    insert_claim_history(
        conn,
        claim_id="clm-aaa",
        run_id="run-001",
        verified_at="2026-01-15T10:05:00+00:00",
        old_status=None,
        new_status="VERIFIED",
        old_confidence=None,
        new_confidence=0.92,
        method="citation_check",
    )
    insert_claim_history(
        conn,
        claim_id="clm-aaa",
        run_id="run-001",
        verified_at="2026-01-15T10:05:30+00:00",
        old_status="VERIFIED",
        new_status="VERIFIED",
        old_confidence=0.92,
        new_confidence=0.95,
        method="semantic_verification",
    )

    insert_token_usage(
        conn,
        run_id="run-001",
        phase="investigation",
        prompt_tokens=5000,
        completion_tokens=2000,
        total_tokens=7000,
        estimated_cost_usd=0.0012,
    )

    upsert_source(
        conn,
        url="https://example.com/quantum",
        domain="example.com",
        title="Quantum Computing Overview",
        run_id="run-001",
        timestamp="2026-01-15T10:05:00+00:00",
    )

    conn.close()
    return db_path


class TestVersion:
    def test_version_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "recon" in result.output
        assert "0.1.0" in result.output

    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "research pipelines" in result.output.lower() or "Usage" in result.output


class TestRun:
    def test_run_requires_plan_or_topic(self) -> None:
        result = runner.invoke(app, ["run"])
        assert result.exit_code == 1
        assert "plan file" in result.output.lower() or "topic" in result.output.lower()

    def test_run_rejects_both_plan_and_topic(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text("topic: test\n")
        result = runner.invoke(app, ["run", str(plan), "--topic", "test"])
        assert result.exit_code == 1
        assert "Cannot specify both" in result.output

    def test_run_nonexistent_plan(self) -> None:
        result = runner.invoke(app, ["run", "/nonexistent/plan.yaml"])
        assert result.exit_code == 1

    def test_run_dry_run(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text(
            "topic: Test topic\nquestions:\n  - What is this?\ndepth: quick\nverify: false\n"
        )
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            result = runner.invoke(app, ["run", str(plan), "--dry-run"])
        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_run_dry_run_verbose(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text(
            "topic: Test topic\nquestions:\n  - What is this?\ndepth: quick\nverify: false\n"
        )
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            result = runner.invoke(app, ["run", str(plan), "--dry-run", "--verbose"])
        assert result.exit_code == 0
        assert "Investigation Agents" in result.output

    def test_run_inline_topic_dry_run(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            result = runner.invoke(
                app, ["run", "--topic", "AI testing", "--dry-run", "--no-verify"]
            )
        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_run_dry_run_shows_incremental_mode(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text(
            "topic: Test topic\nquestions:\n  - What is this?\ndepth: quick\nverify: false\n"
        )
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            result = runner.invoke(app, ["run", str(plan), "--dry-run"])
        assert result.exit_code == 0
        assert "incremental" in result.output

    def test_run_dry_run_force_shows_full_mode(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text(
            "topic: Test topic\nquestions:\n  - What is this?\ndepth: quick\nverify: false\n"
        )
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            result = runner.invoke(app, ["run", str(plan), "--dry-run", "--force"])
        assert result.exit_code == 0
        assert "full" in result.output


class TestInit:
    def test_init_default_template(self, tmp_path: Path) -> None:
        output = tmp_path / "plan.yaml"
        result = runner.invoke(app, ["init", "--output", str(output)])
        assert result.exit_code == 0
        assert "Created" in result.output
        assert output.exists()
        content = output.read_text()
        assert "topic:" in content

    def test_init_specific_template(self, tmp_path: Path) -> None:
        output = tmp_path / "plan.yaml"
        result = runner.invoke(
            app, ["init", "--template", "competitive-analysis", "--output", str(output)]
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_init_nonexistent_template(self, tmp_path: Path) -> None:
        output = tmp_path / "plan.yaml"
        result = runner.invoke(app, ["init", "--template", "nonexistent", "--output", str(output)])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_init_overwrite_decline(self, tmp_path: Path) -> None:
        output = tmp_path / "plan.yaml"
        output.write_text("existing content")
        result = runner.invoke(app, ["init", "--output", str(output)], input="n\n")
        assert result.exit_code == 0
        # File should still have original content.
        assert output.read_text() == "existing content"


class TestTemplates:
    def test_templates_lists_all(self) -> None:
        result = runner.invoke(app, ["templates"])
        assert result.exit_code == 0
        assert "market-research" in result.output
        assert "competitive-analysis" in result.output
        assert "technical-landscape" in result.output
        assert "opportunity-finder" in result.output


class TestStatus:
    def test_status_no_audit_log(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["status", str(tmp_path)])
        assert result.exit_code == 1
        assert "No audit log" in result.output

    def test_status_with_audit_log(self, tmp_path: Path) -> None:
        audit_file = tmp_path / "audit-log.jsonl"
        entries = [
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "phase": "investigation",
                "agent": "pipeline",
                "action": "phase_start",
                "detail": "",
            },
            {
                "timestamp": "2026-01-01T00:01:00+00:00",
                "phase": "investigation",
                "agent": "pipeline",
                "action": "phase_end",
                "detail": "",
                "metadata": {"output_files": ["research/test.md"]},
            },
        ]
        audit_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        result = runner.invoke(app, ["status", str(tmp_path)])
        assert result.exit_code == 0
        assert "investigation" in result.output
        assert "done" in result.output


class TestVerify:
    def test_verify_missing_dir(self) -> None:
        result = runner.invoke(app, ["verify", "/nonexistent/dir"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_verify_empty_dir(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["verify", str(tmp_path)])
        assert result.exit_code == 0
        assert "No markdown files" in result.output

    def test_verify_with_files_finds_them(self, tmp_path: Path) -> None:
        """Verify that the command detects markdown files and reports the count."""
        research_dir = tmp_path / "research"
        research_dir.mkdir()
        (research_dir / "test.md").write_text("# Test\nSome research content.\n")
        output_dir = tmp_path / "verification"

        # The command will fail at create_llm (no API key), but it finds files first.
        result = runner.invoke(app, ["verify", str(research_dir), "--output", str(output_dir)])
        assert "1 research file" in result.output


class TestRerun:
    def test_rerun_invalid_phase(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text("topic: test\nquestions:\n  - q1\ndepth: quick\n")
        result = runner.invoke(app, ["rerun", str(plan), "--phase", "invalid"])
        assert result.exit_code == 1
        assert "Invalid phase" in result.output

    def test_rerun_nonexistent_plan(self) -> None:
        result = runner.invoke(app, ["rerun", "/nonexistent/plan.yaml"])
        assert result.exit_code == 1

    def test_rerun_valid_phase_accepted(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text("topic: test\nquestions:\n  - q1\ndepth: quick\nverify: true\n")
        # Will fail at create_llm (no API key) but phase validation should pass.
        result = runner.invoke(app, ["rerun", str(plan), "--phase", "verification"])
        assert result.exit_code != 0
        assert "Re-running phase" in result.output


# ===========================================================================
# Knowledge DB commands (v0.3)
# ===========================================================================


class TestClaims:
    """Tests for ``recon claims``."""

    def test_claims_no_db_exits(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nonexistent.db"
        result = runner.invoke(app, ["claims", "--db", str(db_path)])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_claims_lists_all(self, seeded_db: Path) -> None:
        result = runner.invoke(app, ["claims", "--db", str(seeded_db)])
        assert result.exit_code == 0
        assert "clm-aaa" in result.output
        assert "clm-bbb" in result.output
        assert "VERIFIED" in result.output
        assert "CONTRADICTED" in result.output

    def test_claims_filter_by_status(self, seeded_db: Path) -> None:
        result = runner.invoke(
            app, ["claims", "--db", str(seeded_db), "--status", "VERIFIED"]
        )
        assert result.exit_code == 0
        assert "clm-aaa" in result.output
        # clm-bbb is CONTRADICTED, should not appear
        assert "clm-bbb" not in result.output

    def test_claims_filter_by_run(self, seeded_db: Path) -> None:
        result = runner.invoke(
            app, ["claims", "--db", str(seeded_db), "--run", "run-001"]
        )
        assert result.exit_code == 0
        assert "clm-aaa" in result.output

    def test_claims_search_fts(self, seeded_db: Path) -> None:
        result = runner.invoke(
            app, ["claims", "--db", str(seeded_db), "--search", "quantum"]
        )
        assert result.exit_code == 0
        assert "clm-aaa" in result.output

    def test_claims_search_no_results(self, seeded_db: Path) -> None:
        result = runner.invoke(
            app, ["claims", "--db", str(seeded_db), "--search", "xyznonexistent"]
        )
        assert result.exit_code == 0
        assert "No claims found" in result.output

    def test_claims_limit(self, seeded_db: Path) -> None:
        result = runner.invoke(
            app, ["claims", "--db", str(seeded_db), "--limit", "1"]
        )
        assert result.exit_code == 0
        assert "1 results" in result.output

    def test_claims_shows_confidence(self, seeded_db: Path) -> None:
        result = runner.invoke(app, ["claims", "--db", str(seeded_db)])
        assert result.exit_code == 0
        assert "92%" in result.output  # 0.92 formatted as 92%

    def test_claims_empty_db(self, tmp_path: Path) -> None:
        from recon.db import get_db

        db_path = tmp_path / "empty.db"
        conn = get_db(str(db_path))
        conn.close()
        result = runner.invoke(app, ["claims", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "No claims found" in result.output


class TestHistory:
    """Tests for ``recon history``."""

    def test_history_no_db_exits(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nonexistent.db"
        result = runner.invoke(app, ["history", "clm-aaa", "--db", str(db_path)])
        assert result.exit_code == 1

    def test_history_claim_not_found(self, seeded_db: Path) -> None:
        result = runner.invoke(
            app, ["history", "clm-nonexistent", "--db", str(seeded_db)]
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_history_shows_claim_info(self, seeded_db: Path) -> None:
        result = runner.invoke(
            app, ["history", "clm-aaa", "--db", str(seeded_db)]
        )
        assert result.exit_code == 0
        assert "Quantum computers" in result.output
        assert "VERIFIED" in result.output

    def test_history_shows_entries(self, seeded_db: Path) -> None:
        result = runner.invoke(
            app, ["history", "clm-aaa", "--db", str(seeded_db)]
        )
        assert result.exit_code == 0
        assert "citation_check" in result.output
        # Rich may truncate long method names, so check prefix
        assert "semantic_veri" in result.output
        assert "run-001" in result.output

    def test_history_shows_confidence_values(self, seeded_db: Path) -> None:
        result = runner.invoke(
            app, ["history", "clm-aaa", "--db", str(seeded_db)]
        )
        assert result.exit_code == 0
        assert "92%" in result.output
        assert "95%" in result.output

    def test_history_claim_without_history(self, seeded_db: Path) -> None:
        """clm-bbb has no history entries."""
        result = runner.invoke(
            app, ["history", "clm-bbb", "--db", str(seeded_db)]
        )
        assert result.exit_code == 0
        assert "No verification history" in result.output


class TestStats:
    """Tests for ``recon stats``."""

    def test_stats_no_db_exits(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nonexistent.db"
        result = runner.invoke(app, ["stats", "--db", str(db_path)])
        assert result.exit_code == 1

    def test_stats_global(self, seeded_db: Path) -> None:
        result = runner.invoke(app, ["stats", "--db", str(seeded_db)])
        assert result.exit_code == 0
        assert "Knowledge Database Statistics" in result.output
        assert "Total runs" in result.output

    def test_stats_global_shows_claims_breakdown(self, seeded_db: Path) -> None:
        result = runner.invoke(app, ["stats", "--db", str(seeded_db)])
        assert result.exit_code == 0
        assert "Verified" in result.output
        assert "Contradicted" in result.output

    def test_stats_global_shows_tokens(self, seeded_db: Path) -> None:
        result = runner.invoke(app, ["stats", "--db", str(seeded_db)])
        assert result.exit_code == 0
        assert "7,000" in result.output  # total_tokens formatted with commas

    def test_stats_global_shows_sources(self, seeded_db: Path) -> None:
        result = runner.invoke(app, ["stats", "--db", str(seeded_db)])
        assert result.exit_code == 0
        assert "Total sources" in result.output

    def test_stats_global_shows_recent_runs(self, seeded_db: Path) -> None:
        result = runner.invoke(app, ["stats", "--db", str(seeded_db)])
        assert result.exit_code == 0
        assert "Recent Runs" in result.output
        assert "Quantum computing" in result.output

    def test_stats_per_run(self, seeded_db: Path) -> None:
        result = runner.invoke(
            app, ["stats", "--db", str(seeded_db), "--run", "run-001"]
        )
        assert result.exit_code == 0
        assert "run-001" in result.output.lower() or "Run" in result.output

    def test_stats_per_run_shows_claims(self, seeded_db: Path) -> None:
        result = runner.invoke(
            app, ["stats", "--db", str(seeded_db), "--run", "run-001"]
        )
        assert result.exit_code == 0
        assert "Claims" in result.output

    def test_stats_per_run_shows_tokens(self, seeded_db: Path) -> None:
        result = runner.invoke(
            app, ["stats", "--db", str(seeded_db), "--run", "run-001"]
        )
        assert result.exit_code == 0
        assert "Token Usage" in result.output
        assert "7,000" in result.output

    def test_stats_empty_db(self, tmp_path: Path) -> None:
        from recon.db import get_db

        db_path = tmp_path / "empty.db"
        conn = get_db(str(db_path))
        conn.close()
        result = runner.invoke(app, ["stats", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "Total runs" in result.output


class TestReverify:
    """Tests for ``recon reverify``."""

    def test_reverify_no_db_exits(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nonexistent.db"
        result = runner.invoke(app, ["reverify", "--db", str(db_path)])
        assert result.exit_code == 1

    def test_reverify_finds_stale_claims(self, seeded_db: Path) -> None:
        """clm-old was verified in 2024 — definitely stale at 30 days."""
        result = runner.invoke(
            app, ["reverify", "--db", str(seeded_db), "--days", "30"]
        )
        assert result.exit_code == 0
        assert "clm-old" in result.output
        assert "Stale Claims" in result.output

    def test_reverify_no_stale_with_high_threshold(self, seeded_db: Path) -> None:
        """With --days=99999, nothing should be stale."""
        result = runner.invoke(
            app, ["reverify", "--db", str(seeded_db), "--days", "99999"]
        )
        assert result.exit_code == 0
        assert "No stale claims" in result.output

    def test_reverify_shows_recommendation(self, seeded_db: Path) -> None:
        result = runner.invoke(
            app, ["reverify", "--db", str(seeded_db), "--days", "30"]
        )
        assert result.exit_code == 0
        assert "re-verification" in result.output.lower()

    def test_reverify_limit(self, seeded_db: Path) -> None:
        result = runner.invoke(
            app, ["reverify", "--db", str(seeded_db), "--days", "30", "--limit", "1"]
        )
        assert result.exit_code == 0
        # Should still work, just limited results
        assert "clm-old" in result.output

    def test_reverify_topic_filter(self, seeded_db: Path) -> None:
        """clm-old has no topic_tags, so filtering by a topic should exclude it."""
        result = runner.invoke(
            app, ["reverify", "--db", str(seeded_db), "--days", "30", "--topic", "xyznonexistent"]
        )
        assert result.exit_code == 0
        assert "No stale claims" in result.output

    def test_reverify_empty_db(self, tmp_path: Path) -> None:
        from recon.db import get_db

        db_path = tmp_path / "empty.db"
        conn = get_db(str(db_path))
        conn.close()
        result = runner.invoke(app, ["reverify", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "No stale claims" in result.output
