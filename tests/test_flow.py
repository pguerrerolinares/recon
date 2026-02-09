"""Tests for recon.flow_builder.

These tests mock CrewAI to avoid real API calls. They verify that:
1. The flow builder creates correct crew configurations
2. The pipeline handles errors gracefully
3. Progress tracking works
4. The audit logger records events
"""

from __future__ import annotations

import sqlite3  # noqa: TC003
from pathlib import Path  # noqa: TC003
from unittest.mock import MagicMock, patch

import pytest

from recon.callbacks.audit import AuditLogger
from recon.callbacks.progress import ProgressTracker
from recon.config import Depth, KnowledgeConfig, ReconPlan


class TestAuditLogger:
    def test_log_and_read(self, tmp_path: Path) -> None:
        logger = AuditLogger(output_dir=str(tmp_path))
        logger.log(
            phase="investigation", agent="researcher", action="tool_call", detail="Searched for X"
        )
        logger.log(phase="investigation", agent="researcher", action="task_end")

        entries = logger.get_entries()
        assert len(entries) == 2
        assert entries[0]["action"] == "tool_call"

    def test_log_persists_to_disk(self, tmp_path: Path) -> None:
        logger = AuditLogger(output_dir=str(tmp_path))
        logger.log(phase="test", agent="test", action="test")

        # Read from disk
        entries = logger.read_log()
        assert len(entries) == 1

    def test_phase_start_end(self, tmp_path: Path) -> None:
        logger = AuditLogger(output_dir=str(tmp_path))
        logger.log_phase_start("investigation")
        logger.log_phase_end("investigation", output_files=["research/test.md"])

        entries = logger.get_entries()
        assert entries[0]["action"] == "phase_start"
        assert entries[1]["action"] == "phase_end"
        assert "research/test.md" in entries[1]["metadata"]["output_files"]

    def test_error_logging(self, tmp_path: Path) -> None:
        logger = AuditLogger(output_dir=str(tmp_path))
        logger.log_error("investigation", "researcher", "API timeout")

        entries = logger.get_entries()
        assert entries[0]["action"] == "error"
        assert "API timeout" in entries[0]["detail"]

    def test_detail_truncation(self, tmp_path: Path) -> None:
        logger = AuditLogger(output_dir=str(tmp_path))
        long_detail = "x" * 5000
        logger.log(phase="test", agent="test", action="test", detail=long_detail)

        entries = logger.get_entries()
        assert len(entries[0]["detail"]) <= 2000


class TestAuditLoggerDB:
    """Test AuditLogger dual-write to SQLite events table."""

    def _make_conn(self, tmp_path: Path) -> sqlite3.Connection:
        from recon.db import get_db, insert_run

        conn = get_db(tmp_path / "audit-test.db")
        insert_run(
            conn,
            run_id="run-audit",
            timestamp="2026-01-15T10:00:00Z",
            topic="test",
            depth="quick",
        )
        return conn

    def test_log_writes_to_db(self, tmp_path: Path) -> None:
        conn = self._make_conn(tmp_path)
        logger = AuditLogger(output_dir=str(tmp_path), run_id="run-audit", conn=conn)
        logger.log(phase="investigation", agent="researcher", action="tool_call", detail="searched")

        from recon.db import get_events

        events = get_events(conn, run_id="run-audit")
        assert len(events) == 1
        assert events[0]["action"] == "tool_call"
        assert events[0]["phase"] == "investigation"
        assert events[0]["agent"] == "researcher"
        conn.close()

    def test_log_writes_both_sinks(self, tmp_path: Path) -> None:
        conn = self._make_conn(tmp_path)
        logger = AuditLogger(output_dir=str(tmp_path), run_id="run-audit", conn=conn)
        logger.log(phase="test", agent="a", action="x")
        logger.log(phase="test", agent="b", action="y")

        from recon.db import get_events

        # DB
        events = get_events(conn, run_id="run-audit")
        assert len(events) == 2
        # JSONL
        disk = logger.read_log()
        assert len(disk) == 2
        conn.close()

    def test_log_with_task_and_tokens(self, tmp_path: Path) -> None:
        conn = self._make_conn(tmp_path)
        logger = AuditLogger(output_dir=str(tmp_path), run_id="run-audit", conn=conn)
        logger.log(
            phase="investigation",
            agent="researcher",
            action="llm_call",
            task="market_research",
            tokens_used=1500,
        )

        from recon.db import get_events

        events = get_events(conn, run_id="run-audit")
        assert events[0]["task"] == "market_research"
        assert events[0]["tokens_used"] == 1500
        conn.close()

    def test_db_failure_does_not_break_pipeline(self, tmp_path: Path) -> None:
        """If DB write fails, JSONL should still succeed."""
        from recon.db import get_db, insert_run

        conn = get_db(tmp_path / "fail.db")
        insert_run(
            conn,
            run_id="run-fail",
            timestamp="2026-01-15T10:00:00Z",
            topic="test",
            depth="quick",
        )
        # Close the connection so DB writes will fail
        conn.close()

        logger = AuditLogger(output_dir=str(tmp_path), run_id="run-fail", conn=conn)
        # Should NOT raise
        logger.log(phase="test", agent="test", action="test")

        # JSONL still written
        disk = logger.read_log()
        assert len(disk) == 1

    def test_no_conn_jsonl_only(self, tmp_path: Path) -> None:
        """Without conn, only JSONL is written (backward-compat)."""
        logger = AuditLogger(output_dir=str(tmp_path), run_id="run-no-db")
        logger.log(phase="test", agent="test", action="test")

        entries = logger.get_entries()
        assert len(entries) == 1
        disk = logger.read_log()
        assert len(disk) == 1

    def test_phase_start_end_in_db(self, tmp_path: Path) -> None:
        conn = self._make_conn(tmp_path)
        logger = AuditLogger(output_dir=str(tmp_path), run_id="run-audit", conn=conn)
        logger.log_phase_start("investigation")
        logger.log_phase_end("investigation", output_files=["research/test.md"])

        from recon.db import get_events

        events = get_events(conn, run_id="run-audit")
        assert len(events) == 2
        assert events[0]["action"] == "phase_start"
        assert events[1]["action"] == "phase_end"
        conn.close()

    def test_metadata_stored_in_db(self, tmp_path: Path) -> None:
        conn = self._make_conn(tmp_path)
        logger = AuditLogger(output_dir=str(tmp_path), run_id="run-audit", conn=conn)
        logger.log(
            phase="test",
            agent="test",
            action="test",
            metadata={"key": "value", "count": 42},
        )

        from recon.db import get_events

        events = get_events(conn, run_id="run-audit")
        import json

        meta = json.loads(events[0]["metadata_json"])
        assert meta["key"] == "value"
        assert meta["count"] == 42
        conn.close()


class TestRecordTokenUsage:
    """Test _record_token_usage helper."""

    def test_records_usage(self, tmp_path: Path) -> None:
        from recon.db import get_db, get_token_usage_by_run, insert_run
        from recon.flow_builder import _record_token_usage

        conn = get_db(tmp_path / "token.db")
        insert_run(
            conn, run_id="run-tok", timestamp="2026-01-15T10:00:00Z",
            topic="test", depth="quick", model="kimi-k2.5",
        )

        mock_output = MagicMock()
        mock_output.token_usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
            "cached_tokens": 200,
            "successful_requests": 3,
        }

        _record_token_usage(conn, "run-tok", "investigation", "kimi-k2.5", mock_output)

        usage = get_token_usage_by_run(conn, "run-tok")
        assert len(usage) == 1
        assert usage[0]["prompt_tokens"] == 1000
        assert usage[0]["completion_tokens"] == 500
        assert usage[0]["total_tokens"] == 1500
        assert usage[0]["cached_tokens"] == 200
        assert usage[0]["successful_requests"] == 3
        assert usage[0]["estimated_cost_usd"] is not None
        conn.close()

    def test_no_conn_is_noop(self) -> None:
        from recon.flow_builder import _record_token_usage

        mock_output = MagicMock()
        mock_output.token_usage = {"total_tokens": 100}
        # Should not raise
        _record_token_usage(None, "run-x", "test", "model", mock_output)

    def test_no_token_usage_attr(self, tmp_path: Path) -> None:
        from recon.db import get_db, get_token_usage_by_run, insert_run
        from recon.flow_builder import _record_token_usage

        conn = get_db(tmp_path / "token.db")
        insert_run(
            conn, run_id="run-no", timestamp="2026-01-15T10:00:00Z",
            topic="test", depth="quick",
        )

        # CrewOutput without token_usage attribute
        _record_token_usage(conn, "run-no", "investigation", "model", "just a string")

        usage = get_token_usage_by_run(conn, "run-no")
        assert len(usage) == 0
        conn.close()


class TestOpenDB:
    """Test _open_db helper."""

    def test_opens_when_enabled(self, tmp_path: Path) -> None:
        from recon.config import KnowledgeConfig
        from recon.flow_builder import _open_db

        plan = ReconPlan(
            topic="Test",
            knowledge=KnowledgeConfig(
                enabled=True,
                db_path=str(tmp_path / "knowledge.db"),
            ),
        )
        conn = _open_db(plan)
        assert conn is not None
        conn.close()

    def test_none_when_disabled(self) -> None:
        from recon.config import KnowledgeConfig
        from recon.flow_builder import _open_db

        plan = ReconPlan(
            topic="Test",
            knowledge=KnowledgeConfig(enabled=False),
        )
        conn = _open_db(plan)
        assert conn is None


class TestQueryPriorKnowledge:
    """Test _query_prior_knowledge helper."""

    def test_returns_prior_claims(self, tmp_path: Path) -> None:
        from recon.db import get_db, insert_run, upsert_claim
        from recon.flow_builder import _query_prior_knowledge

        conn = get_db(tmp_path / "prior.db")
        insert_run(
            conn, run_id="run-old", timestamp="2026-01-10T10:00:00Z",
            topic="AI agents", depth="standard",
        )
        upsert_claim(
            conn, claim_id="c-old-1", text="CrewAI has 44K GitHub stars",
            verification_status="VERIFIED", confidence=0.92,
            run_id="run-old", timestamp="2026-01-10T10:05:00Z",
            topic_tags=["AI", "agents"],
        )

        plan = ReconPlan(
            topic="AI agents",
            knowledge=KnowledgeConfig(
                enabled=True, db_path=str(tmp_path / "prior.db")
            ),
        )
        audit = AuditLogger(output_dir=str(tmp_path), run_id="run-new")

        result = _query_prior_knowledge(conn, plan, audit)
        assert result is not None
        assert "PRIOR VERIFIED CLAIMS" in result
        assert "CrewAI" in result
        conn.close()

    def test_returns_none_when_disabled(self, tmp_path: Path) -> None:
        from recon.flow_builder import _query_prior_knowledge

        plan = ReconPlan(
            topic="Test",
            knowledge=KnowledgeConfig(enabled=False),
        )
        audit = AuditLogger(output_dir=str(tmp_path), run_id="run-test")

        result = _query_prior_knowledge(None, plan, audit)
        assert result is None

    def test_returns_none_when_no_results(self, tmp_path: Path) -> None:
        from recon.db import get_db
        from recon.flow_builder import _query_prior_knowledge

        conn = get_db(tmp_path / "empty.db")
        plan = ReconPlan(
            topic="completely unknown topic xyz",
            knowledge=KnowledgeConfig(
                enabled=True, db_path=str(tmp_path / "empty.db")
            ),
        )
        audit = AuditLogger(output_dir=str(tmp_path), run_id="run-empty")

        result = _query_prior_knowledge(conn, plan, audit)
        assert result is None
        conn.close()


class TestProgressTracker:
    def test_phase_lifecycle(self) -> None:
        tracker = ProgressTracker()
        tracker.phase_start("investigation")
        tracker.agent_start("Market Research", "research/market.md")
        tracker.agent_end("Market Research", "research/market.md")
        tracker.phase_end("investigation", "research/market.md")

        assert len(tracker.events) == 4
        assert tracker.events[0]["type"] == "phase_start"
        assert tracker.events[-1]["type"] == "phase_end"

    def test_error_tracking(self) -> None:
        tracker = ProgressTracker()
        tracker.error("investigation", "API key not set")

        assert len(tracker.events) == 1
        assert tracker.events[0]["type"] == "error"

    def test_pipeline_timing(self) -> None:
        tracker = ProgressTracker()
        tracker.pipeline_start("Test topic")
        tracker.pipeline_end()

        assert tracker.events[-1]["type"] == "pipeline_end"
        # Should have recorded elapsed time
        assert "s" in tracker.events[-1]["detail"]


class TestFlowBuilder:
    """Test flow_builder.build_and_run with mocked CrewAI."""

    @patch("recon.flow_builder.create_search_tools")
    @patch("recon.flow_builder.create_llm")
    @patch("recon.flow_builder.build_synthesis_crew")
    @patch("recon.flow_builder.build_verification_crew")
    @patch("recon.flow_builder.build_investigation_crew")
    def test_full_pipeline(
        self,
        mock_inv_crew: MagicMock,
        mock_ver_crew: MagicMock,
        mock_syn_crew: MagicMock,
        mock_llm: MagicMock,
        mock_search: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Full pipeline with all 3 phases."""
        from recon.flow_builder import build_and_run

        plan = ReconPlan(
            topic="Test",
            depth=Depth.QUICK,
            verify=True,
            research_dir=str(tmp_path / "research"),
            output_dir=str(tmp_path / "output"),
            verification_dir=str(tmp_path / "verification"),
        )

        # Mock all crews to return successfully
        mock_inv = MagicMock()
        mock_inv.kickoff.return_value = "investigation done"
        mock_inv_crew.return_value = mock_inv

        mock_ver = MagicMock()
        mock_ver.kickoff.return_value = "verification done"
        mock_ver_crew.return_value = mock_ver

        mock_syn = MagicMock()
        mock_syn.kickoff.return_value = "synthesis done"
        mock_syn_crew.return_value = mock_syn

        mock_llm.return_value = MagicMock()
        mock_search.return_value = [MagicMock()]

        build_and_run(plan, verbose=False)

        # All 3 crews should have been built and kicked off
        mock_inv_crew.assert_called_once()
        mock_ver_crew.assert_called_once()
        mock_syn_crew.assert_called_once()
        mock_inv.kickoff.assert_called_once()
        mock_ver.kickoff.assert_called_once()
        mock_syn.kickoff.assert_called_once()

    @patch("recon.flow_builder.create_search_tools")
    @patch("recon.flow_builder.create_llm")
    @patch("recon.flow_builder.build_synthesis_crew")
    @patch("recon.flow_builder.build_investigation_crew")
    def test_pipeline_without_verification(
        self,
        mock_inv_crew: MagicMock,
        mock_syn_crew: MagicMock,
        mock_llm: MagicMock,
        mock_search: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Pipeline with verify=False skips verification."""
        from recon.flow_builder import build_and_run

        plan = ReconPlan(
            topic="Test",
            depth=Depth.QUICK,
            verify=False,
            research_dir=str(tmp_path / "research"),
            output_dir=str(tmp_path / "output"),
            verification_dir=str(tmp_path / "verification"),
        )

        mock_inv = MagicMock()
        mock_inv.kickoff.return_value = "done"
        mock_inv_crew.return_value = mock_inv

        mock_syn = MagicMock()
        mock_syn.kickoff.return_value = "done"
        mock_syn_crew.return_value = mock_syn

        mock_llm.return_value = MagicMock()
        mock_search.return_value = [MagicMock()]

        build_and_run(plan, verbose=False)

        mock_inv_crew.assert_called_once()
        mock_syn_crew.assert_called_once()

    @patch("recon.flow_builder.create_search_tools")
    @patch("recon.flow_builder.create_llm")
    @patch("recon.flow_builder.build_synthesis_crew")
    @patch("recon.flow_builder.build_verification_crew")
    @patch("recon.flow_builder.build_investigation_crew")
    def test_investigation_failure_raises(
        self,
        mock_inv_crew: MagicMock,
        mock_ver_crew: MagicMock,
        mock_syn_crew: MagicMock,
        mock_llm: MagicMock,
        mock_search: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Investigation failure should propagate."""
        from recon.flow_builder import build_and_run

        plan = ReconPlan(
            topic="Test",
            depth=Depth.QUICK,
            research_dir=str(tmp_path / "research"),
            output_dir=str(tmp_path / "output"),
        )

        mock_inv = MagicMock()
        mock_inv.kickoff.side_effect = RuntimeError("API rate limit")
        mock_inv_crew.return_value = mock_inv

        mock_llm.return_value = MagicMock()
        mock_search.return_value = [MagicMock()]

        with pytest.raises(RuntimeError, match="API rate limit"):
            build_and_run(plan, verbose=False)

    @patch("recon.flow_builder.create_search_tools")
    @patch("recon.flow_builder.create_llm")
    @patch("recon.flow_builder.build_synthesis_crew")
    @patch("recon.flow_builder.build_verification_crew")
    @patch("recon.flow_builder.build_investigation_crew")
    def test_verification_failure_continues(
        self,
        mock_inv_crew: MagicMock,
        mock_ver_crew: MagicMock,
        mock_syn_crew: MagicMock,
        mock_llm: MagicMock,
        mock_search: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verification failure should continue to synthesis."""
        from recon.flow_builder import build_and_run

        plan = ReconPlan(
            topic="Test",
            depth=Depth.QUICK,
            verify=True,
            research_dir=str(tmp_path / "research"),
            output_dir=str(tmp_path / "output"),
            verification_dir=str(tmp_path / "verification"),
        )

        mock_inv = MagicMock()
        mock_inv.kickoff.return_value = "done"
        mock_inv_crew.return_value = mock_inv

        mock_ver = MagicMock()
        mock_ver.kickoff.side_effect = RuntimeError("Verification timeout")
        mock_ver_crew.return_value = mock_ver

        mock_syn = MagicMock()
        mock_syn.kickoff.return_value = "done"
        mock_syn_crew.return_value = mock_syn

        mock_llm.return_value = MagicMock()
        mock_search.return_value = [MagicMock()]

        # Should NOT raise -- verification failure is non-fatal
        build_and_run(plan, verbose=False)

        # Synthesis should still run
        mock_syn.kickoff.assert_called_once()


class TestIncrementalSkip:
    """Test incremental pipeline mode (auto-skip phases with existing output)."""

    @patch("recon.flow_builder.create_search_tools")
    @patch("recon.flow_builder.create_llm")
    @patch("recon.flow_builder.build_synthesis_crew")
    @patch("recon.flow_builder.build_verification_crew")
    @patch("recon.flow_builder.build_investigation_crew")
    def test_skip_investigation_when_files_exist(
        self,
        mock_inv_crew: MagicMock,
        mock_ver_crew: MagicMock,
        mock_syn_crew: MagicMock,
        mock_llm: MagicMock,
        mock_search: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Investigation should be skipped when research files already exist."""
        from recon.flow_builder import build_and_run

        research_dir = tmp_path / "research"
        research_dir.mkdir()
        (research_dir / "general-overview.md").write_text("# Existing research\nContent here.\n")

        plan = ReconPlan(
            topic="Test",
            depth=Depth.QUICK,
            verify=False,
            research_dir=str(research_dir),
            output_dir=str(tmp_path / "output"),
            verification_dir=str(tmp_path / "verification"),
        )

        mock_syn = MagicMock()
        mock_syn.kickoff.return_value = "done"
        mock_syn_crew.return_value = mock_syn

        mock_llm.return_value = MagicMock()
        mock_search.return_value = [MagicMock()]

        build_and_run(plan, verbose=False)

        # Investigation should NOT have been called
        mock_inv_crew.assert_not_called()
        # Synthesis should still run
        mock_syn.kickoff.assert_called_once()

    @patch("recon.flow_builder.create_search_tools")
    @patch("recon.flow_builder.create_llm")
    @patch("recon.flow_builder.build_synthesis_crew")
    @patch("recon.flow_builder.build_verification_crew")
    @patch("recon.flow_builder.build_investigation_crew")
    def test_force_reruns_investigation(
        self,
        mock_inv_crew: MagicMock,
        mock_ver_crew: MagicMock,
        mock_syn_crew: MagicMock,
        mock_llm: MagicMock,
        mock_search: MagicMock,
        tmp_path: Path,
    ) -> None:
        """With force=True, investigation should run even if files exist."""
        from recon.flow_builder import build_and_run

        research_dir = tmp_path / "research"
        research_dir.mkdir()
        (research_dir / "general-overview.md").write_text("# Existing research\nContent.\n")

        plan = ReconPlan(
            topic="Test",
            depth=Depth.QUICK,
            verify=False,
            research_dir=str(research_dir),
            output_dir=str(tmp_path / "output"),
            verification_dir=str(tmp_path / "verification"),
        )

        mock_inv = MagicMock()
        mock_inv.kickoff.return_value = "done"
        mock_inv_crew.return_value = mock_inv

        mock_syn = MagicMock()
        mock_syn.kickoff.return_value = "done"
        mock_syn_crew.return_value = mock_syn

        mock_llm.return_value = MagicMock()
        mock_search.return_value = [MagicMock()]

        build_and_run(plan, verbose=False, force=True)

        # Investigation SHOULD have been called despite existing files
        mock_inv_crew.assert_called_once()
        mock_inv.kickoff.assert_called_once()

    @patch("recon.flow_builder.create_search_tools")
    @patch("recon.flow_builder.create_llm")
    @patch("recon.flow_builder.build_synthesis_crew")
    @patch("recon.flow_builder.build_verification_crew")
    @patch("recon.flow_builder.build_investigation_crew")
    def test_skip_verification_when_report_exists(
        self,
        mock_inv_crew: MagicMock,
        mock_ver_crew: MagicMock,
        mock_syn_crew: MagicMock,
        mock_llm: MagicMock,
        mock_search: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verification should be skipped when report already exists."""
        from recon.flow_builder import build_and_run

        research_dir = tmp_path / "research"
        research_dir.mkdir()
        (research_dir / "general-overview.md").write_text("# Research\nContent.\n")

        verification_dir = tmp_path / "verification"
        verification_dir.mkdir()
        (verification_dir / "report.md").write_text("# Verification Report\nAll verified.\n")

        plan = ReconPlan(
            topic="Test",
            depth=Depth.QUICK,
            verify=True,
            research_dir=str(research_dir),
            output_dir=str(tmp_path / "output"),
            verification_dir=str(verification_dir),
        )

        mock_syn = MagicMock()
        mock_syn.kickoff.return_value = "done"
        mock_syn_crew.return_value = mock_syn

        mock_llm.return_value = MagicMock()
        mock_search.return_value = [MagicMock()]

        build_and_run(plan, verbose=False)

        # Neither investigation nor verification should have been called
        mock_inv_crew.assert_not_called()
        mock_ver_crew.assert_not_called()
        # Synthesis should still run
        mock_syn.kickoff.assert_called_once()

    @patch("recon.flow_builder.create_search_tools")
    @patch("recon.flow_builder.create_llm")
    @patch("recon.flow_builder.build_synthesis_crew")
    @patch("recon.flow_builder.build_investigation_crew")
    def test_skip_synthesis_when_report_exists(
        self,
        mock_inv_crew: MagicMock,
        mock_syn_crew: MagicMock,
        mock_llm: MagicMock,
        mock_search: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Synthesis should be skipped when final report already exists."""
        from recon.flow_builder import build_and_run

        research_dir = tmp_path / "research"
        research_dir.mkdir()
        (research_dir / "general-overview.md").write_text("# Research\nContent.\n")

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "final-report.md").write_text("# Final Report\nDone.\n")

        plan = ReconPlan(
            topic="Test",
            depth=Depth.QUICK,
            verify=False,
            research_dir=str(research_dir),
            output_dir=str(output_dir),
            verification_dir=str(tmp_path / "verification"),
        )

        mock_llm.return_value = MagicMock()
        mock_search.return_value = [MagicMock()]

        build_and_run(plan, verbose=False)

        # Nothing should have been called -- all phases skipped
        mock_inv_crew.assert_not_called()
        mock_syn_crew.assert_not_called()


class TestHasPhaseOutput:
    """Test _has_phase_output helper."""

    def test_nonexistent_directory(self) -> None:
        from recon.flow_builder import _has_phase_output

        assert _has_phase_output("/nonexistent/dir") == []

    def test_empty_directory(self, tmp_path: Path) -> None:
        from recon.flow_builder import _has_phase_output

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert _has_phase_output(str(empty_dir)) == []

    def test_directory_with_files(self, tmp_path: Path) -> None:
        from recon.flow_builder import _has_phase_output

        d = tmp_path / "research"
        d.mkdir()
        (d / "file1.md").write_text("content")
        (d / "file2.md").write_text("content")
        (d / "file3.txt").write_text("not markdown")

        result = _has_phase_output(str(d))
        assert len(result) == 2

    def test_ignores_empty_files(self, tmp_path: Path) -> None:
        from recon.flow_builder import _has_phase_output

        d = tmp_path / "research"
        d.mkdir()
        (d / "empty.md").write_text("")
        (d / "real.md").write_text("content")

        result = _has_phase_output(str(d))
        assert len(result) == 1
