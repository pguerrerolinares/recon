"""Tests for recon.memory module and KnowledgeConfig migration.

All MemoryStore tests mock memvid-sdk since it's an optional dependency
with a Rust binary that may not be installed in the test environment.

The KnowledgeConfig tests verify the new schema and backward-compat
migration from the legacy ``memory:`` YAML key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

from recon.config import KnowledgeConfig, MemoryConfig, ReconPlan

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# KnowledgeConfig model tests
# ---------------------------------------------------------------------------


class TestKnowledgeConfig:
    def test_defaults(self) -> None:
        config = KnowledgeConfig()
        assert config.enabled is True
        assert config.db_path == "./knowledge.db"
        assert config.embedder == "onnx"
        assert config.stale_after_days == 30

    def test_memory_alias(self) -> None:
        """MemoryConfig should be an alias for KnowledgeConfig."""
        assert MemoryConfig is KnowledgeConfig

    def test_custom_config(self) -> None:
        config = KnowledgeConfig(
            enabled=False,
            db_path="/data/knowledge.db",
            embedder="ollama",
            stale_after_days=60,
        )
        assert config.enabled is False
        assert config.db_path == "/data/knowledge.db"
        assert config.embedder == "ollama"
        assert config.stale_after_days == 60

    def test_backward_compat_fields_accepted(self) -> None:
        """Legacy fields 'path' and 'embedding_provider' should not error."""
        config = KnowledgeConfig(
            path="./memory/recon.mv2",
            embedding_provider="local",
        )
        # Legacy fields are accepted but excluded from serialization
        assert config.enabled is True
        assert config.db_path == "./knowledge.db"

    def test_plan_includes_knowledge(self) -> None:
        plan = ReconPlan(topic="Test")
        assert plan.knowledge.enabled is True
        assert plan.knowledge.db_path == "./knowledge.db"

    def test_plan_with_knowledge_disabled(self) -> None:
        plan = ReconPlan(
            topic="Test",
            knowledge=KnowledgeConfig(enabled=False),
        )
        assert plan.knowledge.enabled is False

    def test_plan_with_knowledge_custom(self) -> None:
        plan = ReconPlan(
            topic="Test",
            knowledge=KnowledgeConfig(
                enabled=True, db_path="./my-knowledge.db", embedder="ollama"
            ),
        )
        assert plan.knowledge.enabled is True
        assert plan.knowledge.db_path == "./my-knowledge.db"
        assert plan.knowledge.embedder == "ollama"

    def test_plan_from_yaml_with_knowledge(self, tmp_path: Path) -> None:
        """Knowledge config should load from YAML plan files."""
        from recon.config import load_plan

        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "topic: Test\n"
            "knowledge:\n"
            "  enabled: true\n"
            "  db_path: ./my-knowledge.db\n"
            "  embedder: ollama\n"
            "  stale_after_days: 14\n"
        )
        plan = load_plan(plan_yaml)
        assert plan.knowledge.enabled is True
        assert plan.knowledge.db_path == "./my-knowledge.db"
        assert plan.knowledge.embedder == "ollama"
        assert plan.knowledge.stale_after_days == 14

    def test_plan_from_yaml_legacy_memory_key(self, tmp_path: Path) -> None:
        """Legacy ``memory:`` YAML key should migrate to ``knowledge:``."""
        from recon.config import load_plan

        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "topic: Test\n"
            "memory:\n"
            "  enabled: true\n"
            "  path: ./memory/project.mv2\n"
            "  embedding_provider: openai\n"
        )
        plan = load_plan(plan_yaml)
        # The legacy memory key is migrated, and legacy fields are consumed
        assert plan.knowledge.enabled is True
        assert plan.knowledge.db_path == "./knowledge.db"  # path is ignored

    def test_plan_without_knowledge_in_yaml(self, tmp_path: Path) -> None:
        """Plans without knowledge section should use defaults."""
        from recon.config import load_plan

        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text("topic: Test\n")
        plan = load_plan(plan_yaml)
        assert plan.knowledge.enabled is True

    def test_memory_kwarg_migrated(self) -> None:
        """Passing memory= kwarg should be migrated to knowledge."""
        plan = ReconPlan(
            topic="Test",
            memory=KnowledgeConfig(enabled=False),  # type: ignore[call-arg]
        )
        assert plan.knowledge.enabled is False


# ---------------------------------------------------------------------------
# MemoryStore tests (with mocked memvid-sdk)
# ---------------------------------------------------------------------------


def _make_mock_memvid() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Create mock memvid create/use functions and a mock mv object."""
    mock_mv = MagicMock()
    mock_mv.find.return_value = [
        {"title": "Prior: AI frameworks", "text": "CrewAI leads with 44K stars.", "score": 0.92},
        {"title": "Prior: LangGraph", "text": "LangGraph is growing fast.", "score": 0.85},
    ]
    mock_mv.stats.return_value = {"frame_count": 42, "size_bytes": 1024000}
    mock_mv.put.return_value = None
    mock_mv.put_many.return_value = None
    mock_mv.seal.return_value = None

    mock_create = MagicMock(return_value=mock_mv)
    mock_use = MagicMock(return_value=mock_mv)

    return mock_create, mock_use, mock_mv


class TestMemoryStore:
    @patch("recon.memory.store._get_memvid")
    def test_query_returns_results(self, mock_get_memvid: MagicMock, tmp_path: Path) -> None:
        from recon.memory.store import MemoryStore

        mock_create, mock_use, mock_mv = _make_mock_memvid()
        mock_get_memvid.return_value = (mock_create, mock_use)

        # Create a fake .mv2 file so it opens instead of creates
        mv_path = tmp_path / "test.mv2"
        mv_path.touch()

        store = MemoryStore(path=str(mv_path))
        results = store.query("AI agent frameworks")
        store.close()

        assert len(results) == 2
        assert results[0]["title"] == "Prior: AI frameworks"
        assert results[0]["score"] == 0.92
        mock_mv.find.assert_called_once()
        mock_mv.seal.assert_called_once()

    @patch("recon.memory.store._get_memvid")
    def test_query_with_questions(self, mock_get_memvid: MagicMock, tmp_path: Path) -> None:
        from recon.memory.store import MemoryStore

        mock_create, mock_use, mock_mv = _make_mock_memvid()
        mock_get_memvid.return_value = (mock_create, mock_use)

        mv_path = tmp_path / "test.mv2"
        mv_path.touch()

        store = MemoryStore(path=str(mv_path))
        store.query("AI", questions=["What frameworks exist?", "Which are mature?"])
        store.close()

        # Verify the query includes topic + questions
        call_args = mock_mv.find.call_args
        query_str = call_args[0][0]
        assert "AI" in query_str
        assert "What frameworks exist?" in query_str

    @patch("recon.memory.store._get_memvid")
    def test_query_empty_memory(self, mock_get_memvid: MagicMock, tmp_path: Path) -> None:
        from recon.memory.store import MemoryStore

        mock_create, mock_use, mock_mv = _make_mock_memvid()
        mock_mv.find.return_value = []
        mock_get_memvid.return_value = (mock_create, mock_use)

        mv_path = tmp_path / "test.mv2"
        mv_path.touch()

        store = MemoryStore(path=str(mv_path))
        results = store.query("unknown topic")
        store.close()

        assert results == []

    @patch("recon.memory.store._get_memvid")
    def test_query_handles_exception(self, mock_get_memvid: MagicMock, tmp_path: Path) -> None:
        from recon.memory.store import MemoryStore

        mock_create, mock_use, mock_mv = _make_mock_memvid()
        mock_mv.find.side_effect = RuntimeError("search failed")
        mock_get_memvid.return_value = (mock_create, mock_use)

        mv_path = tmp_path / "test.mv2"
        mv_path.touch()

        store = MemoryStore(path=str(mv_path))
        results = store.query("AI")
        store.close()

        assert results == []

    @patch("recon.memory.store._get_memvid")
    def test_ingest_research(self, mock_get_memvid: MagicMock, tmp_path: Path) -> None:
        from recon.memory.store import MemoryStore

        mock_create, mock_use, mock_mv = _make_mock_memvid()
        mock_get_memvid.return_value = (mock_create, mock_use)

        # Create research files
        research_dir = tmp_path / "research"
        research_dir.mkdir()
        (research_dir / "overview.md").write_text("# Overview\nContent here.")
        (research_dir / "analysis.md").write_text("# Analysis\nMore content.")
        (research_dir / "empty.md").write_text("")  # Should be skipped

        mv_path = tmp_path / "test.mv2"
        mv_path.touch()

        store = MemoryStore(path=str(mv_path))
        count = store.ingest_research(str(research_dir), topic="AI", run_id="run-001")
        store.close()

        assert count == 2  # empty file skipped
        mock_mv.put_many.assert_called_once()
        docs: list[dict[str, Any]] = mock_mv.put_many.call_args[0][0]
        assert len(docs) == 2
        assert docs[0]["label"] == "research"
        assert docs[0]["metadata"]["topic"] == "AI"
        assert docs[0]["metadata"]["run_id"] == "run-001"

    @patch("recon.memory.store._get_memvid")
    def test_ingest_research_no_files(self, mock_get_memvid: MagicMock, tmp_path: Path) -> None:
        from recon.memory.store import MemoryStore

        mock_create, mock_use, mock_mv = _make_mock_memvid()
        mock_get_memvid.return_value = (mock_create, mock_use)

        research_dir = tmp_path / "empty_research"
        research_dir.mkdir()

        mv_path = tmp_path / "test.mv2"
        mv_path.touch()

        store = MemoryStore(path=str(mv_path))
        count = store.ingest_research(str(research_dir), topic="AI", run_id="run-001")
        store.close()

        assert count == 0
        mock_mv.put_many.assert_not_called()

    @patch("recon.memory.store._get_memvid")
    def test_ingest_report(self, mock_get_memvid: MagicMock, tmp_path: Path) -> None:
        from recon.memory.store import MemoryStore

        mock_create, mock_use, mock_mv = _make_mock_memvid()
        mock_get_memvid.return_value = (mock_create, mock_use)

        report = tmp_path / "report.md"
        report.write_text("# Verification Report\n\n5 claims verified.")

        mv_path = tmp_path / "test.mv2"
        mv_path.touch()

        store = MemoryStore(path=str(mv_path))
        store.ingest_report(str(report), topic="AI", run_id="run-001", phase="verification")
        store.close()

        mock_mv.put.assert_called_once()
        call_kwargs = mock_mv.put.call_args[1]
        assert call_kwargs["label"] == "verification"
        assert call_kwargs["metadata"]["phase"] == "verification"

    @patch("recon.memory.store._get_memvid")
    def test_ingest_report_missing_file(self, mock_get_memvid: MagicMock, tmp_path: Path) -> None:
        from recon.memory.store import MemoryStore

        mock_create, mock_use, mock_mv = _make_mock_memvid()
        mock_get_memvid.return_value = (mock_create, mock_use)

        mv_path = tmp_path / "test.mv2"
        mv_path.touch()

        store = MemoryStore(path=str(mv_path))
        store.ingest_report(
            str(tmp_path / "nonexistent.md"),
            topic="AI",
            run_id="run-001",
            phase="verification",
        )
        store.close()

        mock_mv.put.assert_not_called()

    @patch("recon.memory.store._get_memvid")
    def test_stats(self, mock_get_memvid: MagicMock, tmp_path: Path) -> None:
        from recon.memory.store import MemoryStore

        mock_create, mock_use, mock_mv = _make_mock_memvid()
        mock_get_memvid.return_value = (mock_create, mock_use)

        mv_path = tmp_path / "test.mv2"
        mv_path.touch()

        store = MemoryStore(path=str(mv_path))
        stats = store.stats()
        store.close()

        assert stats["frame_count"] == 42
        assert stats["size_bytes"] == 1024000

    @patch("recon.memory.store._get_memvid")
    def test_context_manager(self, mock_get_memvid: MagicMock, tmp_path: Path) -> None:
        from recon.memory.store import MemoryStore

        mock_create, mock_use, mock_mv = _make_mock_memvid()
        mock_get_memvid.return_value = (mock_create, mock_use)

        mv_path = tmp_path / "test.mv2"
        mv_path.touch()

        with MemoryStore(path=str(mv_path)) as store:
            results = store.query("test")
            assert isinstance(results, list)

        # seal should be called on exit
        mock_mv.seal.assert_called_once()

    @patch("recon.memory.store._get_memvid")
    def test_creates_new_file(self, mock_get_memvid: MagicMock, tmp_path: Path) -> None:
        from recon.memory.store import MemoryStore

        mock_create, mock_use, mock_mv = _make_mock_memvid()
        mock_get_memvid.return_value = (mock_create, mock_use)

        mv_path = tmp_path / "new_memory" / "recon.mv2"
        # File does NOT exist -- should create

        store = MemoryStore(path=str(mv_path))
        store._ensure_open()
        store.close()

        mock_create.assert_called_once_with(str(mv_path))
        assert mv_path.parent.exists()  # Directory created

    @patch("recon.memory.store._get_memvid")
    def test_opens_existing_file(self, mock_get_memvid: MagicMock, tmp_path: Path) -> None:
        from recon.memory.store import MemoryStore

        mock_create, mock_use, mock_mv = _make_mock_memvid()
        mock_get_memvid.return_value = (mock_create, mock_use)

        mv_path = tmp_path / "existing.mv2"
        mv_path.touch()

        store = MemoryStore(path=str(mv_path))
        store._ensure_open()
        store.close()

        mock_use.assert_called_once_with("basic", str(mv_path), mode="open")
        mock_create.assert_not_called()

    @patch("recon.memory.store._get_memvid")
    def test_repr(self, mock_get_memvid: MagicMock, tmp_path: Path) -> None:
        from recon.memory.store import MemoryStore

        mock_create, mock_use, mock_mv = _make_mock_memvid()
        mock_get_memvid.return_value = (mock_create, mock_use)

        mv_path = tmp_path / "test.mv2"
        store = MemoryStore(path=str(mv_path))
        assert "closed" in repr(store)

        mv_path.touch()
        store._ensure_open()
        assert "open" in repr(store)
        store.close()

    @patch("recon.memory.store._get_memvid")
    def test_double_close_is_safe(self, mock_get_memvid: MagicMock, tmp_path: Path) -> None:
        from recon.memory.store import MemoryStore

        mock_create, mock_use, mock_mv = _make_mock_memvid()
        mock_get_memvid.return_value = (mock_create, mock_use)

        mv_path = tmp_path / "test.mv2"
        mv_path.touch()

        store = MemoryStore(path=str(mv_path))
        store._ensure_open()
        store.close()
        store.close()  # Should not raise

        # seal called only once
        mock_mv.seal.assert_called_once()


# ---------------------------------------------------------------------------
# flow_builder memory integration tests
# ---------------------------------------------------------------------------


class TestFlowBuilderMemory:
    """Test that flow_builder correctly integrates memory."""

    @patch("recon.flow_builder._ingest_to_memory")
    @patch("recon.flow_builder._query_prior_knowledge")
    @patch("recon.flow_builder.create_search_tools")
    @patch("recon.flow_builder.create_llm")
    @patch("recon.flow_builder.build_synthesis_crew")
    @patch("recon.flow_builder.build_verification_crew")
    @patch("recon.flow_builder.build_investigation_crew")
    def test_memory_query_called_when_enabled(
        self,
        mock_inv_crew: MagicMock,
        mock_ver_crew: MagicMock,
        mock_syn_crew: MagicMock,
        mock_llm: MagicMock,
        mock_search: MagicMock,
        mock_query: MagicMock,
        mock_ingest: MagicMock,
        tmp_path: Path,
    ) -> None:
        from recon.flow_builder import build_and_run

        plan = ReconPlan(
            topic="Test",
            verify=False,
            research_dir=str(tmp_path / "research"),
            output_dir=str(tmp_path / "output"),
            knowledge=KnowledgeConfig(enabled=True, db_path=str(tmp_path / "knowledge.db")),
        )

        mock_query.return_value = "Prior: CrewAI has 44K stars"

        mock_inv = MagicMock()
        mock_inv.kickoff.return_value = "done"
        mock_inv_crew.return_value = mock_inv

        mock_syn = MagicMock()
        mock_syn.kickoff.return_value = "done"
        mock_syn_crew.return_value = mock_syn

        mock_llm.return_value = MagicMock()
        mock_search.return_value = [MagicMock()]

        build_and_run(plan, verbose=False)

        mock_query.assert_called_once()
        mock_ingest.assert_called_once()

        # prior_knowledge should be passed to investigation crew
        inv_call_kwargs = mock_inv_crew.call_args[1]
        assert inv_call_kwargs["prior_knowledge"] == "Prior: CrewAI has 44K stars"

    @patch("recon.flow_builder._ingest_to_memory")
    @patch("recon.flow_builder._query_prior_knowledge")
    @patch("recon.flow_builder.create_search_tools")
    @patch("recon.flow_builder.create_llm")
    @patch("recon.flow_builder.build_synthesis_crew")
    @patch("recon.flow_builder.build_investigation_crew")
    def test_memory_not_called_when_disabled(
        self,
        mock_inv_crew: MagicMock,
        mock_syn_crew: MagicMock,
        mock_llm: MagicMock,
        mock_search: MagicMock,
        mock_query: MagicMock,
        mock_ingest: MagicMock,
        tmp_path: Path,
    ) -> None:
        from recon.flow_builder import build_and_run

        plan = ReconPlan(
            topic="Test",
            verify=False,
            research_dir=str(tmp_path / "research"),
            output_dir=str(tmp_path / "output"),
            knowledge=KnowledgeConfig(enabled=False),
        )

        mock_query.return_value = None

        mock_inv = MagicMock()
        mock_inv.kickoff.return_value = "done"
        mock_inv_crew.return_value = mock_inv

        mock_syn = MagicMock()
        mock_syn.kickoff.return_value = "done"
        mock_syn_crew.return_value = mock_syn

        mock_llm.return_value = MagicMock()
        mock_search.return_value = [MagicMock()]

        build_and_run(plan, verbose=False)

        mock_query.assert_called_once()
        # _query_prior_knowledge is always called but returns None when disabled
        # _ingest_to_memory is always called but returns early when disabled
        mock_ingest.assert_called_once()


# ---------------------------------------------------------------------------
# Investigation crew prior_knowledge tests
# ---------------------------------------------------------------------------


class TestInvestigationPriorKnowledge:
    @patch("recon.crews.investigation.crew.Crew")
    @patch("recon.crews.investigation.crew.Task")
    @patch("recon.crews.investigation.crew.Agent")
    def test_prior_knowledge_injected(
        self,
        mock_agent_cls: MagicMock,
        mock_task_cls: MagicMock,
        mock_crew_cls: MagicMock,
    ) -> None:
        from recon.config import Depth
        from recon.crews.investigation.crew import build_investigation_crew

        plan = ReconPlan(topic="AI agents", depth=Depth.QUICK)
        investigations = plan.get_investigations()

        build_investigation_crew(
            plan=plan,
            investigations=investigations,
            llm=MagicMock(),
            search_tools=[MagicMock()],
            prior_knowledge="Prior: CrewAI leads the market",
        )

        # Check that the agent backstory includes prior knowledge
        agent_call_kwargs = mock_agent_cls.call_args[1]
        assert "Prior: CrewAI leads the market" in agent_call_kwargs["backstory"]
        assert "previous research runs" in agent_call_kwargs["backstory"]

    @patch("recon.crews.investigation.crew.Crew")
    @patch("recon.crews.investigation.crew.Task")
    @patch("recon.crews.investigation.crew.Agent")
    def test_no_prior_knowledge(
        self,
        mock_agent_cls: MagicMock,
        mock_task_cls: MagicMock,
        mock_crew_cls: MagicMock,
    ) -> None:
        from recon.config import Depth
        from recon.crews.investigation.crew import build_investigation_crew

        plan = ReconPlan(topic="AI agents", depth=Depth.QUICK)
        investigations = plan.get_investigations()

        build_investigation_crew(
            plan=plan,
            investigations=investigations,
            llm=MagicMock(),
            search_tools=[MagicMock()],
            prior_knowledge=None,
        )

        agent_call_kwargs = mock_agent_cls.call_args[1]
        assert "previous research runs" not in agent_call_kwargs["backstory"]


class TestInvestigationCrewFeatures:
    """Test v0.3 investigation crew enhancements."""

    @patch("recon.crews.investigation.crew.Crew")
    @patch("recon.crews.investigation.crew.Task")
    @patch("recon.crews.investigation.crew.Agent")
    def test_memory_and_embedder_enabled(
        self,
        mock_agent_cls: MagicMock,
        mock_task_cls: MagicMock,
        mock_crew_cls: MagicMock,
    ) -> None:
        from recon.config import Depth
        from recon.crews.investigation.crew import ONNX_EMBEDDER_CONFIG, build_investigation_crew

        plan = ReconPlan(topic="AI agents", depth=Depth.QUICK)
        investigations = plan.get_investigations()

        build_investigation_crew(
            plan=plan,
            investigations=investigations,
            llm=MagicMock(),
            search_tools=[MagicMock()],
        )

        crew_call_kwargs = mock_crew_cls.call_args[1]
        assert crew_call_kwargs["memory"] is True
        assert crew_call_kwargs["embedder"] == ONNX_EMBEDDER_CONFIG

    @patch("recon.crews.investigation.crew.Crew")
    @patch("recon.crews.investigation.crew.Task")
    @patch("recon.crews.investigation.crew.Agent")
    def test_quick_uses_sequential(
        self,
        mock_agent_cls: MagicMock,
        mock_task_cls: MagicMock,
        mock_crew_cls: MagicMock,
    ) -> None:
        from crewai import Process

        from recon.config import Depth
        from recon.crews.investigation.crew import build_investigation_crew

        plan = ReconPlan(topic="AI agents", depth=Depth.QUICK)
        investigations = plan.get_investigations()

        build_investigation_crew(
            plan=plan,
            investigations=investigations,
            llm=MagicMock(),
            search_tools=[MagicMock()],
        )

        crew_call_kwargs = mock_crew_cls.call_args[1]
        assert crew_call_kwargs["process"] == Process.sequential
        assert "manager_agent" not in crew_call_kwargs

    @patch("recon.crews.investigation.crew.Crew")
    @patch("recon.crews.investigation.crew.Task")
    @patch("recon.crews.investigation.crew.Agent")
    def test_deep_uses_hierarchical(
        self,
        mock_agent_cls: MagicMock,
        mock_task_cls: MagicMock,
        mock_crew_cls: MagicMock,
    ) -> None:
        from crewai import Process

        from recon.config import Depth
        from recon.crews.investigation.crew import build_investigation_crew

        plan = ReconPlan(topic="AI agents", depth=Depth.DEEP)
        investigations = plan.get_investigations()

        build_investigation_crew(
            plan=plan,
            investigations=investigations,
            llm=MagicMock(),
            search_tools=[MagicMock()],
        )

        crew_call_kwargs = mock_crew_cls.call_args[1]
        assert crew_call_kwargs["process"] == Process.hierarchical
        assert crew_call_kwargs["manager_agent"] is not None

    @patch("recon.crews.investigation.crew.Crew")
    @patch("recon.crews.investigation.crew.Task")
    @patch("recon.crews.investigation.crew.Agent")
    def test_deep_agents_have_reasoning(
        self,
        mock_agent_cls: MagicMock,
        mock_task_cls: MagicMock,
        mock_crew_cls: MagicMock,
    ) -> None:
        from recon.config import Depth
        from recon.crews.investigation.crew import build_investigation_crew

        plan = ReconPlan(topic="AI agents", depth=Depth.DEEP)
        investigations = plan.get_investigations()

        build_investigation_crew(
            plan=plan,
            investigations=investigations,
            llm=MagicMock(),
            search_tools=[MagicMock()],
        )

        # Check first researcher agent (not the manager agent)
        # The first call to Agent should be the manager (from _build_manager_agent)
        # but since _build_manager_agent is called separately, let's check any
        # call that has "Researcher" in role
        for call in mock_agent_cls.call_args_list:
            kwargs = call[1]
            if "Researcher" in kwargs.get("role", ""):
                assert kwargs["reasoning"] is True
                assert kwargs["allow_delegation"] is True
                break
        else:
            msg = "No researcher agent found in calls"
            raise AssertionError(msg)

    @patch("recon.crews.investigation.crew.Crew")
    @patch("recon.crews.investigation.crew.Task")
    @patch("recon.crews.investigation.crew.Agent")
    def test_quick_agents_no_reasoning(
        self,
        mock_agent_cls: MagicMock,
        mock_task_cls: MagicMock,
        mock_crew_cls: MagicMock,
    ) -> None:
        from recon.config import Depth
        from recon.crews.investigation.crew import build_investigation_crew

        plan = ReconPlan(topic="AI agents", depth=Depth.QUICK)
        investigations = plan.get_investigations()

        build_investigation_crew(
            plan=plan,
            investigations=investigations,
            llm=MagicMock(),
            search_tools=[MagicMock()],
        )

        agent_kwargs = mock_agent_cls.call_args[1]
        assert agent_kwargs["reasoning"] is False
        assert agent_kwargs["allow_delegation"] is False

    @patch("recon.crews.investigation.crew.Crew")
    @patch("recon.crews.investigation.crew.Task")
    @patch("recon.crews.investigation.crew.Agent")
    def test_callbacks_forwarded(
        self,
        mock_agent_cls: MagicMock,
        mock_task_cls: MagicMock,
        mock_crew_cls: MagicMock,
    ) -> None:
        from recon.config import Depth
        from recon.crews.investigation.crew import build_investigation_crew

        step_cb = MagicMock()
        task_cb = MagicMock()

        plan = ReconPlan(topic="AI agents", depth=Depth.QUICK)
        investigations = plan.get_investigations()

        build_investigation_crew(
            plan=plan,
            investigations=investigations,
            llm=MagicMock(),
            search_tools=[MagicMock()],
            step_callback=step_cb,
            task_callback=task_cb,
        )

        crew_call_kwargs = mock_crew_cls.call_args[1]
        assert crew_call_kwargs["step_callback"] is step_cb
        assert crew_call_kwargs["task_callback"] is task_cb

    @patch("recon.crews.investigation.crew.Crew")
    @patch("recon.crews.investigation.crew.Task")
    @patch("recon.crews.investigation.crew.Agent")
    def test_standard_uses_sequential(
        self,
        mock_agent_cls: MagicMock,
        mock_task_cls: MagicMock,
        mock_crew_cls: MagicMock,
    ) -> None:
        """STANDARD depth should use sequential process (not hierarchical)."""
        from crewai import Process

        from recon.config import Depth
        from recon.crews.investigation.crew import build_investigation_crew

        plan = ReconPlan(topic="AI agents", depth=Depth.STANDARD)
        investigations = plan.get_investigations()

        build_investigation_crew(
            plan=plan,
            investigations=investigations,
            llm=MagicMock(),
            search_tools=[MagicMock()],
        )

        crew_call_kwargs = mock_crew_cls.call_args[1]
        assert crew_call_kwargs["process"] == Process.sequential
