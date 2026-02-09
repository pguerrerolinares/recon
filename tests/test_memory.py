"""Tests for KnowledgeConfig and investigation crew features.

The KnowledgeConfig tests verify the new schema and backward-compat
migration from the legacy ``memory:`` YAML key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
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
            knowledge=KnowledgeConfig(enabled=True, db_path="./my-knowledge.db", embedder="ollama"),
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
# flow_builder knowledge integration tests
# ---------------------------------------------------------------------------


class TestFlowBuilderKnowledge:
    """Test that flow_builder correctly integrates with the knowledge DB."""

    @patch("recon.flow_builder._query_prior_knowledge")
    @patch("recon.flow_builder.create_search_tools")
    @patch("recon.flow_builder.create_llm")
    @patch("recon.flow_builder.build_synthesis_crew")
    @patch("recon.flow_builder.build_verification_crew")
    @patch("recon.flow_builder.build_investigation_crew")
    def test_knowledge_query_called_when_enabled(
        self,
        mock_inv_crew: MagicMock,
        mock_ver_crew: MagicMock,
        mock_syn_crew: MagicMock,
        mock_llm: MagicMock,
        mock_search: MagicMock,
        mock_query: MagicMock,
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

        # prior_knowledge should be passed to investigation crew
        inv_call_kwargs = mock_inv_crew.call_args[1]
        assert inv_call_kwargs["prior_knowledge"] == "Prior: CrewAI has 44K stars"

    @patch("recon.flow_builder._query_prior_knowledge")
    @patch("recon.flow_builder.create_search_tools")
    @patch("recon.flow_builder.create_llm")
    @patch("recon.flow_builder.build_synthesis_crew")
    @patch("recon.flow_builder.build_investigation_crew")
    def test_knowledge_returns_none_when_disabled(
        self,
        mock_inv_crew: MagicMock,
        mock_syn_crew: MagicMock,
        mock_llm: MagicMock,
        mock_search: MagicMock,
        mock_query: MagicMock,
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
