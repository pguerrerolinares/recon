"""Tests for recon.config module."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import pytest
import yaml

from recon.config import (
    Depth,
    Investigation,
    ReconPlan,
    SearchConfig,
    VerificationConfig,
    create_plan_from_topic,
    load_plan,
)


class TestDepth:
    def test_depth_values(self) -> None:
        assert Depth.QUICK.value == "quick"
        assert Depth.STANDARD.value == "standard"
        assert Depth.DEEP.value == "deep"


class TestReconPlan:
    def test_minimal_plan(self) -> None:
        plan = ReconPlan(topic="AI agents")
        assert plan.topic == "AI agents"
        assert plan.depth == Depth.STANDARD
        assert plan.verify is True
        assert plan.provider == "openrouter"
        assert plan.model == "anthropic/claude-sonnet-4"
        assert plan.search.provider == "tavily"

    def test_plan_with_all_fields(self) -> None:
        plan = ReconPlan(
            topic="MCP ecosystem",
            questions=["What is MCP?"],
            focus="Developer tools",
            depth=Depth.DEEP,
            verify=True,
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            search=SearchConfig(provider="brave"),
            output_dir="./my-output",
            research_dir="./my-research",
            verification_dir="./my-verification",
            context_strategy="summarize",
        )
        assert plan.topic == "MCP ecosystem"
        assert plan.depth == Depth.DEEP
        assert plan.search.provider == "brave"
        assert plan.output_dir == "./my-output"

    def test_verify_false_syncs_verification(self) -> None:
        plan = ReconPlan(topic="test", verify=False)
        assert plan.verification.enabled is False

    def test_verify_true_keeps_verification(self) -> None:
        plan = ReconPlan(topic="test", verify=True)
        assert plan.verification.enabled is True

    def test_custom_verification_config(self) -> None:
        plan = ReconPlan(
            topic="test",
            verification=VerificationConfig(
                max_queries_per_claim=5,
                max_fetches_per_claim=3,
                timeout_per_fetch=30,
            ),
        )
        assert plan.verification.max_queries_per_claim == 5
        assert plan.verification.max_fetches_per_claim == 3
        assert plan.verification.timeout_per_fetch == 30


class TestGetInvestigations:
    def test_quick_generates_1_investigation(self) -> None:
        plan = ReconPlan(topic="test", depth=Depth.QUICK)
        investigations = plan.get_investigations()
        assert len(investigations) == 1
        assert investigations[0].id == "general"

    def test_standard_generates_3_investigations(self) -> None:
        plan = ReconPlan(topic="test", depth=Depth.STANDARD)
        investigations = plan.get_investigations()
        assert len(investigations) == 3

    def test_deep_generates_5_investigations(self) -> None:
        plan = ReconPlan(topic="test", depth=Depth.DEEP)
        investigations = plan.get_investigations()
        assert len(investigations) == 5

    def test_custom_investigations_override(self) -> None:
        custom = [
            Investigation(id="my-inv", name="My Investigation", questions=["Q1"]),
        ]
        plan = ReconPlan(topic="test", depth=Depth.DEEP, investigations=custom)
        investigations = plan.get_investigations()
        assert len(investigations) == 1
        assert investigations[0].id == "my-inv"

    def test_user_questions_distributed(self) -> None:
        plan = ReconPlan(
            topic="test",
            questions=["What is X?", "How does Y work?"],
            depth=Depth.STANDARD,
        )
        investigations = plan.get_investigations()
        # Each investigation should have the user questions
        for inv in investigations:
            assert "What is X?" in inv.questions
            assert "How does Y work?" in inv.questions


class TestGetApiKey:
    def test_openrouter_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
        plan = ReconPlan(topic="test", provider="openrouter")
        assert plan.get_api_key() == "sk-test-123"

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        plan = ReconPlan(topic="test", provider="openrouter")
        with pytest.raises(ValueError, match="API key not found"):
            plan.get_api_key()

    def test_ollama_no_key_needed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Ollama should not raise even without a key
        plan = ReconPlan(topic="test", provider="ollama")
        key = plan.get_api_key()
        assert key == ""


class TestGetSearchApiKey:
    def test_tavily_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-123")
        plan = ReconPlan(topic="test")
        assert plan.get_search_api_key() == "tvly-test-123"

    def test_missing_search_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        plan = ReconPlan(topic="test")
        with pytest.raises(ValueError, match="Search API key not found"):
            plan.get_search_api_key()


class TestLoadPlan:
    def test_load_valid_plan(self, tmp_plan_file: Path) -> None:
        plan = load_plan(tmp_plan_file)
        assert plan.topic == "Test topic"
        assert plan.depth == Depth.QUICK
        assert plan.verify is False

    def test_load_nonexistent_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_plan("/nonexistent/plan.yaml")

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("not: a: valid: yaml: [[[")
        with pytest.raises((yaml.YAMLError, ValueError)):
            load_plan(bad_file)

    def test_load_non_mapping(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "list.yaml"
        bad_file.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_plan(bad_file)


class TestCreatePlanFromTopic:
    def test_basic_creation(self) -> None:
        plan = create_plan_from_topic("AI agents")
        assert plan.topic == "AI agents"
        assert plan.depth == Depth.STANDARD

    def test_with_overrides(self) -> None:
        plan = create_plan_from_topic(
            "AI agents",
            depth="deep",
            verify=False,
            provider="anthropic",
            model="claude-sonnet-4-20250514",
        )
        assert plan.depth == Depth.DEEP
        assert plan.verify is False
        assert plan.provider == "anthropic"

    def test_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RECON_PROVIDER", "gemini")
        monkeypatch.setenv("RECON_MODEL", "gemini-2.5-flash")
        plan = create_plan_from_topic("test")
        assert plan.provider == "gemini"
        assert plan.model == "gemini-2.5-flash"
