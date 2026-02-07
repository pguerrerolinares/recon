"""Shared fixtures for Recon tests.

All tests mock external dependencies (LLM, search, web) to avoid
real API calls and costs.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import pytest

from recon.config import (
    Depth,
    Investigation,
    ReconPlan,
    SearchConfig,
    SynthesisConfig,
    VerificationConfig,
)


@pytest.fixture
def simple_plan() -> ReconPlan:
    """A minimal plan for testing."""
    return ReconPlan(
        topic="Test topic: AI agent frameworks",
        depth=Depth.QUICK,
        verify=False,
    )


@pytest.fixture
def standard_plan() -> ReconPlan:
    """A standard-depth plan with questions."""
    return ReconPlan(
        topic="Test topic: AI agent frameworks",
        questions=[
            "What frameworks exist?",
            "What is their adoption?",
        ],
        focus="Open-source projects",
        depth=Depth.STANDARD,
        verify=True,
    )


@pytest.fixture
def deep_plan() -> ReconPlan:
    """A deep plan with custom investigations."""
    return ReconPlan(
        topic="Test topic: MCP ecosystem",
        focus="Developer tools",
        depth=Depth.DEEP,
        verify=True,
        provider="openrouter",
        model="anthropic/claude-sonnet-4",
        search=SearchConfig(provider="tavily"),
        investigations=[
            Investigation(
                id="protocol",
                name="Protocol Analysis",
                questions=["What is MCP?", "How does it work?"],
                instructions="Reference the official spec.",
            ),
            Investigation(
                id="servers",
                name="Server Ecosystem",
                questions=["What servers exist?"],
            ),
        ],
        verification=VerificationConfig(
            enabled=True,
            max_queries_per_claim=3,
        ),
        synthesis=SynthesisConfig(
            instructions="Rank opportunities by feasibility.",
        ),
    )


@pytest.fixture
def tmp_plan_file(tmp_path: Path) -> Path:
    """Create a temporary plan YAML file."""
    plan_content = """\
topic: "Test topic"
questions:
  - "What is the current state?"
depth: quick
verify: false
"""
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(plan_content)
    return plan_file


@pytest.fixture
def tmp_research_dir(tmp_path: Path) -> Path:
    """Create a temporary research directory with sample files."""
    research_dir = tmp_path / "research"
    research_dir.mkdir()

    (research_dir / "general-overview.md").write_text(
        "# Overview\n\nThe market has 10 players. [Source: example.com]\n"
        "Revenue reached $1B in 2025. [UNVERIFIED]\n"
    )
    (research_dir / "competitors-analysis.md").write_text(
        "# Competitors\n\n| Name | Stars |\n|------|-------|\n"
        "| CrewAI | 44K |\n| LangGraph | 8K |\n"
    )

    return research_dir
