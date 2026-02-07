"""YAML plan parser and Pydantic validation models.

This module defines the schema for Recon plan files and handles
parsing, validation, and default resolution.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class Depth(StrEnum):
    """Research depth controls the number of parallel investigation agents."""

    QUICK = "quick"  # 1 researcher
    STANDARD = "standard"  # 3 researchers
    DEEP = "deep"  # 5 researchers


DEPTH_AGENT_COUNT = {
    Depth.QUICK: 1,
    Depth.STANDARD: 3,
    Depth.DEEP: 5,
}

# Default investigation angles generated when user doesn't provide custom ones.
# Maps depth -> list of (id, name, description) tuples.
DEFAULT_ANGLES = {
    Depth.QUICK: [
        ("general", "General Investigation", "Comprehensive overview of the topic."),
    ],
    Depth.STANDARD: [
        ("landscape", "Market Landscape", "Market size, key players, and ecosystem overview."),
        ("competitors", "Competitor Analysis", "Existing solutions, strengths, weaknesses, gaps."),
        (
            "trends",
            "Trends and Opportunities",
            "Emerging trends, unsolved problems, community needs.",
        ),
    ],
    Depth.DEEP: [
        ("landscape", "Market Landscape", "Market size, key players, and ecosystem overview."),
        ("competitors", "Competitor Analysis", "Existing solutions, strengths, weaknesses, gaps."),
        (
            "trends",
            "Trends and Opportunities",
            "Emerging trends, unsolved problems, community needs.",
        ),
        (
            "business",
            "Business Models",
            "Monetization patterns, revenue data, paths to profitability.",
        ),
        (
            "community",
            "Community Sentiment",
            "Developer opinions, pain points, feature requests from Reddit/HN/GitHub.",
        ),
    ],
}


class SearchConfig(BaseModel):
    """Search provider configuration."""

    provider: str = "tavily"  # tavily | brave | serper | exa


class Investigation(BaseModel):
    """A single investigation angle with its own set of questions."""

    id: str
    name: str
    questions: list[str]
    instructions: str | None = None


class VerificationConfig(BaseModel):
    """Fact-checking phase configuration."""

    enabled: bool = True
    max_queries_per_claim: int = Field(default=2, ge=1, le=10)
    max_fetches_per_claim: int = Field(default=2, ge=1, le=10)
    timeout_per_fetch: int = Field(default=10, ge=1, le=60)


class SynthesisConfig(BaseModel):
    """Synthesis phase configuration."""

    instructions: str | None = None


class ReconPlan(BaseModel):
    """Root configuration model for a Recon research plan.

    Supports two modes:
    - Simple: just topic + questions + depth (auto-generates investigators)
    - Extended: custom investigations with per-agent questions and instructions
    """

    # Core fields (required)
    topic: str

    # Optional fields with smart defaults
    questions: list[str] | None = None
    focus: str | None = None
    depth: Depth = Depth.STANDARD
    verify: bool = True

    # Provider config
    provider: str = "openrouter"
    model: str = "anthropic/claude-sonnet-4"
    search: SearchConfig = SearchConfig()
    context_strategy: str = "auto"  # auto | direct | summarize | map_reduce

    # Output directories
    output_dir: str = "./output"
    research_dir: str = "./research"
    verification_dir: str = "./verification"

    # Extended mode: custom investigations
    investigations: list[Investigation] | None = None

    # Phase configs
    verification: VerificationConfig = VerificationConfig()
    synthesis: SynthesisConfig = SynthesisConfig()

    @model_validator(mode="after")
    def sync_verify_flag(self) -> ReconPlan:
        """Keep verify flag and verification.enabled in sync."""
        if not self.verify:
            self.verification.enabled = False
        return self

    def get_investigations(self) -> list[Investigation]:
        """Return investigations, auto-generating from depth if not provided."""
        if self.investigations:
            return self.investigations

        angles = DEFAULT_ANGLES[self.depth]
        result = []

        # Distribute user questions across auto-generated angles
        user_questions = self.questions or []

        for angle_id, angle_name, angle_desc in angles:
            # Each angle gets the full set of user questions plus its own context
            investigation_questions = list(user_questions) if user_questions else []
            if not investigation_questions:
                investigation_questions = [f"Investigate: {angle_desc}"]

            result.append(
                Investigation(
                    id=angle_id,
                    name=angle_name,
                    questions=investigation_questions,
                    instructions=angle_desc,
                )
            )

        return result

    def get_api_key(self) -> str:
        """Resolve API key from environment variables based on provider."""
        env_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "kimi": "KIMI_API_KEY",
            "copilot": "GITHUB_TOKEN",
            "custom": "RECON_API_KEY",
        }
        env_var = env_map.get(self.provider, "RECON_API_KEY")
        key = os.environ.get(env_var, "")
        if not key and self.provider != "ollama":
            msg = (
                f"API key not found. Set {env_var} environment variable "
                f"for provider '{self.provider}'."
            )
            raise ValueError(msg)
        return key

    def get_search_api_key(self) -> str:
        """Resolve search API key from environment variables."""
        env_map = {
            "tavily": "TAVILY_API_KEY",
            "brave": "BRAVE_API_KEY",
            "serper": "SERPER_API_KEY",
            "exa": "EXA_API_KEY",
        }
        env_var = env_map.get(self.search.provider, "TAVILY_API_KEY")
        key = os.environ.get(env_var, "")
        if not key:
            msg = (
                f"Search API key not found. Set {env_var} environment variable "
                f"for search provider '{self.search.provider}'."
            )
            raise ValueError(msg)
        return key


def load_plan(path: str | Path) -> ReconPlan:
    """Load and validate a Recon plan from a YAML file.

    Args:
        path: Path to the YAML plan file.

    Returns:
        Validated ReconPlan instance.

    Raises:
        FileNotFoundError: If the plan file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
        pydantic.ValidationError: If the plan doesn't match the schema.
    """
    plan_path = Path(path)
    if not plan_path.exists():
        msg = f"Plan file not found: {plan_path}"
        raise FileNotFoundError(msg)

    with open(plan_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        msg = f"Plan file must be a YAML mapping, got {type(raw).__name__}"
        raise ValueError(msg)

    return ReconPlan(**raw)


def create_plan_from_topic(
    topic: str,
    depth: str = "standard",
    verify: bool = True,
    provider: str | None = None,
    model: str | None = None,
) -> ReconPlan:
    """Create a ReconPlan from inline CLI arguments.

    Args:
        topic: Research topic.
        depth: Research depth (quick/standard/deep).
        verify: Whether to run verification phase.
        provider: LLM provider override.
        model: Model override.

    Returns:
        Configured ReconPlan instance.
    """
    kwargs: dict = {
        "topic": topic,
        "depth": depth,
        "verify": verify,
    }
    if provider:
        kwargs["provider"] = provider
    if model:
        kwargs["model"] = model

    # Read defaults from env if set
    if not provider and os.environ.get("RECON_PROVIDER"):
        kwargs["provider"] = os.environ["RECON_PROVIDER"]
    if not model and os.environ.get("RECON_MODEL"):
        kwargs["model"] = os.environ["RECON_MODEL"]

    return ReconPlan(**kwargs)
