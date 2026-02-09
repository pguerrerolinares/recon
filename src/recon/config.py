"""YAML plan parser and Pydantic validation models.

This module defines the schema for Recon plan files and handles
parsing, validation, and default resolution.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Any

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

# Maximum CrewAI agent iterations per depth level.
# Higher values give agents more tool-call rounds for thorough research.
DEPTH_MAX_ITER: dict[Depth, int] = {
    Depth.QUICK: 10,
    Depth.STANDARD: 25,
    Depth.DEEP: 40,
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

    def output_path(self, base_dir: str) -> str:
        """Return the canonical output file path for this investigation.

        Args:
            base_dir: Research output directory.

        Returns:
            Path string like ``base_dir/landscape-market-landscape.md``.
        """
        slug = self.name.lower().replace(" ", "-")
        return str(Path(base_dir) / f"{self.id}-{slug}.md")


class VerificationConfig(BaseModel):
    """Fact-checking phase configuration."""

    enabled: bool = True
    max_queries_per_claim: int = Field(default=2, ge=1, le=10)
    max_fetches_per_claim: int = Field(default=2, ge=1, le=10)
    timeout_per_fetch: int = Field(default=10, ge=1, le=60)
    min_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to consider a claim acceptable.",
    )
    require_primary_source: bool = Field(
        default=False,
        description="When True, flag claims that lack a primary source citation.",
    )
    max_claims: int = Field(
        default=40,
        ge=5,
        le=200,
        description=("Maximum claims to verify. Prioritized by type and source availability."),
    )
    phase_timeout: int = Field(
        default=600,
        ge=60,
        le=3600,
        description=(
            "Verification phase timeout in seconds. On timeout, pipeline proceeds to synthesis."
        ),
    )


class KnowledgeConfig(BaseModel):
    """Knowledge persistence and cross-run memory configuration.

    Replaces the legacy ``MemoryConfig``.  The knowledge layer uses a
    SQLite database for structured claim/source storage and CrewAI's
    built-in memory system (ChromaDB + ONNX embeddings) for semantic
    retrieval during agent execution.
    """

    enabled: bool = Field(
        default=True,
        description="Enable knowledge persistence across runs.",
    )
    db_path: str = Field(
        default="./knowledge.db",
        description="Path to the SQLite knowledge database.",
    )
    embedder: str = Field(
        default="onnx",
        description="Embedding provider for CrewAI memory: onnx | ollama | openai.",
    )
    stale_after_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Claims older than this are flagged for re-verification.",
    )

    # --- Backward compatibility for YAML plans using the old schema ---
    # These fields are accepted but ignored (silently consumed so that
    # loading an old plan.yaml doesn't raise a validation error).
    path: str | None = Field(default=None, exclude=True)
    embedding_provider: str | None = Field(default=None, exclude=True)


# Keep the old name importable for tests/code that reference it.
MemoryConfig = KnowledgeConfig


# Provider pricing: (input USD / 1M tokens, output USD / 1M tokens).
# Used for cost estimation in token_usage tracking.
PROVIDER_PRICING: dict[str, tuple[float, float]] = {
    # Kimi
    "kimi-k2.5": (0.14, 0.28),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o1": (15.00, 60.00),
    "o1-mini": (1.10, 4.40),
    # Anthropic
    "claude-sonnet-4": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    # Google
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.00),
    # Meta via Groq / OpenRouter
    "llama-4-maverick": (0.05, 0.25),
    "llama-3.1-70b": (0.10, 0.20),
}


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float | None:
    """Estimate USD cost based on token counts and known pricing.

    Returns *None* when the model is not in :data:`PROVIDER_PRICING`
    (e.g. local Ollama models).
    """
    # Strip provider prefix (e.g. "anthropic/claude-sonnet-4" → "claude-sonnet-4")
    name = model.split("/")[-1] if "/" in model else model

    # Exact match first, then prefix match
    pricing = PROVIDER_PRICING.get(name)
    if pricing is None:
        for known, p in PROVIDER_PRICING.items():
            if name.startswith(known):
                pricing = p
                break

    if pricing is None:
        return None

    input_price, output_price = pricing
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000


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

    # Research options
    auto_questions: bool = True  # Generate sub-questions per angle via LLM before investigation

    # Provider config
    provider: str = "openrouter"
    model: str = "anthropic/claude-sonnet-4"
    search: SearchConfig = SearchConfig()
    context_strategy: str = "auto"  # auto | direct | summarize | map_reduce

    # Output directories
    output_dir: str = "./output"
    research_dir: str = "./research"
    verification_dir: str = "./verification"
    memory_dir: str = "./memory"

    # Extended mode: custom investigations
    investigations: list[Investigation] | None = None

    # Phase configs
    verification: VerificationConfig = VerificationConfig()
    synthesis: SynthesisConfig = SynthesisConfig()
    knowledge: KnowledgeConfig = KnowledgeConfig()

    @model_validator(mode="before")
    @classmethod
    def migrate_memory_to_knowledge(cls, data: Any) -> Any:
        """Accept legacy ``memory:`` YAML key and promote it to ``knowledge:``.

        This allows old plan files with ``memory:`` sections and Python code
        using ``memory=MemoryConfig(...)`` to keep working transparently.
        """
        if isinstance(data, dict) and "memory" in data:
            mem = data.pop("memory")
            if "knowledge" not in data:
                data["knowledge"] = mem
        return data

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
        """Resolve API key from environment variables based on provider.

        Uses the canonical mapping from ``providers.llm.PROVIDER_API_KEY_ENV``.
        """
        from recon.providers.llm import PROVIDER_API_KEY_ENV

        env_var = PROVIDER_API_KEY_ENV.get(self.provider, "RECON_API_KEY")
        key = os.environ.get(env_var, "")
        if not key and self.provider != "ollama":
            msg = (
                f"API key not found. Set {env_var} environment variable "
                f"for provider '{self.provider}'."
            )
            raise ValueError(msg)
        return key

    def get_search_api_key(self) -> str:
        """Resolve search API key from environment variables.

        Uses the canonical mapping from ``providers.search.SEARCH_API_KEY_ENV``.
        """
        from recon.providers.search import SEARCH_API_KEY_ENV

        env_var = SEARCH_API_KEY_ENV.get(self.search.provider, "TAVILY_API_KEY")
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
