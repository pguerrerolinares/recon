"""LLM provider configuration and factory.

Creates CrewAI LLM instances from provider config. All supported providers
use the OpenAI-compatible API interface.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from recon.config import ReconPlan

# Base URLs for all OpenAI-compatible providers.
PROVIDER_BASE_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
    "kimi": "https://api.moonshot.ai/v1",
    "ollama": "http://localhost:11434/v1",
}

# Map from provider name to environment variable for the API key.
PROVIDER_API_KEY_ENV: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "kimi": "KIMI_API_KEY",
    "copilot": "GITHUB_TOKEN",
    "custom": "RECON_API_KEY",
}


def get_base_url(provider: str) -> str | None:
    """Get the base URL for a provider.

    Args:
        provider: Provider name (e.g. 'openrouter', 'anthropic').

    Returns:
        Base URL string, or None for providers using default SDK URL.
    """
    if provider == "custom":
        return os.environ.get("RECON_BASE_URL")
    return PROVIDER_BASE_URLS.get(provider)


def get_api_key(provider: str) -> str:
    """Resolve the API key for a provider from environment variables.

    Args:
        provider: Provider name.

    Returns:
        API key string.

    Raises:
        ValueError: If the required API key is not set.
    """
    env_var = PROVIDER_API_KEY_ENV.get(provider, "RECON_API_KEY")
    key = os.environ.get(env_var, "")
    if not key and provider != "ollama":
        msg = f"API key not found. Set {env_var} environment variable for provider '{provider}'."
        raise ValueError(msg)
    return key


def create_llm(plan: ReconPlan) -> Any:
    """Create a CrewAI LLM instance from the plan's provider config.

    All providers use the OpenAI-compatible interface via CrewAI's LLM class.

    Args:
        plan: Validated ReconPlan with provider and model info.

    Returns:
        A CrewAI LLM instance configured for the specified provider.
    """
    from crewai import LLM

    api_key = plan.get_api_key()
    base_url = get_base_url(plan.provider)

    return LLM(
        model=plan.model,
        base_url=base_url,
        api_key=api_key,
    )
