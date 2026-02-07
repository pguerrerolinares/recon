"""Search provider configuration and tool factory.

Creates CrewAI search tool instances based on the plan's search config.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from recon.config import ReconPlan

# Map from search provider name to environment variable for the API key.
SEARCH_API_KEY_ENV: dict[str, str] = {
    "tavily": "TAVILY_API_KEY",
    "brave": "BRAVE_API_KEY",
    "serper": "SERPER_API_KEY",
    "exa": "EXA_API_KEY",
}


def get_search_api_key(provider: str) -> str:
    """Resolve the search API key for a provider from environment variables.

    Args:
        provider: Search provider name.

    Returns:
        API key string.

    Raises:
        ValueError: If the required API key is not set.
    """
    env_var = SEARCH_API_KEY_ENV.get(provider, "TAVILY_API_KEY")
    key = os.environ.get(env_var, "")
    if not key:
        msg = (
            f"Search API key not found. Set {env_var} environment variable "
            f"for search provider '{provider}'."
        )
        raise ValueError(msg)
    return key


def create_search_tools(plan: ReconPlan) -> list[Any]:
    """Create search tools based on the plan's search provider config.

    Returns a list of CrewAI tools ready to be assigned to agents.

    Args:
        plan: Validated ReconPlan with search provider info.

    Returns:
        List of CrewAI tool instances.
    """
    tools: list[Any] = []
    provider = plan.search.provider

    if provider == "tavily":
        from crewai_tools import TavilySearchTool

        tools.append(TavilySearchTool())
    elif provider == "brave":
        from crewai_tools import BraveSearchTool

        tools.append(BraveSearchTool())
    elif provider == "serper":
        from crewai_tools import SerperDevTool

        tools.append(SerperDevTool())
    elif provider == "exa":
        from crewai_tools import EXASearchTool

        tools.append(EXASearchTool())

    # Always add web scraping for content extraction
    from crewai_tools import ScrapeWebsiteTool

    tools.append(ScrapeWebsiteTool())

    return tools
