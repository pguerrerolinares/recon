"""Tests for LLM and search provider configuration.

Tests env var resolution, base URLs, API key errors, and factory functions.
All external SDK calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from recon.providers.llm import (
    PROVIDER_API_KEY_ENV,
    PROVIDER_BASE_URLS,
    create_llm,
    get_api_key,
    get_base_url,
)
from recon.providers.search import (
    SEARCH_API_KEY_ENV,
    create_search_tools,
    get_search_api_key,
)


class TestGetBaseUrl:
    def test_openrouter(self) -> None:
        assert get_base_url("openrouter") == "https://openrouter.ai/api/v1"

    def test_kimi(self) -> None:
        assert get_base_url("kimi") == "https://api.moonshot.ai/v1"

    def test_ollama(self) -> None:
        assert get_base_url("ollama") == "http://localhost:11434/v1"

    def test_custom_reads_env(self) -> None:
        with patch.dict("os.environ", {"RECON_BASE_URL": "https://custom.api/v1"}):
            assert get_base_url("custom") == "https://custom.api/v1"

    def test_custom_without_env(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert get_base_url("custom") is None

    def test_unknown_provider(self) -> None:
        assert get_base_url("nonexistent") is None

    def test_all_providers_have_urls(self) -> None:
        """Every provider in the URL map has a non-empty URL."""
        for provider, url in PROVIDER_BASE_URLS.items():
            assert url, f"Provider '{provider}' has empty base URL"
            assert url.startswith("http"), f"Provider '{provider}' URL invalid: {url}"


class TestGetApiKey:
    def test_openrouter_key(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "or-key-123"}):
            assert get_api_key("openrouter") == "or-key-123"

    def test_missing_key_raises(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="API key not found"),
        ):
            get_api_key("openrouter")

    def test_ollama_no_key_needed(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            # Ollama should not raise even without a key.
            result = get_api_key("ollama")
            assert result == ""

    def test_all_providers_have_env_var(self) -> None:
        """Every provider in the key map points to a non-empty env var name."""
        for provider, env_var in PROVIDER_API_KEY_ENV.items():
            assert env_var, f"Provider '{provider}' has empty env var name"

    def test_anthropic_key(self) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ant-key"}):
            assert get_api_key("anthropic") == "ant-key"

    def test_kimi_key(self) -> None:
        with patch.dict("os.environ", {"KIMI_API_KEY": "kimi-key"}):
            assert get_api_key("kimi") == "kimi-key"

    def test_unknown_provider_uses_recon_key(self) -> None:
        with patch.dict("os.environ", {"RECON_API_KEY": "recon-key"}):
            assert get_api_key("some_new_provider") == "recon-key"


class TestCreateLlm:
    def test_creates_llm_with_correct_params(self) -> None:
        plan = MagicMock()
        plan.provider = "openrouter"
        plan.model = "anthropic/claude-sonnet-4"
        plan.get_api_key.return_value = "test-key"

        with patch("crewai.LLM") as mock_llm_cls:
            create_llm(plan)

        mock_llm_cls.assert_called_once_with(
            model="anthropic/claude-sonnet-4",
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
        )


class TestGetSearchApiKey:
    def test_tavily_key(self) -> None:
        with patch.dict("os.environ", {"TAVILY_API_KEY": "tvly-123"}):
            assert get_search_api_key("tavily") == "tvly-123"

    def test_missing_search_key_raises(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="Search API key not found"),
        ):
            get_search_api_key("brave")

    def test_all_search_providers_have_env_var(self) -> None:
        for provider, env_var in SEARCH_API_KEY_ENV.items():
            assert env_var, f"Search provider '{provider}' has empty env var name"


class TestCreateSearchTools:
    def test_tavily_creates_tool(self) -> None:
        plan = MagicMock()
        plan.search.provider = "tavily"

        with (
            patch("crewai_tools.TavilySearchTool") as mock_tavily,
            patch("crewai_tools.ScrapeWebsiteTool") as mock_scrape,
        ):
            tools = create_search_tools(plan)

        assert len(tools) == 2
        mock_tavily.assert_called_once()
        mock_scrape.assert_called_once()

    def test_brave_creates_tool(self) -> None:
        plan = MagicMock()
        plan.search.provider = "brave"

        with (
            patch("crewai_tools.BraveSearchTool") as mock_brave,
            patch("crewai_tools.ScrapeWebsiteTool") as mock_scrape,
        ):
            tools = create_search_tools(plan)

        assert len(tools) == 2
        mock_brave.assert_called_once()
        mock_scrape.assert_called_once()

    def test_unknown_provider_only_scrape(self) -> None:
        plan = MagicMock()
        plan.search.provider = "unknown_provider"

        with patch("crewai_tools.ScrapeWebsiteTool") as mock_scrape:
            tools = create_search_tools(plan)

        assert len(tools) == 1
        mock_scrape.assert_called_once()
