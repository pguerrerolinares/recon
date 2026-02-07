"""Tests for recon.context.strategy module."""

from __future__ import annotations

from recon.context.strategy import (
    Strategy,
    choose_strategy,
    count_tokens_approx,
    get_context_window,
)


class TestCountTokensApprox:
    def test_empty_string(self) -> None:
        assert count_tokens_approx("") == 0

    def test_short_string(self) -> None:
        # 20 chars / 4 = 5 tokens
        assert count_tokens_approx("a" * 20) == 5

    def test_longer_string(self) -> None:
        assert count_tokens_approx("a" * 1000) == 250


class TestChooseStrategy:
    def test_small_input_is_direct(self) -> None:
        inputs = ["Short document."]
        assert choose_strategy(inputs, model_context_window=128_000) == Strategy.DIRECT

    def test_medium_input_is_summarize(self) -> None:
        # Create input that's between 0.8x and 3x the context window
        # 128K tokens * 4 chars/token * 0.9 = ~460K chars
        big_text = "x" * 460_000
        assert choose_strategy([big_text], model_context_window=128_000) == Strategy.SUMMARIZE

    def test_huge_input_is_map_reduce(self) -> None:
        # Create input that's > 3x the context window
        # 128K * 4 * 3.5 = ~1.8M chars
        huge_text = "x" * 1_800_000
        assert choose_strategy([huge_text], model_context_window=128_000) == Strategy.MAP_REDUCE

    def test_override_direct(self) -> None:
        huge = "x" * 1_800_000
        assert choose_strategy([huge], override="direct") == Strategy.DIRECT

    def test_override_summarize(self) -> None:
        assert choose_strategy(["small"], override="summarize") == Strategy.SUMMARIZE

    def test_override_map_reduce(self) -> None:
        assert choose_strategy(["small"], override="map_reduce") == Strategy.MAP_REDUCE

    def test_invalid_override_falls_to_auto(self) -> None:
        result = choose_strategy(["small"], override="invalid")
        assert result == Strategy.DIRECT  # small input -> direct


class TestGetContextWindow:
    def test_known_model(self) -> None:
        assert get_context_window("gpt-4o") == 128_000

    def test_model_with_provider_prefix(self) -> None:
        assert get_context_window("anthropic/claude-sonnet-4") == 200_000

    def test_model_with_version_suffix(self) -> None:
        assert get_context_window("claude-sonnet-4-20250514") == 200_000

    def test_unknown_model_defaults(self) -> None:
        assert get_context_window("some-unknown-model") == 128_000

    def test_gemini_large_context(self) -> None:
        assert get_context_window("gemini-2.5-flash") == 1_000_000

    def test_kimi_context(self) -> None:
        assert get_context_window("kimi-k2.5") == 256_000
