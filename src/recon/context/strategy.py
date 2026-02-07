"""Context management strategies for handling large inputs.

When the synthesizer or verifier receives multiple research documents,
the total token count may exceed the model's context window. This module
decides which strategy to use and executes it.
"""

from __future__ import annotations

from enum import StrEnum


class Strategy(StrEnum):
    """Context management strategy."""

    DIRECT = "direct"  # Pass everything as-is
    SUMMARIZE = "summarize"  # Summarize each doc first
    MAP_REDUCE = "map_reduce"  # Process each doc separately, then combine


# Approximate tokens per character for different model families.
# OpenAI/Anthropic models average ~4 chars per token.
CHARS_PER_TOKEN = 4

# Overhead for system prompt + output buffer (in tokens).
PROMPT_OVERHEAD = 4000


def count_tokens_approx(text: str) -> int:
    """Approximate token count using character-based heuristic.

    For more accurate counting, use tiktoken with the specific model's
    encoding. This fallback works for any model.

    Args:
        text: Input text.

    Returns:
        Approximate token count.
    """
    return len(text) // CHARS_PER_TOKEN


def count_tokens_tiktoken(text: str, model: str = "gpt-4o") -> int:
    """Count tokens using tiktoken for OpenAI-family models.

    Falls back to character-based approximation if tiktoken fails.

    Args:
        text: Input text.
        model: Model name for encoding selection.

    Returns:
        Token count.
    """
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        return count_tokens_approx(text)


def choose_strategy(
    inputs: list[str],
    model_context_window: int = 128_000,
    override: str = "auto",
) -> Strategy:
    """Choose the best context management strategy based on input size.

    Args:
        inputs: List of input documents (as strings).
        model_context_window: Model's context window in tokens.
        override: User override ('auto', 'direct', 'summarize', 'map_reduce').

    Returns:
        The chosen Strategy.
    """
    # User override
    if override != "auto":
        try:
            return Strategy(override)
        except ValueError:
            pass  # Fall through to auto

    total_text = "".join(inputs)
    total_tokens = count_tokens_approx(total_text) + PROMPT_OVERHEAD
    window = model_context_window

    if total_tokens < window * 0.8:
        return Strategy.DIRECT
    elif total_tokens < window * 3:
        return Strategy.SUMMARIZE
    else:
        return Strategy.MAP_REDUCE


# Default context windows for known model families (in tokens).
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "o1": 200_000,
    "o1-mini": 128_000,
    # Anthropic
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "claude-3.5-sonnet": 200_000,
    "claude-3-opus": 200_000,
    # Google
    "gemini-2.5-flash": 1_000_000,
    "gemini-2.5-pro": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
    # Meta (via Groq/OpenRouter)
    "llama-4-maverick": 128_000,
    "llama-3.1-70b": 128_000,
    # Kimi
    "kimi-k2.5": 256_000,
}


def get_context_window(model: str) -> int:
    """Get the context window size for a model.

    Checks known model families first, then defaults to 128K.

    Args:
        model: Model name or path (e.g. 'anthropic/claude-sonnet-4').

    Returns:
        Context window size in tokens.
    """
    # Strip provider prefix if present (e.g. 'anthropic/claude-sonnet-4' -> 'claude-sonnet-4')
    model_name = model.split("/")[-1] if "/" in model else model

    # Check exact match first
    if model_name in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model_name]

    # Check partial match (e.g. 'claude-sonnet-4-20250514' matches 'claude-sonnet-4')
    for known_model, window in MODEL_CONTEXT_WINDOWS.items():
        if model_name.startswith(known_model):
            return window

    # Default fallback
    return 128_000
