"""Shared helpers for Recon verification tools."""

from __future__ import annotations

import json


def parse_tool_input(raw: str) -> tuple[dict | None, str | None]:
    """Parse a JSON string into a dict, returning an error response on failure.

    Every Recon tool that accepts JSON input uses this pattern.

    Args:
        raw: Raw JSON string from the agent.

    Returns:
        ``(data, None)`` on success, or ``(None, error_json)`` on failure
        where *error_json* is a ready-to-return JSON string.
    """
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None, json.dumps({"error": "Expected a JSON object, got " + type(data).__name__})
        return data, None
    except (json.JSONDecodeError, TypeError) as exc:
        return None, json.dumps({"error": f"Invalid JSON input: {exc}"})
