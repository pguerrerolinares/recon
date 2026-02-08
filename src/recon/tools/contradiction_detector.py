"""ContradictionDetectorTool - Detect contradictions between claims.

Compares two claims about the same topic from different sources and
determines if they contradict each other. Uses heuristic comparison
of numbers, dates, and key assertions without requiring an LLM.
"""

from __future__ import annotations

import json
import re

from crewai.tools import BaseTool


def _extract_numbers(text: str) -> list[float]:
    """Extract all numbers from text, normalizing K/M/B suffixes."""
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "t": 1_000_000_000_000}

    numbers: list[float] = []
    # Match numbers with optional K/M/B suffix
    for match in re.finditer(r"(\d[\d,.]*)\s*([KMBTkmbt])?", text):
        num_str = match.group(1).replace(",", "")
        suffix = (match.group(2) or "").lower()

        try:
            value = float(num_str)
            if suffix in multipliers:
                value *= multipliers[suffix]
            numbers.append(value)
        except ValueError:
            continue

    return numbers


def _extract_years(text: str) -> list[int]:
    """Extract 4-digit years from text."""
    return [int(m) for m in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]


def _numbers_contradict(nums_a: list[float], nums_b: list[float], tolerance: float = 0.15) -> bool:
    """Check if two sets of numbers contradict each other.

    Two numbers contradict if they differ by more than the tolerance
    (as a fraction of the larger number) and are in the same order
    of magnitude (likely referring to the same metric).
    """
    for a in nums_a:
        for b in nums_b:
            if a == 0 and b == 0:
                continue
            max_val = max(abs(a), abs(b))
            if max_val == 0:
                continue

            # Only compare numbers in the same order of magnitude
            if abs(a) > 0 and abs(b) > 0:
                ratio = max(abs(a), abs(b)) / min(abs(a), abs(b))
                if ratio > 100:
                    continue  # Different metrics, skip

            diff = abs(a - b) / max_val
            if diff > tolerance:
                return True

    return False


def _years_contradict(years_a: list[int], years_b: list[int]) -> bool:
    """Check if year references contradict each other."""
    for a in years_a:
        for b in years_b:
            if abs(a - b) >= 2:
                return True
    return False


def detect_contradiction(
    claim_a: str,
    source_a: str,
    claim_b: str,
    source_b: str,
) -> dict:
    """Compare two claims and determine if they contradict.

    Args:
        claim_a: First claim text.
        source_a: Source identifier for first claim.
        claim_b: Second claim text.
        source_b: Source identifier for second claim.

    Returns:
        Dict with keys: status, explanation.
        status is one of: CONSISTENT, CONTRADICTED, AMBIGUOUS.
    """
    nums_a = _extract_numbers(claim_a)
    nums_b = _extract_numbers(claim_b)
    years_a = _extract_years(claim_a)
    years_b = _extract_years(claim_b)

    contradictions: list[str] = []

    # Check number contradictions
    if nums_a and nums_b and _numbers_contradict(nums_a, nums_b):
        contradictions.append(
            f"Numeric mismatch: {source_a} says {nums_a} vs {source_b} says {nums_b}"
        )

    # Check year contradictions
    if years_a and years_b and _years_contradict(years_a, years_b):
        contradictions.append(
            f"Date mismatch: {source_a} says {years_a} vs {source_b} says {years_b}"
        )

    # Check for negation patterns
    negation_a = bool(
        re.search(r"\b(?:not|no|never|none|neither|nor|without|lack)\b", claim_a, re.IGNORECASE)
    )
    negation_b = bool(
        re.search(r"\b(?:not|no|never|none|neither|nor|without|lack)\b", claim_b, re.IGNORECASE)
    )

    if negation_a != negation_b:
        # One claim negates, the other doesn't -- could be contradiction
        # but we need shared key terms to be sure
        shared_terms = set(claim_a.lower().split()) & set(claim_b.lower().split())
        # Filter out common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "has",
            "have",
            "had",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "and",
            "or",
            "but",
            "not",
            "no",
            "it",
            "its",
            "this",
            "that",
        }
        meaningful_shared = shared_terms - stop_words
        if len(meaningful_shared) >= 2:
            contradictions.append(
                f"Negation mismatch: one claim negates while the other affirms. "
                f"Shared terms: {meaningful_shared}"
            )

    if contradictions:
        return {
            "status": "CONTRADICTED",
            "explanation": "; ".join(contradictions),
            "claim_a": claim_a,
            "source_a": source_a,
            "claim_b": claim_b,
            "source_b": source_b,
        }

    # If no contradictions found but both have numbers/dates, they might be consistent
    if (nums_a and nums_b) or (years_a and years_b):
        return {
            "status": "CONSISTENT",
            "explanation": "No contradictions detected between the claims.",
            "claim_a": claim_a,
            "source_a": source_a,
            "claim_b": claim_b,
            "source_b": source_b,
        }

    return {
        "status": "AMBIGUOUS",
        "explanation": "Insufficient quantitative data to determine consistency.",
        "claim_a": claim_a,
        "source_a": source_a,
        "claim_b": claim_b,
        "source_b": source_b,
    }


class ContradictionDetectorTool(BaseTool):
    """Compare two claims about the same topic and detect contradictions.

    Uses heuristic comparison of numbers, dates, and assertions.
    """

    name: str = "contradiction_detector"
    description: str = (
        "Compare two claims about the same topic from different sources "
        "and determine if they contradict each other. "
        "Input: JSON with 'claim_a', 'source_a', 'claim_b', 'source_b'. "
        "Returns CONSISTENT/CONTRADICTED/AMBIGUOUS with explanation."
    )

    def _run(self, input_data: str) -> str:
        """Detect contradictions between two claims.

        Args:
            input_data: JSON string with claim_a, source_a, claim_b, source_b.

        Returns:
            JSON string with contradiction detection result.
        """
        from recon.tools._helpers import parse_tool_input

        data, err = parse_tool_input(input_data)
        if err:
            return json.dumps(
                {
                    "status": "AMBIGUOUS",
                    "explanation": "Invalid input. Expected JSON with claim pairs.",
                }
            )

        assert data is not None  # guaranteed by parse_tool_input when err is None

        result = detect_contradiction(
            claim_a=data.get("claim_a", ""),
            source_a=data.get("source_a", "unknown"),
            claim_b=data.get("claim_b", ""),
            source_b=data.get("source_b", "unknown"),
        )

        return json.dumps(result, indent=2)
