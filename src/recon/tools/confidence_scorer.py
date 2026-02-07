"""ConfidenceScorerTool - Score claim confidence based on verification results.

Returns a confidence score between 0.0 and 1.0 based on verification
status and source count. This is a deterministic scoring function,
not an LLM call.
"""

from __future__ import annotations

import json

from crewai.tools import BaseTool

# Score ranges per verification status.
SCORE_RANGES: dict[str, tuple[float, float]] = {
    "VERIFIED": (0.7, 1.0),
    "PARTIALLY_VERIFIED": (0.4, 0.7),
    "UNVERIFIABLE": (0.2, 0.4),
    "CONTRADICTED": (0.0, 0.1),
    "ERROR": (0.1, 0.3),
}


def score_confidence(
    verification_status: str,
    source_count: int = 1,
    has_primary_source: bool = False,
) -> float:
    """Score the confidence of a claim based on verification results.

    Args:
        verification_status: One of VERIFIED, PARTIALLY_VERIFIED,
            UNVERIFIABLE, CONTRADICTED, ERROR.
        source_count: Number of independent sources confirming the claim.
        has_primary_source: Whether verification used a primary source
            (official website, GitHub repo, etc.).

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    status = verification_status.upper()
    low, high = SCORE_RANGES.get(status, (0.2, 0.4))

    # Base score is the midpoint of the range
    score = (low + high) / 2

    # Bonus for multiple sources (up to +0.15)
    if source_count > 1:
        source_bonus = min(0.15, (source_count - 1) * 0.05)
        score += source_bonus

    # Bonus for primary source (+0.05)
    if has_primary_source:
        score += 0.05

    # Clamp to valid range
    return max(0.0, min(1.0, round(score, 2)))


def score_label(score: float) -> str:
    """Return a human-readable label for a confidence score.

    Args:
        score: Confidence score between 0.0 and 1.0.

    Returns:
        One of: HIGH, MEDIUM, LOW, VERY_LOW.
    """
    if score >= 0.7:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    elif score >= 0.2:
        return "LOW"
    else:
        return "VERY_LOW"


class ConfidenceScorerTool(BaseTool):
    """Score the confidence level of a claim based on verification results.

    Returns a confidence score (0.0-1.0) with a human-readable label.
    """

    name: str = "confidence_scorer"
    description: str = (
        "Score the confidence level of a claim based on verification results. "
        "Input: JSON with 'verification_status' (VERIFIED/PARTIALLY_VERIFIED/"
        "UNVERIFIABLE/CONTRADICTED), 'source_count' (int), and optionally "
        "'has_primary_source' (bool). "
        "Returns a confidence score (0.0-1.0) with label."
    )

    def _run(self, input_data: str) -> str:
        """Score a claim's confidence.

        Args:
            input_data: JSON string with verification_status, source_count,
                       and optionally has_primary_source.

        Returns:
            JSON string with score and label.
        """
        try:
            data = json.loads(input_data)
            status = data.get("verification_status", "UNVERIFIABLE")
            count = int(data.get("source_count", 1))
            primary = bool(data.get("has_primary_source", False))
        except (json.JSONDecodeError, TypeError, ValueError):
            return json.dumps(
                {
                    "score": 0.3,
                    "label": "LOW",
                    "error": "Could not parse input. Using default LOW confidence.",
                }
            )

        score = score_confidence(status, count, primary)
        label = score_label(score)

        return json.dumps(
            {
                "score": score,
                "label": label,
                "verification_status": status,
                "source_count": count,
                "has_primary_source": primary,
            }
        )
