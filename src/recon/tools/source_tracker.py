"""SourceTrackerTool - Track provenance of data points in an audit trail.

Maintains a JSON Lines file linking every verified claim to its origin URL,
access date, and verification status. This creates an auditable chain of
evidence for the final report.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import Field

# Default audit trail file name.
DEFAULT_AUDIT_FILE = "audit-trail.jsonl"


def _get_audit_path(output_dir: str = "./verification") -> Path:
    """Get the path to the audit trail file."""
    path = Path(output_dir) / DEFAULT_AUDIT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def track_source(
    claim_id: str,
    claim_text: str,
    source_url: str,
    verification_status: str,
    confidence_score: float = 0.0,
    evidence_excerpt: str = "",
    output_dir: str = "./verification",
) -> dict:
    """Append a source tracking entry to the audit trail.

    Args:
        claim_id: Claim identifier (e.g. "C1").
        claim_text: The factual claim text.
        source_url: URL where the claim was verified (or attempted).
        verification_status: VERIFIED/PARTIALLY_VERIFIED/UNVERIFIABLE/CONTRADICTED.
        confidence_score: Confidence score (0.0-1.0).
        evidence_excerpt: Excerpt from the source supporting/contradicting the claim.
        output_dir: Directory for the audit trail file.

    Returns:
        The entry dict that was written.
    """
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "claim_id": claim_id,
        "claim_text": claim_text,
        "source_url": source_url,
        "verification_status": verification_status,
        "confidence_score": confidence_score,
        "evidence_excerpt": evidence_excerpt[:500],  # Truncate long excerpts
    }

    audit_path = _get_audit_path(output_dir)
    with open(audit_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def read_audit_trail(output_dir: str = "./verification") -> list[dict]:
    """Read all entries from the audit trail.

    Args:
        output_dir: Directory containing the audit trail file.

    Returns:
        List of audit trail entry dicts.
    """
    audit_path = _get_audit_path(output_dir)
    if not audit_path.exists():
        return []

    entries = []
    for line in audit_path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return entries


class SourceTrackerTool(BaseTool):
    """Track and log the provenance of a verified data point.

    Appends an entry to the audit trail linking a claim to its origin URL
    and verification status.
    """

    name: str = "source_tracker"
    description: str = (
        "Track the provenance of a verified data point. "
        "Input: JSON with 'claim_id', 'claim_text', 'source_url', "
        "'verification_status', and optionally 'confidence_score' and "
        "'evidence_excerpt'. "
        "Appends to the audit trail and returns the entry."
    )

    output_dir: str = Field(
        default="./verification",
        description="Directory for the audit trail file",
    )

    def _run(self, input_data: str) -> str:
        """Track a source in the audit trail.

        Args:
            input_data: JSON string with claim tracking data.

        Returns:
            JSON string confirming the tracked entry.
        """
        try:
            data = json.loads(input_data)
        except (json.JSONDecodeError, TypeError):
            return json.dumps(
                {
                    "error": "Invalid input. Expected JSON with claim tracking data.",
                }
            )

        entry = track_source(
            claim_id=data.get("claim_id", "unknown"),
            claim_text=data.get("claim_text", ""),
            source_url=data.get("source_url", ""),
            verification_status=data.get("verification_status", "UNVERIFIABLE"),
            confidence_score=float(data.get("confidence_score", 0.0)),
            evidence_excerpt=data.get("evidence_excerpt", ""),
            output_dir=self.output_dir,
        )

        return json.dumps({"tracked": True, "entry": entry}, indent=2)
