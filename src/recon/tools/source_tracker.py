"""SourceTrackerTool - Track provenance of data points in an audit trail.

Dual-write strategy:
1. **JSONL** file (``audit-trail.jsonl``) — human-readable per-run file.
2. **SQLite** ``claims`` / ``claim_sources`` / ``claim_history`` tables —
   queryable cross-run knowledge (when a ``sqlite3.Connection`` is provided).

When ``conn`` is *None* the behaviour is identical to the pre-v0.3 version.
"""

from __future__ import annotations

import json
import sqlite3  # noqa: TC003
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from crewai.tools import BaseTool
from pydantic import Field

from recon import db as _db

# Default audit trail file name.
DEFAULT_AUDIT_FILE = "audit-trail.jsonl"


def _extract_domain(url: str) -> str:
    """Extract domain from a URL, stripping ``www.``."""
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return "unknown"


def _get_audit_path(output_dir: str = "./verification") -> Path:
    """Get the path to the audit trail file."""
    path = Path(output_dir) / DEFAULT_AUDIT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# Map SourceTracker statuses → DB support_type values used in claim_sources.
_STATUS_TO_SUPPORT: dict[str, str] = {
    "VERIFIED": "supports",
    "PARTIALLY_VERIFIED": "partial",
    "UNVERIFIABLE": "insufficient",
    "CONTRADICTED": "contradicts",
}


def track_source(
    claim_id: str,
    claim_text: str,
    source_url: str,
    verification_status: str,
    confidence_score: float = 0.0,
    evidence_excerpt: str = "",
    output_dir: str = "./verification",
    conn: sqlite3.Connection | None = None,
    run_id: str | None = None,
) -> dict:
    """Append a source tracking entry to the audit trail **and** the DB.

    Args:
        claim_id: Claim identifier (e.g. "C1").
        claim_text: The factual claim text.
        source_url: URL where the claim was verified (or attempted).
        verification_status: VERIFIED/PARTIALLY_VERIFIED/UNVERIFIABLE/CONTRADICTED.
        confidence_score: Confidence score (0.0-1.0).
        evidence_excerpt: Excerpt from the source supporting/contradicting the claim.
        output_dir: Directory for the audit trail file.
        conn: Optional SQLite connection for DB writes.
        run_id: Optional run identifier (needed for DB writes).

    Returns:
        The entry dict that was written.
    """
    now = datetime.now(UTC).isoformat()
    truncated_excerpt = evidence_excerpt[:500]

    entry = {
        "timestamp": now,
        "claim_id": claim_id,
        "claim_text": claim_text,
        "source_url": source_url,
        "verification_status": verification_status,
        "confidence_score": confidence_score,
        "evidence_excerpt": truncated_excerpt,
    }

    # --- Sink 1: JSONL (always) ---
    audit_path = _get_audit_path(output_dir)
    with open(audit_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # --- Sink 2: SQLite (when available) ---
    if conn is not None:
        try:
            _db.upsert_claim(
                conn,
                claim_id=claim_id,
                text=claim_text,
                verification_status=verification_status,
                confidence=confidence_score,
                evidence_excerpt=truncated_excerpt,
                cited_source=source_url,
                run_id=run_id,
                timestamp=now,
            )
            if source_url:
                domain = _extract_domain(source_url)
                _db.upsert_claim_source(
                    conn,
                    claim_id=claim_id,
                    source_url=source_url,
                    domain=domain,
                    support_type=_STATUS_TO_SUPPORT.get(verification_status, "unknown"),
                    evidence_excerpt=truncated_excerpt,
                    retrieved_at=now,
                )
            if run_id:
                _db.insert_claim_history(
                    conn,
                    claim_id=claim_id,
                    run_id=run_id,
                    verified_at=now,
                    new_status=verification_status,
                    new_confidence=confidence_score,
                    method="source_tracker",
                )
        except Exception:
            # DB write failure must never break the pipeline.
            pass

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
    and verification status.  When ``conn`` and ``run_id`` are set, also
    writes to the knowledge DB.
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
    conn: sqlite3.Connection | None = Field(
        default=None,
        description="Optional SQLite connection for DB writes",
    )
    run_id: str | None = Field(
        default=None,
        description="Optional run identifier for DB writes",
    )

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, input_data: str) -> str:
        """Track a source in the audit trail.

        Args:
            input_data: JSON string with claim tracking data.

        Returns:
            JSON string confirming the tracked entry.
        """
        from recon.tools._helpers import parse_tool_input

        data, err = parse_tool_input(input_data)
        if err:
            return err

        assert data is not None  # guaranteed by parse_tool_input when err is None

        entry = track_source(
            claim_id=data.get("claim_id", "unknown"),
            claim_text=data.get("claim_text", ""),
            source_url=data.get("source_url", ""),
            verification_status=data.get("verification_status", "UNVERIFIABLE"),
            confidence_score=float(data.get("confidence_score", 0.0)),
            evidence_excerpt=data.get("evidence_excerpt", ""),
            output_dir=self.output_dir,
            conn=self.conn,
            run_id=self.run_id,
        )

        return json.dumps({"tracked": True, "entry": entry}, indent=2)
