"""Audit trail callback - logs every significant action during pipeline execution.

Dual-write strategy:
1. **SQLite** ``events`` table (via :func:`recon.db.insert_event`) — structured,
   queryable, cross-run.
2. **JSON Lines** file (``audit-log.jsonl``) — human-readable fallback that
   works even when the DB is unavailable or the pipeline crashes mid-write.

When a ``sqlite3.Connection`` is provided the logger writes to both sinks.
When it is *None* (the default, for backward compatibility) only JSONL is
used.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3  # noqa: TC003
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recon import db as _db


class AuditLogger:
    """Log agent actions to a JSON Lines audit file **and** the knowledge DB.

    Each entry records: timestamp, run_id, phase, agent, action, detail, metadata.
    The audit log complements the SourceTracker tool (which tracks
    claim-level provenance) by providing pipeline-level provenance.
    """

    def __init__(
        self,
        output_dir: str = "./output",
        run_id: str = "",
        conn: sqlite3.Connection | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.run_id = run_id
        self.conn = conn
        self.log_path = Path(output_dir) / "audit-log.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict[str, Any]] = []
        self._phase_starts: dict[str, datetime] = {}

    def log(
        self,
        phase: str,
        agent: str,
        action: str,
        detail: str = "",
        metadata: dict[str, Any] | None = None,
        task: str | None = None,
        tokens_used: int | None = None,
    ) -> None:
        """Write an audit log entry to both JSONL and DB.

        Args:
            phase: Pipeline phase (investigation, verification, synthesis).
            agent: Agent name or identifier.
            action: What happened (tool_call, task_start, task_end, error, etc.).
            detail: Human-readable description.
            metadata: Optional structured data.
            task: Optional task name / identifier.
            tokens_used: Optional token count consumed by this action.
        """
        timestamp: str = datetime.now(UTC).isoformat()
        truncated_detail = detail[:2000]

        entry: dict[str, Any] = {
            "timestamp": timestamp,
            "run_id": self.run_id,
            "phase": phase,
            "agent": agent,
            "action": action,
            "detail": truncated_detail,
        }
        if task:
            entry["task"] = task
        if tokens_used is not None:
            entry["tokens_used"] = tokens_used
        if metadata:
            entry["metadata"] = metadata

        self._entries.append(entry)

        # --- Sink 1: JSONL (always) ---
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # --- Sink 2: SQLite (when available) ---
        if self.conn is not None:
            with contextlib.suppress(Exception):
                _db.insert_event(
                    self.conn,
                    run_id=self.run_id,
                    timestamp=timestamp,
                    action=action,
                    phase=phase,
                    agent=agent,
                    task=task,
                    detail=truncated_detail if truncated_detail else None,
                    metadata=metadata,
                    tokens_used=tokens_used,
                )

    def log_phase_start(self, phase: str) -> None:
        """Log the start of a pipeline phase."""
        self._phase_starts[phase] = datetime.now(UTC)
        self.log(phase=phase, agent="pipeline", action="phase_start")

    def log_phase_end(self, phase: str, output_files: list[str] | None = None) -> None:
        """Log the end of a pipeline phase with duration."""
        duration: float | None = None
        start = self._phase_starts.pop(phase, None)
        if start:
            duration = (datetime.now(UTC) - start).total_seconds()

        meta: dict[str, Any] = {"output_files": output_files or []}
        if duration is not None:
            meta["duration_seconds"] = round(duration, 1)

        self.log(
            phase=phase,
            agent="pipeline",
            action="phase_end",
            metadata=meta,
        )

    def log_agent_start(self, phase: str, agent_name: str) -> None:
        """Log an agent starting work."""
        self.log(phase=phase, agent=agent_name, action="agent_start")

    def log_agent_end(self, phase: str, agent_name: str, output_file: str = "") -> None:
        """Log an agent finishing work."""
        self.log(
            phase=phase,
            agent=agent_name,
            action="agent_end",
            metadata={"output_file": output_file},
        )

    def log_error(self, phase: str, agent: str, error: str) -> None:
        """Log an error."""
        self.log(phase=phase, agent=agent, action="error", detail=error)

    def get_entries(self) -> list[dict[str, Any]]:
        """Return all logged entries."""
        return list(self._entries)

    def read_log(self) -> list[dict[str, Any]]:
        """Read the audit log from disk.

        Returns:
            List of audit log entries.
        """
        if not self.log_path.exists():
            return []

        entries = []
        for line in self.log_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries
