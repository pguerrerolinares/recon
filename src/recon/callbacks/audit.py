"""Audit trail callback - logs every significant action during pipeline execution.

Writes a structured JSON Lines file that can be used to:
1. Reconstruct what each agent did and when
2. Debug failed runs
3. Verify that agents followed their instructions
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class AuditLogger:
    """Log agent actions to a JSON Lines audit file.

    Each entry records: timestamp, run_id, phase, agent, action, detail, metadata.
    The audit log complements the SourceTracker tool (which tracks
    claim-level provenance) by providing pipeline-level provenance.
    """

    def __init__(self, output_dir: str = "./output", run_id: str = "") -> None:
        self.output_dir = output_dir
        self.run_id = run_id
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
    ) -> None:
        """Write an audit log entry.

        Args:
            phase: Pipeline phase (investigation, verification, synthesis).
            agent: Agent name or identifier.
            action: What happened (tool_call, task_start, task_end, error, etc.).
            detail: Human-readable description.
            metadata: Optional structured data.
        """
        timestamp: Any = datetime.now(UTC).isoformat()
        entry: dict[str, Any] = {
            "timestamp": timestamp,
            "run_id": self.run_id,
            "phase": phase,
            "agent": agent,
            "action": action,
            "detail": detail[:2000],
        }
        if metadata:
            entry["metadata"] = metadata

        self._entries.append(entry)

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

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
