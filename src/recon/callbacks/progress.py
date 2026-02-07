"""Rich progress tracking callback for CrewAI execution.

Provides live terminal output showing which agent is working,
what tools are being called, and phase transitions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from rich.console import Console
from rich.table import Table


class ProgressTracker:
    """Track and display pipeline progress using Rich.

    This is not a CrewAI callback class (those have a specific interface).
    Instead, this is a helper used by flow_builder/ResearchFlow to
    display progress between and during crew executions.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self.start_time = datetime.now(UTC)
        self.current_phase = ""
        self.current_agent = ""
        self.events: list[dict[str, Any]] = []

    def phase_start(self, phase: str, description: str = "") -> None:
        """Mark the start of a pipeline phase."""
        self.current_phase = phase
        self._log_event("phase_start", phase, description)

        phase_labels = {
            "investigation": "[bold cyan]Phase 1: Investigation[/] (parallel researchers)",
            "verification": "[bold cyan]Phase 2: Verification[/] (fact-checking)",
            "synthesis": "[bold cyan]Phase 3: Synthesis[/] (producing final report)",
        }

        label = phase_labels.get(phase, f"[bold cyan]{phase}[/]")
        if description:
            label += f" - {description}"

        self.console.print(f"\n{label}")

    def phase_end(self, phase: str, output_path: str = "") -> None:
        """Mark the end of a pipeline phase."""
        self._log_event("phase_end", phase, output_path)

        if output_path:
            self.console.print(f"  [green]Done:[/] {output_path}")

    def phase_skip(self, phase: str, reason: str = "") -> None:
        """Mark a phase as skipped."""
        self._log_event("phase_skip", phase, reason)
        self.console.print(f"  [dim]Skipped: {phase}[/]" + (f" ({reason})" if reason else ""))

    def agent_start(self, agent_name: str, output_file: str = "") -> None:
        """Mark the start of an agent execution."""
        self.current_agent = agent_name
        self._log_event("agent_start", agent_name, output_file)
        self.console.print(
            f"  [dim]Agent:[/] {agent_name}" + (f" -> {output_file}" if output_file else "")
        )

    def agent_end(self, agent_name: str, output_file: str = "") -> None:
        """Mark the end of an agent execution."""
        self._log_event("agent_end", agent_name, output_file)
        if output_file:
            self.console.print(f"  [green]Done:[/] {output_file}")

    def error(self, phase: str, message: str) -> None:
        """Report an error."""
        self._log_event("error", phase, message)
        self.console.print(f"  [red]Error in {phase}:[/] {message}")

    def pipeline_start(self, topic: str) -> None:
        """Mark the start of the full pipeline."""
        self._log_event("pipeline_start", "pipeline", topic)
        self.console.print("\n[bold]Starting research pipeline...[/]\n")

    def pipeline_end(self) -> None:
        """Mark the end of the full pipeline."""
        elapsed = (datetime.now(UTC) - self.start_time).total_seconds()
        self._log_event("pipeline_end", "pipeline", f"{elapsed:.1f}s")
        self.console.print(f"\n[green]Research complete.[/] ({elapsed:.1f}s)")

    def summary(
        self,
        plan_topic: str,
        research_files: list[str],
        verification_report: str,
        final_report: str,
    ) -> None:
        """Print a summary table of all outputs."""
        table = Table(title="Pipeline Output")
        table.add_column("Phase", style="cyan")
        table.add_column("Files", style="white")

        for f in research_files:
            table.add_row("Investigation", f)

        if verification_report:
            table.add_row("Verification", verification_report)

        if final_report:
            table.add_row("Synthesis", final_report)

        self.console.print(table)

    def _log_event(self, event_type: str, subject: str, detail: str = "") -> None:
        """Log an event for later analysis."""
        self.events.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "type": event_type,
                "subject": subject,
                "detail": detail,
            }
        )
