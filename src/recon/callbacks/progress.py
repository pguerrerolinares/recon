"""Rich progress tracking callback for CrewAI execution.

Provides live terminal output with spinners showing which phase is active,
what agents are working, and phase transitions in real time.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Phase display configuration.
PHASE_LABELS: dict[str, str] = {
    "investigation": "Phase 1: Investigation",
    "verification": "Phase 2: Verification",
    "synthesis": "Phase 3: Synthesis",
}

PHASE_DESCRIPTIONS: dict[str, str] = {
    "investigation": "Parallel researchers gathering data",
    "verification": "Fact-checking claims against sources",
    "synthesis": "Producing final report",
}

SPINNER_CHARS = ["|", "/", "-", "\\"]


class ProgressTracker:
    """Track and display pipeline progress using Rich Live.

    This is not a CrewAI callback class (those have a specific interface).
    Instead, this is a helper used by flow_builder/ResearchFlow to
    display progress between and during crew executions.

    Uses Rich Live for real-time updates with spinners during active phases.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self.start_time = datetime.now(UTC)
        self.current_phase = ""
        self.current_agent = ""
        self.events: list[dict[str, Any]] = []

        # State for live display.
        self._phases: list[dict[str, Any]] = []
        self._agents: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._live: Live | None = None
        self._tick = 0

    def _build_display(self) -> Panel:
        """Build the Rich renderable for the current state."""
        self._tick += 1
        spinner = SPINNER_CHARS[self._tick % len(SPINNER_CHARS)]

        table = Table(
            show_header=True,
            header_style="bold",
            show_edge=False,
            pad_edge=False,
            expand=True,
        )
        table.add_column("Phase", style="cyan", width=28)
        table.add_column("Status", width=12)
        table.add_column("Detail", style="dim")

        for phase in self._phases:
            name = PHASE_LABELS.get(phase["name"], phase["name"])
            desc = phase.get("detail", "")

            if phase["status"] == "running":
                status = Text(f" {spinner} Running", style="bold yellow")
            elif phase["status"] == "done":
                elapsed = phase.get("elapsed", "")
                status = Text(f" Done{elapsed}", style="bold green")
            elif phase["status"] == "skipped":
                status = Text(" Skipped", style="dim")
            elif phase["status"] == "error":
                status = Text(" Failed", style="bold red")
            else:
                status = Text(f" {phase['status']}", style="dim")

            table.add_row(name, status, desc)

        # Show active agents under running phase.
        for agent in self._agents:
            if agent["status"] == "running":
                agent_text = f"  {spinner} {agent['name']}"
                file_text = agent.get("file", "")
                table.add_row(
                    Text(agent_text, style="dim cyan"),
                    Text("working", style="yellow"),
                    Text(file_text, style="dim"),
                )
            elif agent["status"] == "done":
                table.add_row(
                    Text(f"    {agent['name']}", style="dim"),
                    Text("done", style="green"),
                    Text(agent.get("file", ""), style="dim"),
                )

        # Show errors.
        for err in self._errors:
            table.add_row(
                Text("  Error", style="red"),
                Text("!", style="bold red"),
                Text(err[:80], style="red"),
            )

        elapsed = (datetime.now(UTC) - self.start_time).total_seconds()
        title = f"Recon Pipeline [{elapsed:.0f}s]"
        return Panel(table, title=title, border_style="blue", expand=True)

    def _refresh(self) -> None:
        """Refresh the live display if active."""
        if self._live is not None:
            self._live.update(self._build_display())

    def start_live(self) -> None:
        """Start the Rich Live display."""
        self._live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=2,
            transient=True,
        )
        self._live.start()

    def stop_live(self) -> None:
        """Stop the Rich Live display."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def phase_start(self, phase: str, description: str = "") -> None:
        """Mark the start of a pipeline phase."""
        self.current_phase = phase
        self._log_event("phase_start", phase, description)

        desc = description or PHASE_DESCRIPTIONS.get(phase, "")
        self._phases.append(
            {
                "name": phase,
                "status": "running",
                "detail": desc,
                "start": datetime.now(UTC),
            }
        )
        self._agents = []  # Reset agents for new phase.
        self._refresh()

    def phase_end(self, phase: str, output_path: str = "") -> None:
        """Mark the end of a pipeline phase."""
        self._log_event("phase_end", phase, output_path)

        for p in self._phases:
            if p["name"] == phase and p["status"] == "running":
                elapsed = (datetime.now(UTC) - p["start"]).total_seconds()
                p["status"] = "done"
                p["detail"] = output_path or p["detail"]
                p["elapsed"] = f" ({elapsed:.0f}s)"
                break
        self._refresh()

    def phase_skip(self, phase: str, reason: str = "") -> None:
        """Mark a phase as skipped."""
        self._log_event("phase_skip", phase, reason)
        self._phases.append(
            {
                "name": phase,
                "status": "skipped",
                "detail": reason,
            }
        )
        self._refresh()

    def agent_start(self, agent_name: str, output_file: str = "") -> None:
        """Mark the start of an agent execution."""
        self.current_agent = agent_name
        self._log_event("agent_start", agent_name, output_file)
        self._agents.append(
            {
                "name": agent_name,
                "status": "running",
                "file": output_file,
            }
        )
        self._refresh()

    def agent_end(self, agent_name: str, output_file: str = "") -> None:
        """Mark the end of an agent execution."""
        self._log_event("agent_end", agent_name, output_file)
        for a in self._agents:
            if a["name"] == agent_name and a["status"] == "running":
                a["status"] = "done"
                if output_file:
                    a["file"] = output_file
                break
        self._refresh()

    def error(self, phase: str, message: str) -> None:
        """Report an error."""
        self._log_event("error", phase, message)
        self._errors.append(f"{phase}: {message}")

        for p in self._phases:
            if p["name"] == phase and p["status"] == "running":
                p["status"] = "error"
                p["detail"] = message[:60]
                break
        self._refresh()

    def pipeline_start(self, topic: str) -> None:
        """Mark the start of the full pipeline."""
        self.start_time = datetime.now(UTC)
        self._log_event("pipeline_start", "pipeline", topic)
        self.start_live()

    def pipeline_end(self) -> None:
        """Mark the end of the full pipeline."""
        self.stop_live()
        elapsed = (datetime.now(UTC) - self.start_time).total_seconds()
        self._log_event("pipeline_end", "pipeline", f"{elapsed:.1f}s")
        self.console.print(f"\n[green bold]Research complete.[/] ({elapsed:.1f}s)\n")

    def summary(
        self,
        plan_topic: str,
        research_files: list[str],
        verification_report: str,
        final_report: str,
    ) -> None:
        """Print a summary table of all outputs."""
        table = Table(title="Pipeline Output", show_edge=True)
        table.add_column("Phase", style="cyan")
        table.add_column("File", style="white")

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
