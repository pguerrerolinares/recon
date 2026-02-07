"""Flow builder - Translates a ReconPlan into a CrewAI execution pipeline.

This is the core bridge between Recon's simple YAML config and CrewAI's
full agent/crew/flow API. The user writes a simple plan.yaml, and this
module builds and runs the entire 3-phase pipeline:

1. Investigation (parallel researchers)
2. Verification (fact-checker)
3. Synthesis (director)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from recon.config import ReconPlan
from recon.crews.investigation.crew import build_investigation_crew
from recon.crews.synthesis.crew import build_synthesis_crew
from recon.crews.verification.crew import build_verification_crew
from recon.providers.llm import create_llm
from recon.providers.search import create_search_tools


def build_and_run(plan: ReconPlan, verbose: bool = False, console: Console | None = None) -> None:
    """Build and execute the full 3-phase research pipeline.

    Args:
        plan: Validated ReconPlan.
        verbose: Whether to show detailed output.
        console: Rich console for output.
    """
    if console is None:
        console = Console()

    # Ensure output directories exist
    Path(plan.research_dir).mkdir(parents=True, exist_ok=True)
    Path(plan.output_dir).mkdir(parents=True, exist_ok=True)
    if plan.verify:
        Path(plan.verification_dir).mkdir(parents=True, exist_ok=True)

    # Create LLM and tools
    llm = create_llm(plan)
    search_tools = create_search_tools(plan)
    investigations = plan.get_investigations()

    # --- Phase 1: Investigation ---
    console.print("[bold cyan]Phase 1: Investigation[/] (parallel researchers)")

    for inv in investigations:
        output_file = (
            f"{plan.research_dir}/{inv.id}-{inv.name.lower().replace(' ', '-')}.md"
        )
        console.print(f"  [dim]Agent:[/] {inv.name} -> {output_file}")

    investigation_crew = build_investigation_crew(
        plan=plan,
        investigations=investigations,
        llm=llm,
        search_tools=search_tools,
        verbose=verbose,
    )

    investigation_crew.kickoff()

    for inv in investigations:
        output_file = (
            f"{plan.research_dir}/{inv.id}-{inv.name.lower().replace(' ', '-')}.md"
        )
        console.print(f"  [green]Done:[/] {output_file}")

    # --- Phase 2: Verification ---
    if plan.verify:
        console.print("\n[bold cyan]Phase 2: Verification[/] (fact-checking)")

        verification_crew = build_verification_crew(
            plan=plan,
            llm=llm,
            search_tools=search_tools,
            research_dir=plan.research_dir,
            verbose=verbose,
        )

        if verification_crew is not None:
            verification_crew.kickoff()
            console.print(f"  [green]Done:[/] {plan.verification_dir}/report.md")
        else:
            console.print("  [yellow]No research files found. Skipping verification.[/]")
    else:
        console.print("\n[dim]Phase 2: Verification (skipped)[/]")

    # --- Phase 3: Synthesis ---
    console.print("\n[bold cyan]Phase 3: Synthesis[/] (producing final report)")

    synthesis_crew = build_synthesis_crew(
        plan=plan,
        llm=llm,
        research_dir=plan.research_dir,
        verification_dir=plan.verification_dir if plan.verify else None,
        verbose=verbose,
    )

    synthesis_crew.kickoff()
    console.print(f"  [green]Done:[/] {plan.output_dir}/final-report.md")
