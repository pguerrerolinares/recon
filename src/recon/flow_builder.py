"""Flow builder - Translates a ReconPlan into an executable pipeline.

Provides two execution modes:
1. build_and_run() - Simple function-based execution (used by CLI)
2. ResearchFlow - Full CrewAI Flow with state persistence (advanced usage)

Both modes execute the same 3-phase pipeline:
Investigation (parallel) -> Verification (fact-checking) -> Synthesis (report)
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from recon.callbacks.audit import AuditLogger
from recon.callbacks.progress import ProgressTracker
from recon.config import ReconPlan  # noqa: TC001
from recon.crews.investigation.crew import build_investigation_crew
from recon.crews.synthesis.crew import build_synthesis_crew
from recon.crews.verification.crew import build_verification_crew
from recon.providers.llm import create_llm
from recon.providers.search import create_search_tools


def build_and_run(
    plan: ReconPlan,
    verbose: bool = False,
    console: Console | None = None,
) -> None:
    """Build and execute the full 3-phase research pipeline.

    Args:
        plan: Validated ReconPlan.
        verbose: Whether to show detailed output.
        console: Rich console for output.
    """
    if console is None:
        console = Console()

    progress = ProgressTracker(console=console)
    audit = AuditLogger(output_dir=plan.output_dir)

    # Ensure output directories exist
    Path(plan.research_dir).mkdir(parents=True, exist_ok=True)
    Path(plan.output_dir).mkdir(parents=True, exist_ok=True)
    if plan.verify:
        Path(plan.verification_dir).mkdir(parents=True, exist_ok=True)

    # Create LLM and tools
    llm = create_llm(plan)
    search_tools = create_search_tools(plan)
    investigations = plan.get_investigations()

    progress.pipeline_start(plan.topic)

    try:
        # --- Phase 1: Investigation ---
        progress.phase_start("investigation")
        audit.log_phase_start("investigation")

        for inv in investigations:
            output_file = f"{plan.research_dir}/{inv.id}-{inv.name.lower().replace(' ', '-')}.md"
            progress.agent_start(inv.name, output_file)

        investigation_crew = build_investigation_crew(
            plan=plan,
            investigations=investigations,
            llm=llm,
            search_tools=search_tools,
            verbose=verbose,
        )

        try:
            investigation_crew.kickoff()
        except Exception as e:
            progress.error("investigation", str(e))
            audit.log_error("investigation", "investigation_crew", str(e))
            raise

        research_files = []
        for inv in investigations:
            output_file = f"{plan.research_dir}/{inv.id}-{inv.name.lower().replace(' ', '-')}.md"
            progress.agent_end(inv.name, output_file)
            research_files.append(output_file)

        audit.log_phase_end("investigation", output_files=research_files)

        # --- Phase 2: Verification ---
        verification_report = ""
        if plan.verify:
            progress.phase_start("verification")
            audit.log_phase_start("verification")

            verification_crew = build_verification_crew(
                plan=plan,
                llm=llm,
                search_tools=search_tools,
                research_dir=plan.research_dir,
                verbose=verbose,
            )

            if verification_crew is not None:
                try:
                    verification_crew.kickoff()
                    verification_report = f"{plan.verification_dir}/report.md"
                    progress.phase_end("verification", verification_report)
                    audit.log_phase_end("verification", output_files=[verification_report])
                except Exception as e:
                    progress.error("verification", str(e))
                    audit.log_error("verification", "verification_crew", str(e))
                    # Verification failure is non-fatal.
            else:
                progress.phase_skip("verification", "no research files found")
        else:
            progress.phase_skip("verification", "disabled")

        # --- Phase 3: Synthesis ---
        progress.phase_start("synthesis")
        audit.log_phase_start("synthesis")

        synthesis_crew = build_synthesis_crew(
            plan=plan,
            llm=llm,
            research_dir=plan.research_dir,
            verification_dir=plan.verification_dir if plan.verify else None,
            verbose=verbose,
        )

        try:
            synthesis_crew.kickoff()
        except Exception as e:
            progress.error("synthesis", str(e))
            audit.log_error("synthesis", "synthesis_crew", str(e))
            raise

        final_report = f"{plan.output_dir}/final-report.md"
        progress.phase_end("synthesis", final_report)
        audit.log_phase_end("synthesis", output_files=[final_report])

    finally:
        # Always stop the live display to restore terminal state.
        progress.stop_live()

    # --- Summary ---
    progress.pipeline_end()
    progress.summary(
        plan_topic=plan.topic,
        research_files=research_files,
        verification_report=verification_report,
        final_report=final_report,
    )
