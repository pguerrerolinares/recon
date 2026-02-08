"""Flow builder - Translates a ReconPlan into an executable pipeline.

Provides two execution modes:
1. build_and_run() - Simple function-based execution (used by CLI)
2. ResearchFlow - Full CrewAI Flow with state persistence (advanced usage)

Both modes execute the same 3-phase pipeline:
Investigation (parallel) -> Verification (fact-checking) -> Synthesis (report)

Incremental mode (default): phases with existing output files are skipped.
Use force=True to re-run all phases regardless.

When memory is enabled, the pipeline also:
- Queries prior knowledge before investigation
- Ingests all outputs into memory after synthesis
"""

from __future__ import annotations

import logging
import signal
import uuid
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

logger = logging.getLogger(__name__)


class _PhaseTimeoutError(Exception):
    """Raised when a pipeline phase exceeds its timeout."""


def _timeout_handler(signum: int, frame: object) -> None:
    """Signal handler that raises _PhaseTimeout."""
    raise _PhaseTimeoutError("Phase timed out")


def _has_phase_output(directory: str, pattern: str = "*.md") -> list[str]:
    """Check if a phase directory contains non-empty output files.

    Args:
        directory: Path to the phase output directory.
        pattern: Glob pattern for output files.

    Returns:
        List of existing non-empty file paths, or empty list.
    """
    path = Path(directory)
    if not path.exists():
        return []
    files = sorted(path.glob(pattern))
    return [str(f) for f in files if f.stat().st_size > 0]


def _query_memory(plan: ReconPlan, audit: AuditLogger) -> str | None:
    """Query cross-run memory for relevant prior knowledge.

    Returns a formatted string of prior findings, or None if memory is
    disabled, unavailable, or has no relevant results.
    """
    if not plan.memory.enabled:
        return None

    try:
        from recon.memory.store import MemoryStore
    except ImportError:
        logger.warning("Memory enabled but memvid-sdk not installed. Skipping memory query.")
        return None

    try:
        store = MemoryStore(
            path=plan.memory.path,
            embedding_provider=plan.memory.embedding_provider,
        )
        results = store.query(
            topic=plan.topic,
            questions=plan.questions,
            k=5,
        )
        store.close()
    except Exception:
        logger.warning("Memory query failed, proceeding without prior knowledge", exc_info=True)
        return None

    if not results:
        return None

    # Format results into a prior knowledge block
    parts: list[str] = []
    for i, hit in enumerate(results, 1):
        title = hit.get("title", "Untitled")
        text = hit.get("text", "")
        if text:
            # Truncate individual results to keep prior knowledge manageable
            snippet = text[:500] + "..." if len(text) > 500 else text
            parts.append(f"{i}. **{title}**\n   {snippet}")

    if not parts:
        return None

    prior = "PRIOR RESEARCH FINDINGS (from previous runs):\n\n" + "\n\n".join(parts)
    audit.log(
        phase="memory",
        agent="memory_store",
        action="query",
        detail=f"Found {len(parts)} prior findings for '{plan.topic}'",
        metadata={"results_count": len(parts), "topic": plan.topic},
    )
    logger.info("Found %d relevant prior findings for topic '%s'", len(parts), plan.topic)
    return prior


def _ingest_to_memory(
    plan: ReconPlan,
    run_id: str,
    research_files: list[str],
    verification_report: str,
    final_report: str,
    audit: AuditLogger,
) -> None:
    """Ingest pipeline outputs into cross-run memory."""
    if not plan.memory.enabled:
        return

    try:
        from recon.memory.store import MemoryStore
    except ImportError:
        logger.warning("Memory enabled but memvid-sdk not installed. Skipping ingest.")
        return

    try:
        store = MemoryStore(
            path=plan.memory.path,
            embedding_provider=plan.memory.embedding_provider,
        )

        # Ingest research files
        count = store.ingest_research(plan.research_dir, topic=plan.topic, run_id=run_id)

        # Ingest verification report
        if verification_report:
            store.ingest_report(
                verification_report,
                topic=plan.topic,
                run_id=run_id,
                phase="verification",
            )

        # Ingest final report
        if final_report:
            store.ingest_report(
                final_report,
                topic=plan.topic,
                run_id=run_id,
                phase="synthesis",
            )

        stats = store.stats()
        store.close()

        audit.log(
            phase="memory",
            agent="memory_store",
            action="ingest",
            detail=f"Ingested {count} documents into {plan.memory.path}",
            metadata={
                "documents_ingested": count,
                "run_id": run_id,
                "memory_stats": stats,
                "memory_path": plan.memory.path,
            },
        )
        logger.info("Ingested %d documents into memory (%s)", count, plan.memory.path)

    except Exception:
        logger.warning(
            "Memory ingest failed, pipeline results are still saved to disk",
            exc_info=True,
        )


def build_and_run(
    plan: ReconPlan,
    verbose: bool = False,
    console: Console | None = None,
    force: bool = False,
) -> None:
    """Build and execute the full 3-phase research pipeline.

    In incremental mode (force=False, the default), phases whose output
    files already exist are skipped. Use force=True to re-run everything.

    Args:
        plan: Validated ReconPlan.
        verbose: Whether to show detailed output.
        console: Rich console for output.
        force: If True, re-run all phases even if output files exist.
    """
    if console is None:
        console = Console()

    progress = ProgressTracker(console=console)
    audit = AuditLogger(output_dir=plan.output_dir)
    run_id = f"run-{uuid.uuid4().hex[:8]}"

    # Ensure output directories exist
    Path(plan.research_dir).mkdir(parents=True, exist_ok=True)
    Path(plan.output_dir).mkdir(parents=True, exist_ok=True)
    if plan.verify:
        Path(plan.verification_dir).mkdir(parents=True, exist_ok=True)

    # Create LLM and tools
    llm = create_llm(plan)
    search_tools = create_search_tools(plan)
    investigations = plan.get_investigations()

    # Query memory for prior knowledge (always, even on incremental runs)
    prior_knowledge = _query_memory(plan, audit)

    # Check existing output for incremental skip
    existing_research = _has_phase_output(plan.research_dir)
    existing_verification = _has_phase_output(plan.verification_dir)
    existing_synthesis = _has_phase_output(plan.output_dir, "final-report.md")

    progress.pipeline_start(plan.topic)

    try:
        # --- Phase 1: Investigation ---
        skip_investigation = not force and len(existing_research) > 0

        if skip_investigation:
            research_files = existing_research
            progress.phase_skip(
                "investigation",
                f"{len(existing_research)} research files already exist. Use --force to re-run.",
            )
            audit.log(
                phase="investigation",
                agent="pipeline",
                action="phase_skip",
                detail=f"Skipped: {len(existing_research)} files exist",
                metadata={"files": existing_research},
            )
        else:
            progress.phase_start("investigation")
            audit.log_phase_start("investigation")

            for inv in investigations:
                output_file = (
                    f"{plan.research_dir}/{inv.id}-{inv.name.lower().replace(' ', '-')}.md"
                )
                progress.agent_start(inv.name, output_file)

            investigation_crew = build_investigation_crew(
                plan=plan,
                investigations=investigations,
                llm=llm,
                search_tools=search_tools,
                verbose=verbose,
                prior_knowledge=prior_knowledge,
            )

            try:
                investigation_crew.kickoff()
            except Exception as e:
                progress.error("investigation", str(e))
                audit.log_error("investigation", "investigation_crew", str(e))
                raise

            research_files = []
            for inv in investigations:
                output_file = (
                    f"{plan.research_dir}/{inv.id}-{inv.name.lower().replace(' ', '-')}.md"
                )
                progress.agent_end(inv.name, output_file)
                research_files.append(output_file)

            audit.log_phase_end("investigation", output_files=research_files)

        # --- Phase 2: Verification ---
        verification_report = ""
        if plan.verify:
            skip_verification = not force and len(existing_verification) > 0

            if skip_verification:
                verification_report = existing_verification[0]
                progress.phase_skip(
                    "verification",
                    "Verification report already exists. Use --force to re-run.",
                )
            else:
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
                    timeout = plan.verification.phase_timeout
                    old_handler: signal.Handlers | None = None
                    alarm_set = False
                    try:
                        # Set alarm-based timeout (Unix only)
                        try:
                            old_handler = signal.signal(
                                signal.SIGALRM, _timeout_handler
                            )
                            signal.alarm(timeout)
                            alarm_set = True
                        except (AttributeError, OSError):
                            # SIGALRM not available (Windows) -- no timeout
                            pass

                        verification_crew.kickoff()

                        verification_report = f"{plan.verification_dir}/report.md"
                        progress.phase_end("verification", verification_report)
                        audit.log_phase_end(
                            "verification", output_files=[verification_report]
                        )
                    except _PhaseTimeoutError:
                        logger.warning(
                            "Verification timed out after %ds, "
                            "proceeding to synthesis",
                            timeout,
                        )
                        progress.error(
                            "verification",
                            f"Timed out after {timeout}s. "
                            "Proceeding to synthesis.",
                        )
                        audit.log_error(
                            "verification",
                            "verification_crew",
                            f"Phase timeout ({timeout}s)",
                        )
                    except Exception as e:
                        progress.error("verification", str(e))
                        audit.log_error(
                            "verification", "verification_crew", str(e)
                        )
                        # Verification failure is non-fatal.
                    finally:
                        # Restore original signal handler
                        if alarm_set:
                            signal.alarm(0)
                            if old_handler is not None:
                                signal.signal(signal.SIGALRM, old_handler)
                else:
                    progress.phase_skip("verification", "no research files found")
        else:
            progress.phase_skip("verification", "disabled")

        # --- Phase 3: Synthesis ---
        skip_synthesis = not force and len(existing_synthesis) > 0

        if skip_synthesis:
            final_report = existing_synthesis[0]
            progress.phase_skip(
                "synthesis",
                "Final report already exists. Use --force to re-run.",
            )
        else:
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

    # --- Memory Ingest (always, even on incremental runs) ---
    _ingest_to_memory(
        plan=plan,
        run_id=run_id,
        research_files=research_files,
        verification_report=verification_report,
        final_report=final_report,
        audit=audit,
    )

    # --- Summary ---
    progress.pipeline_end()
    progress.summary(
        plan_topic=plan.topic,
        research_files=research_files,
        verification_report=verification_report,
        final_report=final_report,
    )
