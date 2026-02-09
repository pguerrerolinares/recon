"""Flow builder - Translates a ReconPlan into an executable pipeline.

Executes the 3-phase pipeline:
Investigation (parallel) -> Verification (fact-checking) -> Synthesis (report)

Incremental mode (default): phases with existing output files are skipped.
Use force=True to re-run all phases regardless.

When knowledge is enabled, the pipeline:
- Opens a SQLite connection and records the run in the ``runs`` table.
- Passes the connection to the AuditLogger, SourceTracker, and
  SourceExtractor so they can dual-write to the DB.
- Captures ``CrewOutput.token_usage`` and writes to ``token_usage`` table.
- Records ``phase_metrics`` for each pipeline phase.
- Queries prior knowledge (FTS5 claims search) before investigation.
"""

from __future__ import annotations

import contextlib
import json
import logging
import signal
import sqlite3  # noqa: TC003
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from recon import db as _db
from recon.callbacks.audit import AuditLogger
from recon.callbacks.progress import ProgressTracker
from recon.config import ReconPlan, estimate_cost  # noqa: TC001
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


# ------------------------------------------------------------------
# DB helpers
# ------------------------------------------------------------------


def _open_db(plan: ReconPlan) -> sqlite3.Connection | None:
    """Open (or create) the knowledge database if knowledge is enabled."""
    if not plan.knowledge.enabled:
        return None
    try:
        return _db.get_db(plan.knowledge.db_path)
    except Exception:
        logger.warning("Could not open knowledge DB, proceeding without DB", exc_info=True)
        return None


def _record_token_usage(
    conn: sqlite3.Connection | None,
    run_id: str,
    phase: str,
    model: str,
    crew_output: Any,
) -> None:
    """Extract token usage from a CrewOutput and write to the DB.

    CrewAI's ``CrewOutput.token_usage`` is a dict like:
    ``{"total_tokens": N, "prompt_tokens": N, "completion_tokens": N,
      "cached_tokens": N, "successful_requests": N}``
    """
    if conn is None:
        return

    usage: dict[str, Any] = {}
    if hasattr(crew_output, "token_usage") and crew_output.token_usage:
        usage = dict(crew_output.token_usage)
    if not usage:
        return

    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    total_tokens = int(usage.get("total_tokens", 0))
    cached_tokens = int(usage.get("cached_tokens", 0))
    successful_requests = int(usage.get("successful_requests", 0))

    cost = estimate_cost(model, prompt_tokens, completion_tokens)

    with contextlib.suppress(Exception):
        _db.insert_token_usage(
            conn,
            run_id=run_id,
            phase=phase,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            successful_requests=successful_requests,
            estimated_cost_usd=cost,
        )


def _query_prior_knowledge(
    conn: sqlite3.Connection | None,
    plan: ReconPlan,
    audit: AuditLogger,
) -> str | None:
    """Query the knowledge DB for prior claims related to the topic.

    Uses FTS5 keyword search over the claims table.

    Returns a formatted string of prior findings, or None.
    """
    if not plan.knowledge.enabled:
        return None

    if conn is None:
        return None

    try:
        # Search for claims related to the topic
        keywords = plan.topic.split()[:6]  # Use first 6 words as query
        query = " OR ".join(keywords)
        results = _db.search_claims_fts(conn, query, limit=10)

        if not results:
            return None

        parts: list[str] = []
        for i, claim in enumerate(results, 1):
            status = claim.get("verification_status", "unknown")
            conf = claim.get("confidence")
            conf_str = f" (confidence: {conf:.0%})" if conf else ""
            parts.append(f"{i}. [{status}]{conf_str} {claim['text']}")

        prior = "PRIOR VERIFIED CLAIMS (from previous runs):\n\n" + "\n".join(parts)
        audit.log(
            phase="knowledge",
            agent="knowledge_db",
            action="query",
            detail=f"Found {len(results)} prior claims for '{plan.topic}'",
            metadata={"results_count": len(results), "topic": plan.topic},
        )
        logger.info("Found %d prior claims for topic '%s'", len(results), plan.topic)
        return prior
    except Exception:
        logger.debug("FTS5 query failed", exc_info=True)
        return None


def _write_run_manifest(
    plan: ReconPlan,
    run_id: str,
    audit: AuditLogger,
    research_files: list[str],
    verification_report: str,
    final_report: str,
    sources_summary: dict,
    pipeline_start: datetime,
) -> None:
    """Write a consolidated run manifest JSON file.

    The manifest captures key metrics from the pipeline run, providing
    a single-file summary useful for comparing runs and future migration
    to a SQLite database.
    """
    total_duration = (datetime.now(UTC) - pipeline_start).total_seconds()

    # Build phase metrics from audit entries
    phase_metrics: dict[str, dict] = {}
    for entry in audit.get_entries():
        if entry.get("action") == "phase_end":
            phase = entry["phase"]
            meta = entry.get("metadata", {})
            phase_metrics[phase] = {
                "status": "done",
                "duration_seconds": meta.get("duration_seconds"),
                "output_files": meta.get("output_files", []),
            }
        elif entry.get("action") == "phase_skip":
            phase_metrics[entry["phase"]] = {"status": "skipped"}
        elif entry.get("action") == "error":
            phase = entry["phase"]
            if phase not in phase_metrics:
                phase_metrics[phase] = {}
            phase_metrics[phase]["status"] = "error"

    # Enrich investigation metrics with source data
    if "investigation" in phase_metrics:
        phase_metrics["investigation"]["total_sources"] = sources_summary.get("total_urls", 0)
        phase_metrics["investigation"]["unique_sources"] = sources_summary.get("unique_urls", 0)
        phase_metrics["investigation"]["agents"] = len(research_files)

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "plan": {
            "topic": plan.topic,
            "depth": plan.depth.value,
            "model": plan.model,
            "provider": plan.provider,
            "verify": plan.verify,
            "search_provider": plan.search.provider,
            "auto_questions": plan.auto_questions,
        },
        "phases": phase_metrics,
        "total_duration_seconds": round(total_duration, 1),
        "output_files": {
            "research": research_files,
            "verification": verification_report,
            "final_report": final_report,
            "sources_json": f"{plan.research_dir}/sources.json",
        },
        "knowledge_enabled": plan.knowledge.enabled,
    }

    manifest_path = Path(plan.output_dir) / f"{run_id}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    audit.log(
        phase="pipeline",
        agent="manifest",
        action="manifest_written",
        detail=f"Run manifest: {manifest_path}",
        metadata={"manifest_path": str(manifest_path)},
    )
    logger.info("Run manifest written to %s", manifest_path)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


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

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    pipeline_start = datetime.now(UTC)

    # --- Open knowledge DB ---
    conn = _open_db(plan)

    # Register the run in the DB
    if conn is not None:
        with contextlib.suppress(Exception):
            _db.insert_run(
                conn,
                run_id=run_id,
                timestamp=pipeline_start.isoformat(),
                topic=plan.topic,
                depth=plan.depth.value,
                model=plan.model,
                provider=plan.provider,
                search_provider=plan.search.provider,
                verify=plan.verify,
                auto_questions=plan.auto_questions,
                config_json={
                    "knowledge": plan.knowledge.model_dump(),
                    "verification": plan.verification.model_dump(),
                    "synthesis": plan.synthesis.model_dump(),
                    "context_strategy": plan.context_strategy,
                },
            )

    progress = ProgressTracker(console=console)
    audit = AuditLogger(output_dir=plan.output_dir, run_id=run_id, conn=conn)

    # Ensure output directories exist
    Path(plan.research_dir).mkdir(parents=True, exist_ok=True)
    Path(plan.output_dir).mkdir(parents=True, exist_ok=True)
    if plan.verify:
        Path(plan.verification_dir).mkdir(parents=True, exist_ok=True)

    # Create LLM and tools
    llm = create_llm(plan)
    search_tools = create_search_tools(plan)
    investigations = plan.get_investigations()

    # Query prior knowledge (DB first, then legacy memvid fallback)
    prior_knowledge = _query_prior_knowledge(conn, plan, audit)

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
            if conn is not None:
                with contextlib.suppress(Exception):
                    _db.insert_phase_metric(
                        conn,
                        run_id=run_id,
                        phase="investigation",
                        status="skipped",
                        metadata={"files": existing_research},
                    )
        else:
            progress.phase_start("investigation")
            audit.log_phase_start("investigation")
            inv_started = datetime.now(UTC)

            # Record phase start in DB
            inv_metric_id = 0
            if conn is not None:
                with contextlib.suppress(Exception):
                    inv_metric_id = _db.insert_phase_metric(
                        conn,
                        run_id=run_id,
                        phase="investigation",
                        status="running",
                        started_at=inv_started.isoformat(),
                    )

            for inv in investigations:
                progress.agent_start(inv.name, inv.output_path(plan.research_dir))

            investigation_crew = build_investigation_crew(
                plan=plan,
                investigations=investigations,
                llm=llm,
                search_tools=search_tools,
                verbose=verbose,
                prior_knowledge=prior_knowledge,
            )

            try:
                inv_result = investigation_crew.kickoff()
            except Exception as e:
                progress.error("investigation", str(e))
                audit.log_error("investigation", "investigation_crew", str(e))
                if conn is not None and inv_metric_id:
                    with contextlib.suppress(Exception):
                        _db.update_phase_metric(
                            conn,
                            inv_metric_id,
                            status="error",
                            ended_at=datetime.now(UTC).isoformat(),
                        )
                raise

            research_files = []
            for inv in investigations:
                out = inv.output_path(plan.research_dir)
                progress.agent_end(inv.name, out)
                research_files.append(out)

            inv_ended = datetime.now(UTC)
            audit.log_phase_end("investigation", output_files=research_files)

            # Record token usage + phase completion
            _record_token_usage(conn, run_id, "investigation", plan.model, inv_result)

            if conn is not None and inv_metric_id:
                with contextlib.suppress(Exception):
                    _db.update_phase_metric(
                        conn,
                        inv_metric_id,
                        status="done",
                        ended_at=inv_ended.isoformat(),
                        duration_seconds=(inv_ended - inv_started).total_seconds(),
                        output_files=research_files,
                    )

        # Post-process: extract and deduplicate sources
        from recon.tools.source_extractor import write_sources_json

        sources_summary = write_sources_json(plan.research_dir, conn=conn, run_id=run_id)
        audit.log(
            phase="investigation",
            agent="source_extractor",
            action="sources_extracted",
            detail=(
                f"Extracted {sources_summary['unique_urls']} unique URLs "
                f"({sources_summary['total_urls']} total) from research files"
            ),
            metadata={
                "unique_urls": sources_summary["unique_urls"],
                "total_urls": sources_summary["total_urls"],
                "top_domains": dict(list(sources_summary["by_domain"].items())[:10]),
            },
        )

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
                if conn is not None:
                    with contextlib.suppress(Exception):
                        _db.insert_phase_metric(
                            conn,
                            run_id=run_id,
                            phase="verification",
                            status="skipped",
                        )
            else:
                progress.phase_start("verification")
                audit.log_phase_start("verification")
                ver_started = datetime.now(UTC)

                ver_metric_id = 0
                if conn is not None:
                    with contextlib.suppress(Exception):
                        ver_metric_id = _db.insert_phase_metric(
                            conn,
                            run_id=run_id,
                            phase="verification",
                            status="running",
                            started_at=ver_started.isoformat(),
                        )

                verification_crew = build_verification_crew(
                    plan=plan,
                    llm=llm,
                    search_tools=search_tools,
                    research_dir=plan.research_dir,
                    verbose=verbose,
                    conn=conn,
                    run_id=run_id,
                )

                if verification_crew is not None:
                    timeout = plan.verification.phase_timeout
                    old_handler: signal.Handlers | None = None
                    alarm_set = False
                    try:
                        # Set alarm-based timeout (Unix only)
                        try:
                            old_handler = signal.signal(  # type: ignore[assignment]
                                signal.SIGALRM, _timeout_handler
                            )
                            signal.alarm(timeout)
                            alarm_set = True
                        except (AttributeError, OSError):
                            # SIGALRM not available (Windows) -- no timeout
                            pass

                        ver_result = verification_crew.kickoff()

                        verification_report = f"{plan.verification_dir}/report.md"
                        ver_ended = datetime.now(UTC)
                        progress.phase_end("verification", verification_report)
                        audit.log_phase_end("verification", output_files=[verification_report])

                        _record_token_usage(conn, run_id, "verification", plan.model, ver_result)

                        if conn is not None and ver_metric_id:
                            with contextlib.suppress(Exception):
                                _db.update_phase_metric(
                                    conn,
                                    ver_metric_id,
                                    status="done",
                                    ended_at=ver_ended.isoformat(),
                                    duration_seconds=(ver_ended - ver_started).total_seconds(),
                                    output_files=[verification_report],
                                )
                    except _PhaseTimeoutError:
                        logger.warning(
                            "Verification timed out after %ds, proceeding to synthesis",
                            timeout,
                        )
                        progress.error(
                            "verification",
                            f"Timed out after {timeout}s. Proceeding to synthesis.",
                        )
                        audit.log_error(
                            "verification",
                            "verification_crew",
                            f"Phase timeout ({timeout}s)",
                        )
                        if conn is not None and ver_metric_id:
                            with contextlib.suppress(Exception):
                                _db.update_phase_metric(
                                    conn,
                                    ver_metric_id,
                                    status="timeout",
                                    ended_at=datetime.now(UTC).isoformat(),
                                )
                    except Exception as e:
                        progress.error("verification", str(e))
                        audit.log_error("verification", "verification_crew", str(e))
                        if conn is not None and ver_metric_id:
                            with contextlib.suppress(Exception):
                                _db.update_phase_metric(
                                    conn,
                                    ver_metric_id,
                                    status="error",
                                    ended_at=datetime.now(UTC).isoformat(),
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
                    if conn is not None and ver_metric_id:
                        with contextlib.suppress(Exception):
                            _db.update_phase_metric(
                                conn,
                                ver_metric_id,
                                status="skipped",
                            )
        else:
            progress.phase_skip("verification", "disabled")
            if conn is not None:
                with contextlib.suppress(Exception):
                    _db.insert_phase_metric(
                        conn,
                        run_id=run_id,
                        phase="verification",
                        status="disabled",
                    )

        # --- Phase 3: Synthesis ---
        skip_synthesis = not force and len(existing_synthesis) > 0

        if skip_synthesis:
            final_report = existing_synthesis[0]
            progress.phase_skip(
                "synthesis",
                "Final report already exists. Use --force to re-run.",
            )
            if conn is not None:
                with contextlib.suppress(Exception):
                    _db.insert_phase_metric(
                        conn,
                        run_id=run_id,
                        phase="synthesis",
                        status="skipped",
                    )
        else:
            progress.phase_start("synthesis")
            audit.log_phase_start("synthesis")
            syn_started = datetime.now(UTC)

            syn_metric_id = 0
            if conn is not None:
                with contextlib.suppress(Exception):
                    syn_metric_id = _db.insert_phase_metric(
                        conn,
                        run_id=run_id,
                        phase="synthesis",
                        status="running",
                        started_at=syn_started.isoformat(),
                    )

            synthesis_crew = build_synthesis_crew(
                plan=plan,
                llm=llm,
                research_dir=plan.research_dir,
                verification_dir=plan.verification_dir if plan.verify else None,
                verbose=verbose,
                conn=conn,
                run_id=run_id,
            )

            try:
                syn_result = synthesis_crew.kickoff()
            except Exception as e:
                progress.error("synthesis", str(e))
                audit.log_error("synthesis", "synthesis_crew", str(e))
                if conn is not None and syn_metric_id:
                    with contextlib.suppress(Exception):
                        _db.update_phase_metric(
                            conn,
                            syn_metric_id,
                            status="error",
                            ended_at=datetime.now(UTC).isoformat(),
                        )
                raise

            final_report = f"{plan.output_dir}/final-report.md"
            syn_ended = datetime.now(UTC)
            progress.phase_end("synthesis", final_report)
            audit.log_phase_end("synthesis", output_files=[final_report])

            _record_token_usage(conn, run_id, "synthesis", plan.model, syn_result)

            if conn is not None and syn_metric_id:
                with contextlib.suppress(Exception):
                    _db.update_phase_metric(
                        conn,
                        syn_metric_id,
                        status="done",
                        ended_at=syn_ended.isoformat(),
                        duration_seconds=(syn_ended - syn_started).total_seconds(),
                        output_files=[final_report],
                    )

    finally:
        # Always stop the live display to restore terminal state.
        progress.stop_live()

    # --- Update run status ---
    pipeline_end = datetime.now(UTC)
    total_duration = (pipeline_end - pipeline_start).total_seconds()
    if conn is not None:
        with contextlib.suppress(Exception):
            _db.update_run(
                conn,
                run_id,
                status="done",
                duration_seconds=round(total_duration, 1),
            )

    # --- Summary ---
    progress.pipeline_end()
    progress.summary(
        plan_topic=plan.topic,
        research_files=research_files,
        verification_report=verification_report,
        final_report=final_report,
        conn=conn,
        run_id=run_id,
    )

    # --- Run Manifest (still written for backward compat) ---
    _write_run_manifest(
        plan=plan,
        run_id=run_id,
        audit=audit,
        research_files=research_files,
        verification_report=verification_report,
        final_report=final_report,
        sources_summary=sources_summary,
        pipeline_start=progress.start_time,
    )

    # Close DB connection
    if conn is not None:
        with contextlib.suppress(Exception):
            conn.close()
