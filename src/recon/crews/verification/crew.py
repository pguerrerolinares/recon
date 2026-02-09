"""Verification crew - fact-checking agent that verifies research claims.

Reads all research files and produces a verification report with
claim-level status marks (VERIFIED, PARTIALLY_VERIFIED, UNVERIFIABLE,
CONTRADICTED).

Uses Recon's custom verification tools alongside search tools for
a structured fact-checking pipeline.

v0.3 enhancements:
- SemanticVerifierTool for LLM-based semantic evidence judgment
- Guardrail on report task to validate output structure
- Prior verified claims from knowledge DB injected as context
- Stale claims from previous runs flagged for re-verification
- memory=True with ONNX embedder for short-term memory
"""

from __future__ import annotations

import contextlib
import logging
import re
import sqlite3  # noqa: TC003
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, Process, Task

from recon import db as _db
from recon.config import DEPTH_MAX_ITER, ReconPlan  # noqa: TC001
from recon.crews.investigation.crew import ONNX_EMBEDDER_CONFIG
from recon.tools.citation_verifier import CitationVerifierTool
from recon.tools.claim_extractor import ClaimExtractorTool
from recon.tools.confidence_scorer import ConfidenceScorerTool
from recon.tools.contradiction_detector import ContradictionDetectorTool
from recon.tools.semantic_verifier import SemanticVerifierTool
from recon.tools.source_tracker import SourceTrackerTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guardrail: validate verification report structure
# ---------------------------------------------------------------------------


def _report_guardrail(result: Any) -> tuple[bool, Any]:
    """Validate that the verification report contains required sections.

    Returns (True, result) if valid, (False, error_message) if the agent
    should retry.  This is a lightweight structural check — it does NOT
    re-verify claims.
    """
    text = str(result)

    # Must have at least a summary section and some claim references
    has_summary = bool(re.search(r"(?i)(summary|overview|results)", text))
    has_claims = bool(re.search(r"(VERIFIED|PARTIALLY_VERIFIED|UNVERIFIABLE|CONTRADICTED)", text))
    has_table = "|" in text  # Markdown table indicator

    if has_summary and has_claims:
        return (True, result)

    missing: list[str] = []
    if not has_summary:
        missing.append("summary section")
    if not has_claims:
        missing.append("claim status marks (VERIFIED/PARTIALLY_VERIFIED/etc.)")
    if not has_table:
        missing.append("claims table")

    return (
        False,
        f"Report is missing required elements: {', '.join(missing)}. "
        "Please add them and regenerate the report.",
    )


# ---------------------------------------------------------------------------
# Prior knowledge helpers
# ---------------------------------------------------------------------------


def _get_prior_claims_context(
    conn: sqlite3.Connection | None,
    topic: str,
) -> str:
    """Query the knowledge DB for prior verified claims related to the topic.

    Returns a formatted string block or empty string.
    """
    if conn is None:
        return ""

    try:
        keywords = topic.split()[:6]
        query = " OR ".join(keywords)
        results = _db.search_claims_fts(conn, query, limit=15)
        if not results:
            return ""

        lines: list[str] = []
        for claim in results:
            status = claim.get("verification_status", "?")
            conf = claim.get("confidence")
            conf_str = f" ({conf:.0%})" if conf else ""
            lines.append(f"- [{status}]{conf_str} {claim['text']}")

        return (
            "\n\nPRIOR VERIFIED CLAIMS (from previous runs):\n"
            + "\n".join(lines)
            + "\n\nUse these as context — if a current claim matches a prior "
            "VERIFIED claim, you can increase confidence. If it contradicts, "
            "investigate more carefully."
        )
    except Exception:
        return ""


def _get_stale_claims_context(
    conn: sqlite3.Connection | None,
    topic: str,
    stale_after_days: int = 30,
) -> str:
    """Query for stale claims that should be re-verified.

    Returns a formatted string block or empty string.
    """
    if conn is None:
        return ""

    try:
        stale = _db.get_stale_claims(conn, older_than_days=stale_after_days, topic=topic, limit=10)
        if not stale:
            return ""

        lines: list[str] = []
        for claim in stale:
            last = claim.get("last_verified_at", "unknown")
            lines.append(f"- {claim['text']} (last verified: {last})")

        return (
            "\n\nSTALE CLAIMS (need re-verification):\n"
            + "\n".join(lines)
            + "\n\nThese claims have not been verified recently. "
            "If you encounter them, please re-verify with fresh searches."
        )
    except Exception:
        return ""


def build_verification_crew(
    plan: ReconPlan,
    llm: Any,
    search_tools: list[Any],
    research_dir: str,
    verbose: bool = False,
    conn: sqlite3.Connection | None = None,
    run_id: str | None = None,
) -> Crew | None:
    """Build a verification crew that fact-checks research documents.

    The crew uses a 3-task pipeline:
    1. Extract claims from research documents
    2. Verify each claim using search + citation + semantic checking
    3. Produce a verification report with confidence scores

    Args:
        plan: Validated ReconPlan.
        llm: CrewAI LLM instance.
        search_tools: List of search tool instances.
        research_dir: Path to directory containing research markdown files.
        verbose: Whether to enable verbose output.
        conn: Optional SQLite connection for DB writes (claim tracking).
        run_id: Optional run identifier for DB writes.

    Returns:
        A configured CrewAI Crew, or None if no research files found.
    """
    # Collect research file paths (content is read by tools, not inlined)
    research_path = Path(research_dir)
    research_files = sorted(research_path.glob("*.md"))

    if not research_files:
        return None

    file_paths_text = [str(f) for f in research_files]
    files_list = "\n".join(f"- {p}" for p in file_paths_text)

    max_claims = plan.verification.max_claims

    # Custom verification tools (now including SemanticVerifier)
    verification_tools: list[Any] = [
        ClaimExtractorTool(max_claims=max_claims, llm=llm),
        CitationVerifierTool(timeout=plan.verification.timeout_per_fetch),
        SemanticVerifierTool(llm=llm),
        ConfidenceScorerTool(),
        SourceTrackerTool(output_dir=plan.verification_dir, conn=conn, run_id=run_id),
        ContradictionDetectorTool(),
    ]

    # Combine custom tools with search tools
    all_tools = verification_tools + search_tools

    # Build knowledge context from prior runs
    prior_claims_ctx = _get_prior_claims_context(conn, plan.topic)
    stale_claims_ctx = _get_stale_claims_context(conn, plan.topic, plan.knowledge.stale_after_days)

    # Build the verification agent with budget-aware instructions
    backstory = (
        "You are a fact-checking agent. Your job is to verify factual claims "
        "in research documents by cross-referencing them with independent sources.\n\n"
        f"BUDGET: You have a maximum of {max_claims} claims to verify. "
        "The claim_extractor tool already prioritizes and caps claims for you.\n\n"
        "EFFICIENCY RULES:\n"
        "1. Use the claim_extractor tool on each research file. It returns "
        f"at most {max_claims} prioritized claims (statistics first, then attributions, "
        "dates, pricing, quotes -- claims with cited URLs are preferred).\n"
        "2. For claims WITH a cited URL: use citation_verifier to check the source. "
        "This is fast and does not use LLM calls.\n"
        "3. After citation_verifier, use semantic_verifier to judge whether the "
        "evidence actually supports the claim (not just keyword matching).\n"
        "4. For claims WITHOUT a cited URL: use 1 web search query maximum. "
        "If the search does not confirm the claim, mark it UNVERIFIABLE and move on.\n"
        "5. IMPORTANT: If you have spent 3+ tool calls on a single claim without "
        "resolution, mark it UNVERIFIABLE and move to the next claim.\n"
        "6. Use confidence_scorer to assign a score to each verified claim.\n"
        "7. Use source_tracker to log every verification in the audit trail.\n"
        "8. Mark each claim as: VERIFIED, PARTIALLY_VERIFIED, UNVERIFIABLE, or CONTRADICTED.\n"
        "9. For CONTRADICTED claims, provide the correct information with source.\n"
        "10. Do NOT verify opinions or subjective assessments.\n"
        f"11. Maximum {plan.verification.max_queries_per_claim} searches per claim.\n"
        f"12. Flag claims with confidence score below {plan.verification.min_confidence} "
        "as LOW CONFIDENCE in the report.\n"
        + (
            "13. Flag claims that lack a primary source (official website, GitHub repo, "
            "official docs) -- require_primary_source is enabled.\n"
            if plan.verification.require_primary_source
            else "13. Primary source citations are recommended but not required.\n"
        )
        + "14. Do NOT use contradiction_detector during individual claim verification. "
        "It will be used in the final summary step to check cross-document consistency."
        + prior_claims_ctx
        + stale_claims_ctx
    )

    verifier = Agent(
        role="Fact Checker",
        goal=(
            f"Verify the top {max_claims} factual claims extracted from the research documents. "
            "Work efficiently within your budget -- prioritize claims with cited URLs."
        ),
        backstory=backstory,
        tools=all_tools,
        llm=llm,
        verbose=verbose,
        max_iter=DEPTH_MAX_ITER[plan.depth],
    )

    output_file = str(Path(plan.verification_dir) / "report.md")

    # Task 1: Extract claims
    extract_task = Task(
        description=(
            "Extract verifiable factual claims from the research documents.\n\n"
            f"Research files:\n{files_list}\n\n"
            "Use the claim_extractor tool on each file to get structured claims. "
            f"The tool is configured to return at most {max_claims} prioritized claims per file "
            "(statistics and claims with cited URLs are prioritized). "
            "Combine all claims into a single list with document attribution."
        ),
        expected_output=(
            "A structured list of factual claims with IDs, types, "
            "source documents, and cited sources."
        ),
        agent=verifier,
    )

    # Task 2: Verify claims (including semantic verification)
    verify_task = Task(
        description=(
            "Verify each extracted claim efficiently:\n\n"
            "1. For claims WITH a cited URL: use citation_verifier first. This is fast.\n"
            "2. After citation_verifier, use semantic_verifier to judge whether the "
            "evidence semantically supports the claim (not just keyword presence).\n"
            "3. For claims WITHOUT a URL: use 1 web search max. "
            "If not confirmed, mark UNVERIFIABLE.\n"
            "4. Use confidence_scorer to assign a score to each claim.\n"
            "5. Use source_tracker to log every verification result.\n"
            "6. If 3+ tool calls on one claim without resolution: mark UNVERIFIABLE, move on.\n\n"
            f"Maximum {plan.verification.max_queries_per_claim} search queries per claim.\n"
            f"Maximum {plan.verification.max_fetches_per_claim} URL fetches per claim.\n\n"
            "Do NOT use contradiction_detector here -- it will be used in the report step."
        ),
        expected_output=(
            "A claim-by-claim verification with status marks "
            "(VERIFIED/PARTIALLY_VERIFIED/UNVERIFIABLE/CONTRADICTED), "
            "evidence URLs, confidence scores, and semantic verification verdicts."
        ),
        agent=verifier,
        context=[extract_task],
    )

    # Task 3: Cross-document consistency + final report (with guardrail)
    report_task = Task(
        description=(
            "First, use contradiction_detector to check claims about the same topic "
            "across different documents for cross-document consistency.\n\n"
            "Then produce the final verification report in markdown with:\n"
            "- Summary: count of verified/partially/unverifiable/contradicted claims\n"
            "- Reliability score per source document (percentage verified)\n"
            "- Table of all claims with: ID, claim text, status, confidence score, evidence URL\n"
            "- Cross-document consistency section (contradictions found between documents)\n"
            "- Detailed section for any CONTRADICTED claims with correct information\n"
            "- Overall confidence assessment\n"
        ),
        expected_output=(
            "A markdown verification report with claim-level status marks, "
            "source URLs, confidence scores, reliability percentages, "
            "and cross-document consistency analysis."
        ),
        agent=verifier,
        context=[verify_task],
        output_file=output_file,
        guardrail=_report_guardrail,
    )

    # Persist stale claims context as an event for traceability
    if conn is not None and run_id and stale_claims_ctx:
        with contextlib.suppress(Exception):
            _db.insert_event(
                conn,
                run_id=run_id,
                timestamp=__import__("datetime")
                .datetime.now(__import__("datetime").UTC)
                .isoformat(),
                action="stale_claims_loaded",
                phase="verification",
                agent="pipeline",
                detail=stale_claims_ctx[:2000],
            )

    return Crew(
        agents=[verifier],
        tasks=[extract_task, verify_task, report_task],
        process=Process.sequential,
        verbose=verbose,
        memory=True,
        embedder=ONNX_EMBEDDER_CONFIG,  # type: ignore[arg-type]
    )
