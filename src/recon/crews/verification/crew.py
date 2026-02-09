"""Verification crew - fact-checking agent that verifies research claims.

Reads all research files and produces a verification report with
claim-level status marks (VERIFIED, PARTIALLY_VERIFIED, UNVERIFIABLE,
CONTRADICTED).

Uses Recon's custom verification tools alongside search tools for
a structured fact-checking pipeline.
"""

from __future__ import annotations

import sqlite3  # noqa: TC003
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, Process, Task

from recon.config import DEPTH_MAX_ITER, ReconPlan  # noqa: TC001
from recon.tools.citation_verifier import CitationVerifierTool
from recon.tools.claim_extractor import ClaimExtractorTool
from recon.tools.confidence_scorer import ConfidenceScorerTool
from recon.tools.contradiction_detector import ContradictionDetectorTool
from recon.tools.source_tracker import SourceTrackerTool


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
    2. Verify each claim using search + citation checking
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

    # Custom verification tools
    verification_tools: list[Any] = [
        ClaimExtractorTool(max_claims=max_claims, llm=llm),
        CitationVerifierTool(timeout=plan.verification.timeout_per_fetch),
        ConfidenceScorerTool(),
        SourceTrackerTool(output_dir=plan.verification_dir, conn=conn, run_id=run_id),
        ContradictionDetectorTool(),
    ]

    # Combine custom tools with search tools
    all_tools = verification_tools + search_tools

    # Build the verification agent with budget-aware instructions
    verifier = Agent(
        role="Fact Checker",
        goal=(
            f"Verify the top {max_claims} factual claims extracted from the research documents. "
            "Work efficiently within your budget -- prioritize claims with cited URLs."
        ),
        backstory=(
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
            "3. For claims WITHOUT a cited URL: use 1 web search query maximum. "
            "If the search does not confirm the claim, mark it UNVERIFIABLE and move on.\n"
            "4. IMPORTANT: If you have spent 3+ tool calls on a single claim without "
            "resolution, mark it UNVERIFIABLE and move to the next claim.\n"
            "5. Use confidence_scorer to assign a score to each verified claim.\n"
            "6. Use source_tracker to log every verification in the audit trail.\n"
            "7. Mark each claim as: VERIFIED, PARTIALLY_VERIFIED, UNVERIFIABLE, or CONTRADICTED.\n"
            "8. For CONTRADICTED claims, provide the correct information with source.\n"
            "9. Do NOT verify opinions or subjective assessments.\n"
            f"10. Maximum {plan.verification.max_queries_per_claim} searches per claim.\n"
            f"11. Flag claims with confidence score below {plan.verification.min_confidence} "
            "as LOW CONFIDENCE in the report.\n"
            + (
                "12. Flag claims that lack a primary source (official website, GitHub repo, "
                "official docs) -- require_primary_source is enabled.\n"
                if plan.verification.require_primary_source
                else "12. Primary source citations are recommended but not required.\n"
            )
            + "13. Do NOT use contradiction_detector during individual claim verification. "
            "It will be used in the final summary step to check cross-document consistency."
        ),
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

    # Task 2: Verify claims (no contradiction_detector here -- that's in Task 3)
    verify_task = Task(
        description=(
            "Verify each extracted claim efficiently:\n\n"
            "1. For claims WITH a cited URL: use citation_verifier. This is fast.\n"
            "2. For claims WITHOUT a URL: use 1 web search max. "
            "If not confirmed, mark UNVERIFIABLE.\n"
            "3. Use confidence_scorer to assign a score to each claim.\n"
            "4. Use source_tracker to log every verification result.\n"
            "5. If 3+ tool calls on one claim without resolution: mark UNVERIFIABLE, move on.\n\n"
            f"Maximum {plan.verification.max_queries_per_claim} search queries per claim.\n"
            f"Maximum {plan.verification.max_fetches_per_claim} URL fetches per claim.\n\n"
            "Do NOT use contradiction_detector here -- it will be used in the report step."
        ),
        expected_output=(
            "A claim-by-claim verification with status marks "
            "(VERIFIED/PARTIALLY_VERIFIED/UNVERIFIABLE/CONTRADICTED), "
            "evidence URLs, and confidence scores."
        ),
        agent=verifier,
        context=[extract_task],
    )

    # Task 3: Cross-document consistency + final report
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
    )

    return Crew(
        agents=[verifier],
        tasks=[extract_task, verify_task, report_task],
        process=Process.sequential,
        verbose=verbose,
    )
