"""Verification crew - fact-checking agent that verifies research claims.

Reads all research files and produces a verification report with
claim-level status marks (VERIFIED, PARTIALLY_VERIFIED, UNVERIFIABLE,
CONTRADICTED).

Uses Recon's custom verification tools alongside search tools for
a structured fact-checking pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crewai import Agent, Crew, Process, Task

from recon.config import ReconPlan  # noqa: TC001
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

    Returns:
        A configured CrewAI Crew, or None if no research files found.
    """
    # Collect research files
    research_path = Path(research_dir)
    research_files = sorted(research_path.glob("*.md"))

    if not research_files:
        return None

    # Read all research content
    research_content_parts = []
    file_paths_text = []
    for f in research_files:
        content = f.read_text()
        research_content_parts.append(f"## Document: {f.name}\n\n{content}")
        file_paths_text.append(str(f))

    files_list = "\n".join(f"- {p}" for p in file_paths_text)

    # Custom verification tools
    verification_tools: list[Any] = [
        ClaimExtractorTool(),
        CitationVerifierTool(timeout=plan.verification.timeout_per_fetch),
        ConfidenceScorerTool(),
        SourceTrackerTool(output_dir=plan.verification_dir),
        ContradictionDetectorTool(),
    ]

    # Combine custom tools with search tools
    all_tools = verification_tools + search_tools

    # Build the verification agent
    verifier = Agent(
        role="Fact Checker",
        goal="Verify every factual claim in the research documents.",
        backstory=(
            "You are a fact-checking agent. Your job is to verify factual claims "
            "in research documents by cross-referencing them with independent sources.\n\n"
            "CORE RULES:\n"
            "1. Use the claim_extractor tool to identify verifiable claims "
            "from each research file.\n"
            "2. For each claim with a cited URL, use citation_verifier "
            "to check the source.\n"
            "3. For claims without sources, use web search to find "
            "confirming/contradicting evidence.\n"
            "4. Use confidence_scorer to assign a score to each verified claim.\n"
            "5. Use source_tracker to log every verification in the audit trail.\n"
            "6. Use contradiction_detector when the same fact appears in multiple documents.\n"
            "7. Mark each claim as: VERIFIED, PARTIALLY_VERIFIED, UNVERIFIABLE, or CONTRADICTED.\n"
            "8. For CONTRADICTED claims, provide the correct information with source.\n"
            "9. Do NOT verify opinions or subjective assessments.\n"
            f"10. Maximum {plan.verification.max_queries_per_claim} searches per claim."
        ),
        tools=all_tools,
        llm=llm,
        verbose=verbose,
    )

    output_file = str(Path(plan.verification_dir) / "report.md")

    # Task 1: Extract claims
    extract_task = Task(
        description=(
            "Extract all verifiable factual claims from the research documents.\n\n"
            f"Research files:\n{files_list}\n\n"
            "Use the claim_extractor tool on each file to get structured claims. "
            "Combine all claims into a single list with document attribution."
        ),
        expected_output=(
            "A structured list of all factual claims with IDs, types, "
            "source documents, and cited sources."
        ),
        agent=verifier,
    )

    # Task 2: Verify claims
    verify_task = Task(
        description=(
            "Verify each extracted claim:\n\n"
            "1. For claims with cited URLs: use citation_verifier to check the source.\n"
            "2. For claims without sources: search for the specific data point.\n"
            "3. Use contradiction_detector to compare claims about the same topic "
            "from different documents.\n"
            "4. Use confidence_scorer to assign a score to each claim.\n"
            "5. Use source_tracker to log every verification result.\n\n"
            f"Maximum {plan.verification.max_queries_per_claim} search queries per claim.\n"
            f"Maximum {plan.verification.max_fetches_per_claim} URL fetches per claim."
        ),
        expected_output=(
            "A claim-by-claim verification with status marks "
            "(VERIFIED/PARTIALLY_VERIFIED/UNVERIFIABLE/CONTRADICTED), "
            "evidence URLs, and confidence scores."
        ),
        agent=verifier,
        context=[extract_task],
    )

    # Task 3: Produce report
    report_task = Task(
        description=(
            "Produce the final verification report in markdown with:\n"
            "- Summary: count of verified/partially/unverifiable/contradicted claims\n"
            "- Reliability score per source document (percentage verified)\n"
            "- Table of all claims with: ID, claim text, status, confidence score, evidence URL\n"
            "- Detailed section for any CONTRADICTED claims with correct information\n"
            "- Overall confidence assessment\n"
        ),
        expected_output=(
            "A markdown verification report with claim-level status marks, "
            "source URLs, confidence scores, and reliability percentages."
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
