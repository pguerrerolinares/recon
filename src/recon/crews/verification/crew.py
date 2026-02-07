"""Verification crew - fact-checking agent that verifies research claims.

Reads all research files and produces a verification report with
claim-level status marks (VERIFIED, PARTIALLY_VERIFIED, UNVERIFIABLE,
CONTRADICTED).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crewai import Agent, Crew, Process, Task

from recon.config import ReconPlan


def build_verification_crew(
    plan: ReconPlan,
    llm: Any,
    search_tools: list[Any],
    research_dir: str,
    verbose: bool = False,
) -> Crew | None:
    """Build a verification crew that fact-checks research documents.

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
    for f in research_files:
        content = f.read_text()
        research_content_parts.append(f"## Document: {f.name}\n\n{content}")

    all_research = "\n\n---\n\n".join(research_content_parts)

    # Build the verification agent
    verifier = Agent(
        role="Fact Checker",
        goal="Verify every factual claim in the research documents.",
        backstory=(
            "You are a fact-checking agent. Your job is to verify factual claims "
            "in research documents by cross-referencing them with independent sources.\n\n"
            "CORE RULES:\n"
            "1. Identify claims with: numbers, statistics, dates, company names, "
            "pricing, funding amounts, user counts, or direct quotes.\n"
            "2. For each claim, search for the specific data point.\n"
            "3. Mark each claim as: VERIFIED, PARTIALLY_VERIFIED, UNVERIFIABLE, "
            "or CONTRADICTED.\n"
            "4. For CONTRADICTED claims, provide the correct information with source.\n"
            "5. Do NOT verify opinions or subjective assessments.\n"
            f"6. Maximum {plan.verification.max_queries_per_claim} searches per claim."
        ),
        tools=search_tools,
        llm=llm,
        verbose=verbose,
    )

    output_file = str(Path(plan.verification_dir) / "report.md")

    verification_task = Task(
        description=(
            "Verify the factual claims in the following research documents:\n\n"
            f"{all_research}\n\n"
            "Produce a verification report with:\n"
            "- Summary: count of verified/partially/unverifiable/contradicted claims\n"
            "- Reliability score per source document\n"
            "- Table of all claims with status and evidence URLs\n"
            "- Detailed section for any CONTRADICTED claims"
        ),
        expected_output=(
            "A markdown verification report with claim-level status marks "
            "(VERIFIED, PARTIALLY_VERIFIED, UNVERIFIABLE, CONTRADICTED), "
            "source URLs, and reliability percentages."
        ),
        agent=verifier,
        output_file=output_file,
    )

    return Crew(
        agents=[verifier],
        tasks=[verification_task],
        process=Process.sequential,
        verbose=verbose,
    )
