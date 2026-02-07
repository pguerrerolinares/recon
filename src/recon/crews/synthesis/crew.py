"""Synthesis crew - director agent that produces the final report.

Reads all research documents and the verification report, then produces
a unified analysis with confidence-weighted findings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crewai import Agent, Crew, Process, Task

from recon.config import ReconPlan


def build_synthesis_crew(
    plan: ReconPlan,
    llm: Any,
    research_dir: str,
    verification_dir: str | None = None,
    verbose: bool = False,
) -> Crew:
    """Build a synthesis crew that produces the final report.

    Args:
        plan: Validated ReconPlan.
        llm: CrewAI LLM instance.
        research_dir: Path to directory containing research markdown files.
        verification_dir: Path to directory with verification report, or None.
        verbose: Whether to enable verbose output.

    Returns:
        A configured CrewAI Crew ready to kickoff.
    """
    # Collect all input content
    input_parts: list[str] = []

    # Research files
    research_path = Path(research_dir)
    for f in sorted(research_path.glob("*.md")):
        content = f.read_text()
        input_parts.append(f"## Research: {f.name}\n\n{content}")

    # Verification report
    if verification_dir:
        ver_path = Path(verification_dir) / "report.md"
        if ver_path.exists():
            ver_content = ver_path.read_text()
            input_parts.append(f"## Verification Report\n\n{ver_content}")

    all_input = "\n\n---\n\n".join(input_parts)

    # Build synthesis instructions
    backstory_parts = [
        "You are a synthesis agent. Read all research and verification data, "
        "then produce a unified analysis with actionable insights.",
        "",
        "CORE RULES:",
        "1. Read ALL input documents before writing.",
        "2. Cite which document supports each claim using [Source: filename].",
        "3. When documents contradict, note both positions.",
        "4. Prioritize claims marked VERIFIED in the verification report.",
        "5. Do NOT rely on CONTRADICTED claims unless noting the contradiction.",
        "6. Treat UNVERIFIABLE claims with lower confidence.",
        "7. Do not introduce new factual claims. Synthesize, do not research.",
        "8. Structure output for decision-making: takeaways, recommendations, actions.",
    ]
    if plan.synthesis.instructions:
        backstory_parts.append(f"\nAdditional instructions:\n{plan.synthesis.instructions}")

    synthesizer = Agent(
        role="Research Director",
        goal="Produce a trustworthy final report with confidence-weighted findings.",
        backstory="\n".join(backstory_parts),
        llm=llm,
        verbose=verbose,
    )

    output_file = str(Path(plan.output_dir) / "final-report.md")

    synthesis_task = Task(
        description=(
            "Synthesize the following research and verification data into a "
            f"final report about '{plan.topic}':\n\n"
            f"{all_input}\n\n"
            "Produce a report with:\n"
            "- Executive summary\n"
            "- Convergent findings (what all sources agree on)\n"
            "- Divergent findings (where sources disagree)\n"
            "- Detailed analysis by topic\n"
            "- Ranked recommendations\n"
            "- Confidence assessment for each major finding"
        ),
        expected_output=(
            "A comprehensive markdown report with executive summary, "
            "analysis, recommendations, and confidence levels for each finding."
        ),
        agent=synthesizer,
        output_file=output_file,
    )

    return Crew(
        agents=[synthesizer],
        tasks=[synthesis_task],
        process=Process.sequential,
        verbose=verbose,
    )
