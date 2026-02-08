"""Synthesis crew - director agent that produces the final report.

Reads all research documents and the verification report, then produces
a unified analysis with confidence-weighted findings.

Uses context strategy selection to handle inputs that may exceed model
context windows. When inputs are too large, content is truncated with
a warning. Full summarize/map_reduce strategies are planned for v0.2.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, Process, Task

from recon.config import ReconPlan  # noqa: TC001
from recon.context.strategy import (
    CHARS_PER_TOKEN,
    PROMPT_OVERHEAD,
    Strategy,
    choose_strategy,
    get_context_window,
)

logger = logging.getLogger(__name__)


def _truncate_to_window(text: str, model: str) -> str:
    """Truncate text to fit within 75% of the model's context window.

    Leaves room for system prompt, agent backstory, and output generation.

    Args:
        text: The combined input text.
        model: Model name for context window lookup.

    Returns:
        Truncated text with a notice appended, or original text if it fits.
    """
    window = get_context_window(model)
    # Reserve 25% for system prompt + backstory + output generation
    max_tokens = int(window * 0.75) - PROMPT_OVERHEAD
    max_chars = max_tokens * CHARS_PER_TOKEN

    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # Try to cut at a paragraph boundary
    last_break = truncated.rfind("\n\n")
    if last_break > max_chars * 0.8:
        truncated = truncated[:last_break]

    notice = (
        "\n\n---\n\n"
        "[NOTE: Input was truncated to fit the model's context window. "
        "Some research content may be missing from this synthesis. "
        "Consider using a model with a larger context window, or reduce "
        "the number of investigation angles (--depth quick).]"
    )
    return truncated + notice


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

    # Apply context window strategy
    strategy = choose_strategy(
        inputs=input_parts,
        model_context_window=get_context_window(plan.model),
        override=plan.context_strategy,
    )

    if strategy == Strategy.DIRECT:
        logger.debug("Context strategy: DIRECT (input fits within context window)")
    elif strategy in (Strategy.SUMMARIZE, Strategy.MAP_REDUCE):
        logger.warning(
            "Context strategy: %s -- input exceeds model context window. "
            "Truncating input to fit. Full %s support planned for v0.2.",
            strategy.value.upper(),
            strategy.value,
        )
        all_input = _truncate_to_window(all_input, plan.model)

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
