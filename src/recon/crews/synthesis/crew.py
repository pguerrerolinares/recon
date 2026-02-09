"""Synthesis crew - director agent that produces the final report.

Reads all research documents and the verification report, then produces
a unified analysis with confidence-weighted findings.

v0.3 enhancements:
- Structured claims JSON from verification DB injected as context
- Inline citation style ([1], [2]) in the final report
- memory=True with ONNX embedder for short-term memory
- Knowledge context from prior runs
"""

from __future__ import annotations

import json
import logging
import sqlite3  # noqa: TC003
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, Process, Task

from recon import db as _db
from recon.config import ReconPlan  # noqa: TC001
from recon.context.strategy import (
    CHARS_PER_TOKEN,
    PROMPT_OVERHEAD,
    Strategy,
    choose_strategy,
    get_context_window,
)
from recon.crews.investigation.crew import ONNX_EMBEDDER_CONFIG

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


def _build_claims_context(
    conn: sqlite3.Connection | None,
    run_id: str | None,
) -> str:
    """Build a structured claims summary from the knowledge DB.

    Returns a JSON-formatted claims summary block, or empty string.
    """
    if conn is None or run_id is None:
        return ""

    try:
        claims = _db.get_claims(conn, run_id=run_id, limit=100)
        if not claims:
            return ""

        summary: list[dict[str, Any]] = []
        for c in claims:
            summary.append({
                "id": c["id"],
                "text": c["text"],
                "status": c.get("verification_status", "unknown"),
                "confidence": c.get("confidence"),
                "source": c.get("cited_source", ""),
            })

        # Group by status for quick reference
        by_status: dict[str, int] = {}
        for c in summary:
            s = c["status"]
            by_status[s] = by_status.get(s, 0) + 1

        header = "VERIFIED CLAIMS DATA (structured, from verification phase):\n"
        stats = f"Status counts: {json.dumps(by_status)}\n"
        data = json.dumps(summary, indent=2)

        return f"\n\n{header}{stats}\n{data}"
    except Exception:
        return ""


def build_synthesis_crew(
    plan: ReconPlan,
    llm: Any,
    research_dir: str,
    verification_dir: str | None = None,
    verbose: bool = False,
    conn: sqlite3.Connection | None = None,
    run_id: str | None = None,
) -> Crew:
    """Build a synthesis crew that produces the final report.

    Args:
        plan: Validated ReconPlan.
        llm: CrewAI LLM instance.
        research_dir: Path to directory containing research markdown files.
        verification_dir: Path to directory with verification report, or None.
        verbose: Whether to enable verbose output.
        conn: Optional SQLite connection for reading claims from DB.
        run_id: Optional run identifier for DB queries.

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

    # Structured claims from DB (supplements the verification report)
    claims_context = _build_claims_context(conn, run_id)
    if claims_context:
        all_input += claims_context

    # Apply context window strategy
    strategy = choose_strategy(
        inputs=input_parts,
        model_context_window=get_context_window(plan.model),
        override=plan.context_strategy,
    )

    if strategy == Strategy.DIRECT:
        logger.debug("Context strategy: DIRECT (input fits within context window)")
    elif strategy == Strategy.TRUNCATE:
        logger.warning(
            "Context strategy: TRUNCATE -- input exceeds model context window. "
            "Truncating input to fit. Consider using a model with a larger "
            "context window or reducing investigation depth.",
        )
        all_input = _truncate_to_window(all_input, plan.model)

    # Build synthesis instructions
    backstory_parts = [
        "You are a synthesis agent. Read all research and verification "
        "data, then produce a unified analysis with actionable insights.",
        "",
        "CORE RULES:",
        "1. Read ALL input documents before writing.",
        "2. Use INLINE CITATIONS in Perplexity style: [1], [2], etc. "
        "Each number references a source listed in the Sources section "
        "at the end. Example: 'Revenue reached $1B [1] while costs '",
        "'remained stable [2].'",
        "3. When documents contradict, note both positions with citations.",
        "4. Prioritize claims marked VERIFIED in the verification report.",
        "5. Do NOT rely on CONTRADICTED claims unless noting the contradiction.",
        "6. UNVERIFIABLE claims: Do NOT include in the main body of "
        "the report. Move them to a separate 'Appendix: Unverified "
        "Claims' section at the end. The reader must trust that "
        "everything in the main report is backed by verified evidence.",
        "7. PARTIALLY_VERIFIED claims: Include in the main body but "
        "clearly mark as 'Partially Verified' with a brief explanation "
        "of what was and was not confirmed.",
        "8. Do not introduce new factual claims. Synthesize, do not research.",
        "9. Structure output for decision-making: takeaways, recommendations, actions.",
        "10. End the report with a numbered Sources section listing "
        "all cited URLs: '[1] https://... — Title or description'.",
    ]
    if plan.synthesis.instructions:
        backstory_parts.append(f"\nAdditional instructions:\n{plan.synthesis.instructions}")

    synthesizer = Agent(
        role="Research Director",
        goal=(
            "Produce a trustworthy final report with confidence-weighted "
            "findings and inline citations."
        ),
        backstory="\n".join(backstory_parts),
        llm=llm,
        verbose=verbose,
    )

    output_file = str(Path(plan.output_dir) / "final-report.md")

    synthesis_task = Task(
        description=(
            "Synthesize the following research and verification data into "
            f"a final report about '{plan.topic}':\n\n"
            f"{all_input}\n\n"
            "Produce a report with:\n"
            "- Executive summary with overall confidence assessment\n"
            "- Convergent findings (what all sources agree on)\n"
            "- Divergent findings (where sources disagree)\n"
            "- Detailed analysis by topic\n"
            "- Ranked recommendations\n"
            "- Confidence assessment for each major finding\n"
            "- Numbered Sources section at the end ([1], [2], etc.)\n\n"
            "CITATION FORMAT: Use inline numbered references like [1], [2]. "
            "Example: 'The market grew 15% [1] despite headwinds [2].'\n\n"
            "CRITICAL: Only include VERIFIED and PARTIALLY_VERIFIED "
            "claims in the main analysis sections. For PARTIALLY_VERIFIED "
            "claims, note what was and was not confirmed.\n\n"
            "At the end, add an 'Appendix: Unverified Claims' section "
            "listing all claims that could not be fact-checked. This "
            "appendix is for researcher follow-up, not for the reader to "
            "rely on."
        ),
        expected_output=(
            "A comprehensive markdown report with inline citations [1], [2], "
            "where the main body contains only verified findings, a numbered "
            "Sources section at the end, plus an appendix of unverified claims."
        ),
        agent=synthesizer,
        output_file=output_file,
    )

    return Crew(
        agents=[synthesizer],
        tasks=[synthesis_task],
        process=Process.sequential,
        verbose=verbose,
        memory=True,
        embedder=ONNX_EMBEDDER_CONFIG,  # type: ignore[arg-type]
    )
