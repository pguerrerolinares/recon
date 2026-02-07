"""Investigation crew - creates parallel researcher agents.

This crew dynamically creates one agent + task per investigation angle
defined in the plan. All agents run with async_execution=True for
parallel research.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crewai import Agent, Crew, Process, Task

from recon.config import Investigation, ReconPlan  # noqa: TC001


def build_investigation_crew(
    plan: ReconPlan,
    investigations: list[Investigation],
    llm: Any,
    search_tools: list[Any],
    verbose: bool = False,
) -> Crew:
    """Build an investigation crew with one agent per investigation angle.

    Args:
        plan: Validated ReconPlan.
        investigations: List of investigation angles to research.
        llm: CrewAI LLM instance.
        search_tools: List of search tool instances.
        verbose: Whether to enable verbose output.

    Returns:
        A configured CrewAI Crew ready to kickoff.
    """
    agents: list[Agent] = []
    tasks: list[Task] = []

    for inv in investigations:
        # Build context strings for template interpolation
        focus_context = f"Focus area: {plan.focus}" if plan.focus else ""
        investigation_instructions = (
            f"Additional context: {inv.instructions}" if inv.instructions else ""
        )

        backstory = (
            "You are a research agent specialized in deep investigation. Your job is to "
            "research a specific topic thoroughly using web search and content extraction.\n\n"
            "CORE RULES:\n"
            "1. NEVER fabricate data, statistics, URLs, or quotes. Every factual claim must "
            "come from a source you actually retrieved.\n"
            "2. When you cannot verify information, mark it as [UNVERIFIED].\n"
            "3. Always cite sources with the actual URL you retrieved from.\n"
            "4. Prefer primary sources (official websites, GitHub repos, docs) over "
            "secondary sources (blog posts, news).\n"
            "5. Include access date for each source.\n"
            "6. If a search or fetch fails, note it and move on. Do not guess.\n"
            "7. Structure output as markdown with clear headers, tables, and a Sources "
            "section at the end.\n\n"
            f"The research topic is: {plan.topic}"
        )
        if focus_context:
            backstory += f"\n{focus_context}"
        if investigation_instructions:
            backstory += f"\n{investigation_instructions}"

        agent = Agent(
            role=f"{inv.name} Researcher",
            goal=f"Produce a comprehensive, well-sourced research document on: {inv.name}",
            backstory=backstory,
            tools=search_tools,
            llm=llm,
            verbose=verbose,
        )

        questions_text = "\n".join(f"- {q}" for q in inv.questions)
        output_file = str(
            Path(plan.research_dir) / f"{inv.id}-{inv.name.lower().replace(' ', '-')}.md"
        )

        task = Task(
            description=(
                f"Investigate the following questions about '{plan.topic}':\n\n"
                f"{questions_text}\n\n"
                "Produce a well-structured markdown document with findings, "
                "data tables where relevant, and a Sources section at the end. "
                "Mark any unverified claims as [UNVERIFIED]."
            ),
            expected_output=(
                "A comprehensive markdown research document with headers, tables, "
                "cited sources, and [UNVERIFIED] markers where applicable."
            ),
            agent=agent,
            output_file=output_file,
            async_execution=True,
        )

        agents.append(agent)
        tasks.append(task)

    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=verbose,
    )
