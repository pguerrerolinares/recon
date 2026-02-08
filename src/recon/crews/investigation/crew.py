"""Investigation crew - creates parallel researcher agents.

This crew dynamically creates one agent + task per investigation angle
defined in the plan. All agents run with async_execution=True for
parallel research.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from crewai import Agent, Crew, Process, Task

from recon.config import Investigation, ReconPlan  # noqa: TC001

if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent


def build_investigation_crew(
    plan: ReconPlan,
    investigations: list[Investigation],
    llm: Any,
    search_tools: list[Any],
    verbose: bool = False,
    prior_knowledge: str | None = None,
) -> Crew:
    """Build an investigation crew with one agent per investigation angle.

    Args:
        plan: Validated ReconPlan.
        investigations: List of investigation angles to research.
        llm: CrewAI LLM instance.
        search_tools: List of search tool instances.
        verbose: Whether to enable verbose output.
        prior_knowledge: Optional prior research findings from memory to inject
            into agent backstories as starting context.

    Returns:
        A configured CrewAI Crew ready to kickoff.
    """
    agents: list[BaseAgent] = []
    tasks: list[Task] = []

    last_index = len(investigations) - 1
    for i, inv in enumerate(investigations):
        # Build context strings for template interpolation
        focus_context = f"Focus area: {plan.focus}" if plan.focus else ""
        investigation_instructions = (
            f"Additional context: {inv.instructions}" if inv.instructions else ""
        )

        backstory = (
            "You are a research agent specialized in deep investigation. "
            "Your job is to research a specific topic thoroughly using web "
            "search and content extraction.\n\n"
            "CORE RULES:\n"
            "1. NEVER fabricate data, statistics, URLs, or quotes. Every "
            "factual claim must come from a source you actually retrieved.\n"
            "2. EVERY factual claim (statistic, study result, date, pricing, "
            "attribution) MUST include a clickable URL inline. For example: "
            "'Revenue reached $1B [Source: https://example.com/report]'.\n"
            "3. Do NOT cite academic papers by author name alone (e.g., "
            "'Smith et al. 2023'). Always search for the DOI link "
            "(https://doi.org/...) or the publisher URL and include it.\n"
            "4. If you find a fact but CANNOT find its URL after searching, "
            "mark it as [UNVERIFIED: source URL not found] and still include "
            "it, but make clear the URL is missing. Claims without URLs will "
            "be flagged as unverifiable during fact-checking.\n"
            "5. Prefer primary sources (official websites, GitHub repos, "
            "government sites, peer-reviewed journals) over secondary sources "
            "(blog posts, news aggregators).\n"
            "6. Include access date for each source.\n"
            "7. If a search or fetch fails, note it and move on. Do not "
            "guess.\n"
            "8. Structure output as markdown with clear headers, tables, and "
            "a Sources section at the end. Every entry in the Sources section "
            "MUST have a URL -- entries without URLs are not valid sources.\n\n"
            f"The research topic is: {plan.topic}"
        )
        if focus_context:
            backstory += f"\n{focus_context}"
        if investigation_instructions:
            backstory += f"\n{investigation_instructions}"
        if prior_knowledge:
            backstory += (
                "\n\nYou have access to findings from previous research runs. "
                "Use them as starting context but always verify independently "
                "with fresh searches:\n\n"
                f"{prior_knowledge}"
            )

        agent = Agent(
            role=f"{inv.name} Researcher",
            goal=f"Produce a comprehensive, well-sourced research document on: {inv.name}",
            backstory=backstory,
            tools=search_tools,
            llm=llm,
            verbose=verbose,
            max_iter=25,
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
                "data tables where relevant, and a Sources section at the "
                "end.\n\n"
                "IMPORTANT: For every statistic, study finding, or factual "
                "claim, include the source URL inline (e.g., "
                "'32% of teen girls reported... "
                "[Source: https://example.com/study]'). "
                "Claims without URLs will be flagged as unverifiable during "
                "fact-checking and excluded from the final report.\n\n"
                "If you cite an academic paper, search for its DOI or "
                "publisher URL. Do not cite by author name alone.\n\n"
                "Mark any claims where you could not find a URL as "
                "[UNVERIFIED: source URL not found]."
            ),
            expected_output=(
                "A comprehensive markdown research document with headers, "
                "tables, inline source URLs for every factual claim, a "
                "Sources section with URLs, and [UNVERIFIED] markers where "
                "a URL could not be found."
            ),
            agent=agent,
            output_file=output_file,
            # CrewAI requires the last task to be synchronous
            async_execution=(i < last_index),
        )

        agents.append(agent)
        tasks.append(task)

    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=verbose,
    )
