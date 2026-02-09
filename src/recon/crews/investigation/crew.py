"""Investigation crew - creates parallel researcher agents.

This crew dynamically creates one agent + task per investigation angle
defined in the plan. All agents run with async_execution=True for
parallel research.

v0.3 enhancements:
- ``memory=True`` with ONNX embedder for shared short-term memory
- ``Process.hierarchical`` with a manager agent for DEEP depth
- ``reasoning=True`` on agents for DEEP depth (chain-of-thought)
- ``allow_delegation=True`` for DEEP depth (manager can delegate)
- ``step_callback`` / ``task_callback`` hooks for observability
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from crewai import Agent, Crew, Process, Task

from recon.config import DEPTH_MAX_ITER, Depth, Investigation, ReconPlan  # noqa: TC001

if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ONNX embedder config (used by CrewAI's ShortTermMemory / ChromaDB)
# ---------------------------------------------------------------------------

ONNX_EMBEDDER_CONFIG: dict[str, Any] = {
    "provider": "onnx",
    "config": {},  # uses all-MiniLM-L6-v2 by default (~80MB, local)
}


# ---------------------------------------------------------------------------
# Sub-question generation
# ---------------------------------------------------------------------------

_QUESTION_PROMPT = """\
You are a research planner. Given a topic and a specific investigation angle, \
generate {n} specific, targeted search questions that will help a researcher \
find factual, verifiable information.

Topic: {topic}
Investigation angle: {angle_name}
Angle description: {angle_description}
{focus_line}
Seed questions for context:
{seed_questions}

Rules:
- Each question should be specific enough to produce a focused web search.
- Cover different aspects of the angle (data, players, trends, risks).
- Prefer questions that will yield quantitative answers (numbers, dates, prices).
- Do NOT repeat the seed questions.

Return ONLY the questions, one per line, no numbering, no bullet points.
"""


def generate_sub_questions(
    topic: str,
    angle_name: str,
    angle_description: str,
    seed_questions: list[str],
    llm: Any,
    focus: str | None = None,
    n: int = 5,
) -> list[str]:
    """Use a single LLM call to generate focused sub-questions for an angle.

    Falls back to *seed_questions* on any error so the pipeline is never
    blocked by a question-generation failure.

    Args:
        topic: Overall research topic.
        angle_name: Name of the investigation angle.
        angle_description: Description of the angle.
        seed_questions: Existing user-provided questions (context, not replaced).
        llm: CrewAI LLM instance (has ``.call()`` method).
        focus: Optional focus qualifier.
        n: Number of sub-questions to generate.

    Returns:
        List of generated question strings.
    """
    focus_line = f"Focus: {focus}" if focus else ""
    seeds_text = "\n".join(f"- {q}" for q in seed_questions) if seed_questions else "(none)"

    prompt = _QUESTION_PROMPT.format(
        n=n,
        topic=topic,
        angle_name=angle_name,
        angle_description=angle_description,
        focus_line=focus_line,
        seed_questions=seeds_text,
    )

    try:
        response = llm.call([prompt])
        # Parse response: one question per non-empty line
        lines = [line.strip().lstrip("-•0123456789.) ") for line in str(response).splitlines()]
        questions = [q for q in lines if q and len(q) > 10]
        if questions:
            logger.info("Generated %d sub-questions for angle '%s'", len(questions), angle_name)
            return questions
    except Exception:
        logger.warning(
            "Sub-question generation failed for angle '%s', using seed questions",
            angle_name,
            exc_info=True,
        )

    return list(seed_questions)


def _build_manager_agent(llm: Any, topic: str) -> Agent:
    """Build a manager agent for hierarchical process (DEEP depth).

    The manager coordinates the research agents, delegates tasks,
    resolves conflicts, and ensures comprehensive coverage.
    """
    return Agent(
        role="Research Director",
        goal=(
            f"Coordinate and direct the research team investigating '{topic}'. "
            "Ensure comprehensive coverage, avoid duplication, and synthesize "
            "findings when sub-topics overlap."
        ),
        backstory=(
            "You are a senior research director with expertise in managing "
            "multi-agent research teams. Your job is to:\n"
            "1. Review each researcher's plan and provide guidance\n"
            "2. Identify gaps in coverage across investigation angles\n"
            "3. Direct researchers to fill coverage gaps\n"
            "4. Resolve conflicting findings between researchers\n"
            "5. Ensure the team produces high-quality, well-sourced output"
        ),
        llm=llm,
        allow_delegation=True,
        verbose=False,
    )


def build_investigation_crew(
    plan: ReconPlan,
    investigations: list[Investigation],
    llm: Any,
    search_tools: list[Any],
    verbose: bool = False,
    prior_knowledge: str | None = None,
    step_callback: Any | None = None,
    task_callback: Any | None = None,
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
        step_callback: Optional callback invoked after each agent step.
        task_callback: Optional callback invoked after each task completes.

    Returns:
        A configured CrewAI Crew ready to kickoff.
    """
    agents: list[BaseAgent] = []
    tasks: list[Task] = []

    is_deep = plan.depth == Depth.DEEP
    use_hierarchical = is_deep and len(investigations) > 1

    # Auto-generate sub-questions per angle if enabled
    if plan.auto_questions:
        for inv in investigations:
            generated = generate_sub_questions(
                topic=plan.topic,
                angle_name=inv.name,
                angle_description=inv.instructions or inv.name,
                seed_questions=inv.questions,
                llm=llm,
                focus=plan.focus,
            )
            # Merge: keep original questions + add generated ones (no dupes)
            existing = set(q.lower() for q in inv.questions)
            for q in generated:
                if q.lower() not in existing:
                    inv.questions.append(q)
                    existing.add(q.lower())

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
            "SOURCE DIVERSITY RULES:\n"
            "9. Search for at least 3 DIFFERENT types of sources per topic:\n"
            "   - Official sources (company websites, government sites, official docs)\n"
            "   - Academic/research (papers, institutional reports, whitepapers)\n"
            "   - Community/developer (GitHub, Stack Overflow, Reddit, Hacker News)\n"
            "   - News/media (reputable news outlets, industry publications)\n"
            "10. Do NOT rely on a single source for any major finding. "
            "Corroborate key statistics from 2+ independent sources.\n"
            "11. Note the source type in brackets: [Official], [Academic], "
            "[Community], [News].\n\n"
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
            max_iter=DEPTH_MAX_ITER[plan.depth],
            # DEEP depth: enable chain-of-thought reasoning and delegation
            reasoning=is_deep,
            allow_delegation=use_hierarchical,
        )

        questions_text = "\n".join(f"- {q}" for q in inv.questions)
        output_file = inv.output_path(plan.research_dir)

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

    # --- Crew configuration ---
    crew_kwargs: dict[str, Any] = {
        "agents": agents,
        "tasks": tasks,
        "verbose": verbose,
        # Enable CrewAI's built-in short-term memory (ChromaDB + ONNX)
        "memory": True,
        "embedder": ONNX_EMBEDDER_CONFIG,
    }

    if use_hierarchical:
        crew_kwargs["process"] = Process.hierarchical
        crew_kwargs["manager_agent"] = _build_manager_agent(llm, plan.topic)
    else:
        crew_kwargs["process"] = Process.sequential

    if step_callback is not None:
        crew_kwargs["step_callback"] = step_callback
    if task_callback is not None:
        crew_kwargs["task_callback"] = task_callback

    return Crew(**crew_kwargs)
