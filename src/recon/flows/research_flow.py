"""Research Flow - CrewAI Flow with 3-phase pipeline and state persistence.

This is the main orchestrator for Recon. It uses CrewAI's Flow API
with @start/@listen decorators and Pydantic state for tracking
progress across the investigation -> verification -> synthesis phases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, Field

from recon.config import ReconPlan  # noqa: TC001
from recon.crews.investigation.crew import build_investigation_crew
from recon.crews.synthesis.crew import build_synthesis_crew
from recon.crews.verification.crew import build_verification_crew
from recon.providers.llm import create_llm
from recon.providers.search import create_search_tools


class ResearchState(BaseModel):
    """State tracked across the research flow phases."""

    plan_topic: str = ""
    phase: str = "pending"  # pending, investigating, verifying, synthesizing, done, failed
    investigation_files: list[str] = Field(default_factory=list)
    verification_report: str = ""
    final_report: str = ""
    error: str = ""


class ResearchFlow(Flow[ResearchState]):
    """3-phase research pipeline as a CrewAI Flow.

    Phases:
    1. Investigation - Parallel researcher agents produce research/*.md files
    2. Verification - Fact-checker verifies claims in research files
    3. Synthesis - Director produces the final report

    Usage:
        flow = ResearchFlow(plan=plan, verbose=False)
        flow.kickoff()
    """

    def __init__(
        self,
        plan: ReconPlan,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.plan = plan
        self.verbose = verbose
        self._llm: Any = None
        self._search_tools: list[Any] = []

    def _ensure_dirs(self) -> None:
        """Create output directories."""
        Path(self.plan.research_dir).mkdir(parents=True, exist_ok=True)
        Path(self.plan.output_dir).mkdir(parents=True, exist_ok=True)
        if self.plan.verify:
            Path(self.plan.verification_dir).mkdir(parents=True, exist_ok=True)

    def _get_llm(self) -> Any:
        """Lazy-create and cache the LLM instance."""
        if self._llm is None:
            self._llm = create_llm(self.plan)
        return self._llm

    def _get_search_tools(self) -> list[Any]:
        """Lazy-create and cache search tools."""
        if not self._search_tools:
            self._search_tools = create_search_tools(self.plan)
        return self._search_tools

    @start()
    def investigate(self) -> str:
        """Phase 1: Run parallel investigation agents."""
        self._ensure_dirs()
        self.state.plan_topic = self.plan.topic
        self.state.phase = "investigating"

        investigations = self.plan.get_investigations()
        crew = build_investigation_crew(
            plan=self.plan,
            investigations=investigations,
            llm=self._get_llm(),
            search_tools=self._get_search_tools(),
            verbose=self.verbose,
        )

        result = crew.kickoff()

        # Track output files
        self.state.investigation_files = [
            f"{self.plan.research_dir}/{inv.id}-{inv.name.lower().replace(' ', '-')}.md"
            for inv in investigations
        ]

        return str(result)

    @listen(investigate)
    def verify(self, investigation_result: str) -> str:
        """Phase 2: Run fact-checking on research files."""
        if not self.plan.verify:
            self.state.phase = "synthesizing"
            return "verification_skipped"

        self.state.phase = "verifying"

        crew = build_verification_crew(
            plan=self.plan,
            llm=self._get_llm(),
            search_tools=self._get_search_tools(),
            research_dir=self.plan.research_dir,
            verbose=self.verbose,
        )

        if crew is None:
            return "no_research_files"

        result = crew.kickoff()
        self.state.verification_report = f"{self.plan.verification_dir}/report.md"

        return str(result)

    @listen(verify)
    def synthesize(self, verification_result: str) -> str:
        """Phase 3: Produce the final report."""
        self.state.phase = "synthesizing"

        crew = build_synthesis_crew(
            plan=self.plan,
            llm=self._get_llm(),
            research_dir=self.plan.research_dir,
            verification_dir=self.plan.verification_dir if self.plan.verify else None,
            verbose=self.verbose,
        )

        result = crew.kickoff()
        self.state.final_report = f"{self.plan.output_dir}/final-report.md"
        self.state.phase = "done"

        return str(result)
