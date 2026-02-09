"""SemanticVerifierTool - LLM-based semantic verification of claims.

After :class:`CitationVerifierTool` performs keyword matching to check
whether a URL contains claim-relevant terms, this tool adds a semantic
layer: an LLM judges whether the **evidence actually supports the claim**.

This catches cases where keywords are present but the evidence contradicts
or is unrelated to the claim.  For example, a page mentioning "44K" and
"GitHub" might be about a different project.

The tool accepts a JSON object with ``claim``, ``evidence``, and
optionally ``url``, and returns a structured verdict.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from crewai.tools import BaseTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verdict schema
# ---------------------------------------------------------------------------

VERDICTS = ("SUPPORTS", "CONTRADICTS", "INSUFFICIENT", "UNRELATED")

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are a fact-checking judge. Given a CLAIM and EVIDENCE text, determine \
whether the evidence supports the claim.

CLAIM: {claim}

EVIDENCE (from {url}):
{evidence}

Return a JSON object with exactly these keys:
- "verdict": one of "SUPPORTS", "CONTRADICTS", "INSUFFICIENT", "UNRELATED"
- "confidence": a float 0.0-1.0 indicating your confidence in the verdict
- "reasoning": a single sentence explaining your verdict (max 50 words)

Definitions:
- SUPPORTS: the evidence clearly confirms the claim
- CONTRADICTS: the evidence directly contradicts the claim
- INSUFFICIENT: the evidence is relevant but does not fully confirm or deny
- UNRELATED: the evidence has no bearing on the claim

Return ONLY the JSON object. No markdown fences, no commentary.
"""


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def verify_semantically(
    claim: str,
    evidence: str,
    llm: Any,
    url: str = "unknown",
) -> dict[str, Any]:
    """Ask an LLM whether *evidence* supports *claim*.

    Args:
        claim: The factual claim to verify.
        evidence: Text excerpt from a source.
        llm: CrewAI LLM instance (has a ``call`` method).
        url: Source URL for context (included in prompt).

    Returns:
        Dict with keys ``verdict``, ``confidence``, ``reasoning``,
        ``claim``, ``url``.  On LLM failure returns verdict
        ``INSUFFICIENT`` with confidence 0.0.
    """
    if not evidence or not evidence.strip():
        return {
            "verdict": "INSUFFICIENT",
            "confidence": 0.0,
            "reasoning": "No evidence text provided.",
            "claim": claim,
            "url": url,
        }

    # Truncate evidence to avoid blowing context
    truncated = evidence[:3000]
    prompt = _JUDGE_PROMPT.format(
        claim=claim, evidence=truncated, url=url,
    )

    try:
        response = llm.call(
            messages=[{"role": "user", "content": prompt}],
        )
        text = response if isinstance(response, str) else str(response)
        text = text.strip()

        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        parsed: dict[str, Any] = json.loads(text)

        verdict = str(parsed.get("verdict", "INSUFFICIENT")).upper()
        if verdict not in VERDICTS:
            verdict = "INSUFFICIENT"

        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        reasoning = str(parsed.get("reasoning", ""))[:200]

        return {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning,
            "claim": claim,
            "url": url,
        }

    except Exception:
        logger.warning(
            "Semantic verification failed for claim: %s",
            claim[:80],
            exc_info=True,
        )
        return {
            "verdict": "INSUFFICIENT",
            "confidence": 0.0,
            "reasoning": "LLM verification call failed.",
            "claim": claim,
            "url": url,
        }


# ---------------------------------------------------------------------------
# CrewAI Tool wrapper
# ---------------------------------------------------------------------------


class SemanticVerifierTool(BaseTool):
    """Judge whether evidence semantically supports a claim.

    Requires an LLM to be set on the tool instance.  The verification
    crew injects its LLM at construction time.
    """

    name: str = "semantic_verifier"
    description: str = (
        "Judge whether evidence text actually supports a factual claim. "
        "Input: JSON with 'claim' and 'evidence' (and optionally 'url'). "
        "Returns a verdict: SUPPORTS / CONTRADICTS / INSUFFICIENT / "
        "UNRELATED, with confidence score and reasoning."
    )

    llm: Any = None  # Injected by verification crew

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, input_data: str) -> str:
        """Run semantic verification.

        Args:
            input_data: JSON string with ``claim``, ``evidence``,
                and optionally ``url``.

        Returns:
            JSON string with the verdict.
        """
        from recon.tools._helpers import parse_tool_input

        data, err = parse_tool_input(input_data)
        if err:
            return err

        assert data is not None

        if self.llm is None:
            return json.dumps({
                "verdict": "INSUFFICIENT",
                "confidence": 0.0,
                "reasoning": "No LLM configured for semantic verification.",
                "claim": data.get("claim", ""),
                "url": data.get("url", ""),
            })

        result = verify_semantically(
            claim=data.get("claim", ""),
            evidence=data.get("evidence", ""),
            llm=self.llm,
            url=data.get("url", "unknown"),
        )
        return json.dumps(result, indent=2)
