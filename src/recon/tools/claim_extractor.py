"""ClaimExtractorTool - Extract verifiable factual claims from markdown.

Reads a markdown document and returns a structured list of factual claims
that can be verified (numbers, statistics, dates, company names, pricing,
funding amounts, user counts, GitHub stars, direct quotes).

This tool does NOT use an LLM -- it uses regex heuristics to identify
claim-like sentences and structures. This keeps cost at zero for extraction.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from crewai.tools import BaseTool


class Claim:
    """A single extractable factual claim."""

    def __init__(
        self,
        claim_id: str,
        text: str,
        source_document: str,
        claim_type: str,
        cited_source: str | None = None,
    ) -> None:
        self.claim_id = claim_id
        self.text = text
        self.source_document = source_document
        self.claim_type = claim_type
        self.cited_source = cited_source

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "source_document": self.source_document,
            "type": self.claim_type,
            "cited_source": self.cited_source,
        }


# Regex patterns for identifying factual claims in text.
# These are intentionally broad to avoid missing claims.
PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "statistic": [
        # Numbers with units: "44K stars", "$1.5B revenue", "128K context"
        re.compile(
            r"[A-Za-z\s]+\b(\d[\d,.]*[KMBTkmbt]?\+?)\s*(?:stars?|users?|downloads?|installs?|tokens?|parameters?)",
            re.IGNORECASE,
        ),
        # Percentages: "grew 45%", "28% market share"
        re.compile(r"\b\d[\d,.]*%", re.IGNORECASE),
        # Dollar amounts: "$1.5M", "$500K", "$10/month"
        re.compile(r"\$\d[\d,.]*[KMBTkmbt]?", re.IGNORECASE),
        # Numeric claims with context: "has 10 players", "reached 1B"
        re.compile(
            r"\b(?:has|have|had|reached|grew|raised|over|approximately"
            r"|around|about|nearly|more than|less than)"
            r"\s+\$?\d[\d,.]*[KMBTkmbt]?",
            re.IGNORECASE,
        ),
    ],
    "pricing": [
        # Price patterns: "$0.45/M", "$10/month", "free tier"
        re.compile(r"\$\d[\d,.]*\s*/\s*\w+", re.IGNORECASE),
        re.compile(r"\b(?:free tier|free plan|freemium|open.?source)", re.IGNORECASE),
        re.compile(
            r"\b\d[\d,.]*\s*(?:per|/)\s*(?:month|year|query|request|token|call|credit)",
            re.IGNORECASE,
        ),
    ],
    "date": [
        # Date patterns: "in 2025", "Q2 2026", "January 2025", "last 12 months"
        re.compile(r"\b(?:in|since|from|as of|by|during)\s+(?:Q[1-4]\s+)?\d{4}\b", re.IGNORECASE),
        re.compile(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:launched|released|founded|acquired|announced|published)\s+(?:in\s+)?\d{4}\b",
            re.IGNORECASE,
        ),
    ],
    "attribution": [
        # Company/product claims: "CrewAI has", "Google released", "acquired by"
        re.compile(
            r"\b(?:acquired|merged|partnered|funded|raised|launched|released)\s+by\s+",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:founded|created|developed|built|maintained)\s+by\s+", re.IGNORECASE),
    ],
    "quote": [
        # Direct quotes (anything in double quotes longer than 10 chars)
        re.compile(r'"[^"]{10,}"'),
    ],
}

# Pattern to detect URLs cited as sources.
URL_PATTERN = re.compile(r'https?://[^\s\])\'">,]+')

# Pattern to detect markdown source citations.
SOURCE_CITATION_PATTERN = re.compile(
    r"\[(?:Source|Ref|Citation|See)[:\s]*([^\]]+)\]", re.IGNORECASE
)


def _extract_sentence(text: str, match_start: int) -> str:
    """Extract the full sentence containing a regex match.

    Looks backward for sentence start (. or newline) and forward for sentence end.
    """
    # Find start of sentence
    start = match_start
    while start > 0 and text[start - 1] not in ".!?\n":
        start -= 1

    # Find end of sentence
    end = match_start
    while end < len(text) and text[end] not in ".!?\n":
        end += 1

    sentence = text[start : end + 1].strip()
    # Clean up markdown artifacts
    sentence = re.sub(r"^[\s*#|\->]+", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence


def _find_cited_source(text: str, match_start: int) -> str | None:
    """Find a URL or source citation near the match position."""
    # Search within 200 chars after the match for a URL
    window = text[match_start : match_start + 300]

    source_match = SOURCE_CITATION_PATTERN.search(window)
    if source_match:
        return source_match.group(1).strip()

    url_match = URL_PATTERN.search(window)
    if url_match:
        return url_match.group(0)

    return None


# Priority order for claim types: higher priority = verified first.
# Statistics and attributions are most valuable to verify; quotes least.
CLAIM_TYPE_PRIORITY: dict[str, int] = {
    "statistic": 0,
    "attribution": 1,
    "date": 2,
    "pricing": 3,
    "quote": 4,
}


def _prioritize_claims(claims: list[Claim], max_claims: int) -> list[Claim]:
    """Prioritize and cap the number of claims.

    Priority rules:
    1. Claims with a cited_source URL are always preferred (verifiable).
    2. Sort by claim type priority
       (statistics > attributions > dates > pricing > quotes).
    3. Return at most max_claims.

    Args:
        claims: Full list of extracted claims.
        max_claims: Maximum number of claims to return.

    Returns:
        Prioritized and capped list of claims, re-numbered.
    """
    if len(claims) <= max_claims:
        return claims

    def sort_key(c: Claim) -> tuple[int, int]:
        has_source = 0 if c.cited_source else 1  # 0 = has source (higher priority)
        type_rank = CLAIM_TYPE_PRIORITY.get(c.claim_type, 99)
        return (has_source, type_rank)

    sorted_claims = sorted(claims, key=sort_key)
    selected = sorted_claims[:max_claims]

    # Re-number claim IDs sequentially
    for i, claim in enumerate(selected, 1):
        claim.claim_id = f"C{i}"

    return selected


def extract_claims(document_path: str, max_claims: int = 0) -> list[Claim]:
    """Extract factual claims from a markdown file.

    Args:
        document_path: Path to the markdown file.
        max_claims: Maximum number of claims to return. 0 means no limit.

    Returns:
        List of Claim objects, prioritized and capped if max_claims > 0.
    """
    path = Path(document_path)
    if not path.exists():
        return []

    text = path.read_text()
    doc_name = path.name
    claims: list[Claim] = []
    seen_sentences: set[str] = set()
    claim_counter = 0

    for claim_type, patterns in PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                sentence = _extract_sentence(text, match.start())
                if not sentence or len(sentence) < 15:
                    continue

                # Deduplicate by sentence
                if sentence in seen_sentences:
                    continue
                seen_sentences.add(sentence)

                claim_counter += 1
                cited_source = _find_cited_source(text, match.start())

                claims.append(
                    Claim(
                        claim_id=f"C{claim_counter}",
                        text=sentence,
                        source_document=doc_name,
                        claim_type=claim_type,
                        cited_source=cited_source,
                    )
                )

    if max_claims > 0:
        claims = _prioritize_claims(claims, max_claims)

    return claims


class ClaimExtractorTool(BaseTool):
    """Extract verifiable factual claims from a markdown document.

    Returns a structured JSON list of claims with IDs, types, and cited sources.
    """

    name: str = "claim_extractor"
    description: str = (
        "Extract verifiable factual claims from a markdown document. "
        "Input: path to a markdown file. "
        "Returns a JSON list of claims with IDs, types, and cited sources."
    )

    max_claims: int = 0  # 0 = no limit; set by verification crew from config

    def _run(self, document_path: str) -> str:
        """Extract claims from a markdown document.

        Args:
            document_path: Path to the markdown file.

        Returns:
            JSON string with list of claim dicts.
        """
        claims = extract_claims(document_path, max_claims=self.max_claims)
        result = [c.to_dict() for c in claims]
        return json.dumps(result, indent=2)
