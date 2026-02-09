"""ClaimExtractorTool - Extract verifiable factual claims from markdown.

Reads a markdown document and returns a structured list of factual claims
that can be verified (numbers, statistics, dates, company names, pricing,
funding amounts, user counts, GitHub stars, direct quotes).

Two-stage extraction pipeline:
1. **Regex heuristics** + **pre-filters**: fast, zero-cost extraction that
   rejects markdown table fragments, bibliography entries, headers, and
   other structural noise.
2. **LLM batch filter** (optional): an LLM call that validates claim quality,
   rejects remaining garbage, and decomposes compound claims into atomic
   verifiable statements.  Falls back gracefully to regex-only results on
   failure.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from crewai.tools import BaseTool

logger = logging.getLogger(__name__)


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

# ---------------------------------------------------------------------------
# Pre-filter: reject markdown artifacts, table fragments, bibliography, etc.
# ---------------------------------------------------------------------------

# Matches markdown table cell fragments: text with leading/trailing pipes.
_TABLE_FRAGMENT_RE = re.compile(r"\|")

# Matches markdown bold key-value labels like "Type**:", "Features**:"
_BOLD_LABEL_RE = re.compile(r"\*\*\s*[:|\]]")

# Matches bibliography/source entries: "Author - \"Title\"**" or "1. **Title**"
_BIBLIOGRAPHY_RE = re.compile(
    r"(?:"
    r'^\d+\.\s+\*\*'  # numbered list bold entry "1. **Title**"
    r'|^-\s+URL:'  # "- URL: ..."
    r'|^-\s+Access Date:'  # "- Access Date: ..."
    r'|^-\s+Data:'  # "- Data: ..."
    r'|\*\*\s*\((?:Nov|Dec|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct)\b'  # "** (Nov 2024)"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

# Matches "Access Date:" standalone entries.
_ACCESS_DATE_RE = re.compile(r"^Access Date:", re.IGNORECASE)

# Incomplete sentences: end with "(e.", "(i.", have unbalanced parens, etc.
_INCOMPLETE_RE = re.compile(
    r"(?:"
    r"\(\s*[a-z]\.\s*$"  # ends with "(e." or "(i."
    r"|[^.!?]$"  # doesn't end with sentence-ending punctuation (after stripping)
    r")",
    re.IGNORECASE,
)

# Minimum word count for a claim to be considered substantive.
_MIN_WORD_COUNT = 4

# Maximum ratio of special characters (|, *, #) to total length.
_MAX_SPECIAL_CHAR_RATIO = 0.15


def _is_markdown_noise(text: str) -> bool:
    """Return True if *text* looks like a markdown structural fragment.

    Catches table rows, bold labels, section headers, and other non-prose
    artifacts that regex extraction incorrectly captures as claims.
    """
    # Count pipe characters — table fragments have many
    pipe_count = text.count("|")
    if pipe_count >= 2:
        return True

    # Bold key-value label: "Type**: ...", "Features**: ..."
    if _BOLD_LABEL_RE.search(text):
        return True

    # Check special-character density (**, ##, ||, etc.)
    special_count = sum(1 for c in text if c in "*#|")
    return len(text) > 0 and special_count / len(text) > _MAX_SPECIAL_CHAR_RATIO


def _is_bibliography_entry(text: str) -> bool:
    """Return True if *text* looks like a bibliography/source-list entry.

    Examples of entries that should be rejected:
    - ``Anthropic - "Introducing the Model Context Protocol"** (Nov 25, 2024)``
    - ``URL: https://www.anthropic.com/news/...``
    - ``Access Date: January 2026``
    """
    # Starts with "N. **" (numbered reference)
    if re.match(r"^\d+\.\s+\*\*", text):
        return True

    # Bibliography metadata lines
    if _ACCESS_DATE_RE.match(text.strip()):
        return True

    # "Author - \"Title\"**" pattern
    if re.search(r'[A-Z]\w+\s*-\s*"[^"]+"\*?\*?', text):
        return True

    # URL-only or URL-prefixed lines
    stripped = text.strip()
    return stripped.startswith("URL:") or stripped.startswith("http")


def _is_incomplete_sentence(text: str) -> bool:
    """Return True if *text* is a truncated or incomplete sentence."""
    stripped = text.rstrip()

    # Ends with an open parenthesis + letter + dot: "(e.", "(i."
    if re.search(r"\([a-z]\.\s*$", stripped, re.IGNORECASE):
        return True

    # Too few words to be a meaningful claim
    words = stripped.split()
    if len(words) < _MIN_WORD_COUNT:
        return True

    # Starts with a lowercase continuation word (mid-sentence fragment)
    if stripped and stripped[0].islower() and not stripped.startswith("http"):
        # Allow things like "open-source" but reject "md and Model Context Protocol..."
        first_word = words[0] if words else ""
        continuation_words = {
            "and", "or", "but", "nor", "yet", "so", "for",
            "the", "a", "an", "md", "to", "of", "in", "on",
        }
        if first_word.lower().rstrip(".,;:") in continuation_words:
            return True

    return False


def _passes_prefilter(text: str) -> bool:
    """Return True if *text* passes all pre-filter heuristic checks.

    A claim must pass ALL of these to survive:
    1. Not a markdown structural fragment (table cell, bold label, header)
    2. Not a bibliography/source-list entry
    3. Not an incomplete or truncated sentence
    """
    if _is_markdown_noise(text):
        return False
    if _is_bibliography_entry(text):
        return False
    return not _is_incomplete_sentence(text)


# ---------------------------------------------------------------------------
# Text pre-processing: strip markdown before extraction
# ---------------------------------------------------------------------------

# Sections to skip entirely (sources/references/bibliography).
_SECTION_SKIP_RE = re.compile(
    r"^#{1,3}\s+(?:Sources?|References?|Bibliography|Works Cited|Endnotes)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_source_sections(text: str) -> str:
    """Remove Sources/References sections from the end of the document.

    These sections contain bibliography entries that regex patterns falsely
    match as claims (dates, quotes, numbers in titles).
    """
    match = _SECTION_SKIP_RE.search(text)
    if match:
        return text[: match.start()]
    return text


# ---------------------------------------------------------------------------
# LLM batch filter
# ---------------------------------------------------------------------------

_LLM_FILTER_PROMPT = """\
You are a claim quality filter for a fact-checking pipeline. You receive a \
JSON list of candidate claims extracted by regex from a research document. \
Your job is to:

1. **REJECT** items that are NOT verifiable factual claims:
   - Markdown table fragments, headers, or formatting artifacts
   - Bibliography entries or source citations (author names, URLs, dates)
   - Opinions, subjective assessments, or vague statements
   - Incomplete sentences or sentence fragments

2. **KEEP** items that ARE verifiable factual claims:
   - Statements with specific numbers, statistics, dates, prices
   - Attribution claims ("X was founded by Y", "acquired by Z")
   - Direct quotes with attribution
   - Any statement that can be checked against an independent source

3. **DECOMPOSE** compound claims into atomic ones. If a single claim \
contains multiple independently verifiable facts, split it into separate \
claims. Each atomic claim should contain exactly ONE verifiable assertion.

Return a JSON array of objects. Each object must have:
- "text": the claim text (cleaned, no markdown artifacts)
- "type": one of "statistic", "attribution", "date", "pricing", "quote"
- "keep": true if this is a valid verifiable claim, false if it should be rejected
- "reason": brief explanation (max 10 words) for the keep/reject decision

IMPORTANT:
- Return ONLY the JSON array, no markdown fences, no commentary
- Preserve the original claim meaning when cleaning text
- When decomposing, mark the original as keep=false and add new atomic claims

INPUT CLAIMS:
{claims_json}
"""


def _llm_filter_claims(
    claims: list[Claim],
    llm: Any,
) -> list[Claim]:
    """Use an LLM to filter and decompose claims.

    Sends all candidate claims to the LLM in a single batch call.
    Returns only the claims the LLM marked as valid, plus any new
    atomic claims from decomposition.

    On any error (LLM failure, JSON parse error, etc.) returns the
    original claims unchanged as a graceful fallback.

    Args:
        claims: Pre-filtered regex-extracted claims.
        llm: A CrewAI LLM instance (has a ``call`` method).

    Returns:
        Filtered and potentially decomposed list of Claims.
    """
    if not claims:
        return claims

    # Build input for the LLM
    input_data = [
        {"id": c.claim_id, "text": c.text, "type": c.claim_type}
        for c in claims
    ]
    prompt = _LLM_FILTER_PROMPT.format(claims_json=json.dumps(input_data, indent=2))

    try:
        response = llm.call(messages=[{"role": "user", "content": prompt}])

        # Parse the response — handle potential markdown fences
        response_text = response if isinstance(response, str) else str(response)
        response_text = response_text.strip()

        # Strip markdown code fences if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = "\n".join(lines)

        parsed: list[dict[str, Any]] = json.loads(response_text)
        if not isinstance(parsed, list):
            logger.warning("LLM filter returned non-list, falling back to regex claims")
            return claims

    except Exception:
        logger.warning("LLM claim filter failed, falling back to regex-only claims", exc_info=True)
        return claims

    doc_name = claims[0].source_document if claims else "unknown"

    filtered: list[Claim] = []
    counter = 0

    for item in parsed:
        if not isinstance(item, dict):
            continue
        if not item.get("keep", False):
            continue

        text = item.get("text", "").strip()
        if not text or len(text) < 15:
            continue

        counter += 1
        claim_type = item.get("type", "statistic")
        # Validate claim_type
        if claim_type not in CLAIM_TYPE_PRIORITY:
            claim_type = "statistic"

        # Try to inherit cited_source from original claim if text matches
        cited_source = None
        for orig in claims:
            if orig.text in text or text in orig.text:
                cited_source = orig.cited_source
                break

        filtered.append(
            Claim(
                claim_id=f"C{counter}",
                text=text,
                source_document=doc_name,
                claim_type=claim_type,
                cited_source=cited_source,
            )
        )

    if not filtered:
        # LLM rejected everything — fall back to regex claims
        logger.warning("LLM filter rejected all claims, falling back to regex results")
        return claims

    return filtered


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


def extract_claims(
    document_path: str,
    max_claims: int = 0,
    llm: Any | None = None,
) -> list[Claim]:
    """Extract factual claims from a markdown file.

    Pipeline:
    1. Strip source/reference sections from the document
    2. Regex extraction to find candidate claim sentences
    3. Pre-filter heuristics to reject markdown noise
    4. (Optional) LLM batch filter to validate quality and decompose
    5. Prioritize and cap to ``max_claims``

    Args:
        document_path: Path to the markdown file.
        max_claims: Maximum number of claims to return. 0 means no limit.
        llm: Optional CrewAI LLM instance for the batch filter step.
             When ``None``, only regex + pre-filter heuristics are used.

    Returns:
        List of Claim objects, prioritized and capped if max_claims > 0.
    """
    path = Path(document_path)
    if not path.exists():
        return []

    raw_text = path.read_text()
    # Strip bibliography/sources sections before extraction
    text = _strip_source_sections(raw_text)
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

                # Pre-filter: reject markdown noise
                if not _passes_prefilter(sentence):
                    continue

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

    # LLM batch filter (optional, with graceful fallback)
    if llm is not None and claims:
        claims = _llm_filter_claims(claims, llm)

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
    llm: Any | None = None  # Optional LLM for batch quality filter

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, document_path: str) -> str:
        """Extract claims from a markdown document.

        Args:
            document_path: Path to the markdown file.

        Returns:
            JSON string with list of claim dicts.
        """
        claims = extract_claims(
            document_path, max_claims=self.max_claims, llm=self.llm
        )
        result = [c.to_dict() for c in claims]
        return json.dumps(result, indent=2)
