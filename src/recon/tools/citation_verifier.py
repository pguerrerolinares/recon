"""CitationVerifierTool - Verify that a cited URL contains the claimed information.

Fetches the URL, extracts text content, and checks whether the claimed
information is present. Returns VERIFIED/CONTRADICTED/UNVERIFIABLE with
an evidence excerpt.

Uses httpx for fetching to avoid adding extra dependencies.
Per-domain rate limiting avoids hitting the same host too frequently
while allowing different domains to be fetched without delay.
"""

from __future__ import annotations

import json
import re
import time
from urllib.parse import urlparse

import httpx
from crewai.tools import BaseTool
from pydantic import Field

from recon import __version__

# Maximum response size to process (512KB).
MAX_RESPONSE_SIZE = 512 * 1024

# Request timeout in seconds.
REQUEST_TIMEOUT = 15.0

# Minimum delay between fetches to the same domain (seconds).
DEFAULT_DOMAIN_DELAY = 1.0


def _normalize_text(text: str) -> str:
    """Normalize text for comparison by lowercasing and collapsing whitespace."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s$%./-]", "", text)
    return text.strip()


def _extract_key_terms(claim: str) -> list[str]:
    """Extract key terms from a claim for matching.

    Focuses on numbers, proper nouns, and significant terms.
    """
    terms: list[str] = []

    # Extract numbers (including with K/M/B suffixes)
    numbers = re.findall(r"\$?\d[\d,.]*[KMBTkmbt%]?", claim)
    terms.extend(numbers)

    # Extract capitalized terms (likely proper nouns)
    proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", claim)
    terms.extend(proper_nouns)

    # Extract quoted terms
    quoted = re.findall(r'"([^"]+)"', claim)
    terms.extend(quoted)

    return terms


def _fetch_url_text(url: str, timeout: float = REQUEST_TIMEOUT) -> str | None:
    """Fetch a URL and return its text content.

    Args:
        url: URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        Extracted text content, or None if fetch failed.
    """
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": f"Recon/{__version__} (research verification tool)"},
        ) as client:
            response = client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "text" not in content_type and "json" not in content_type:
                return None

            text = response.text[:MAX_RESPONSE_SIZE]

            # Strip HTML tags for basic text extraction
            text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)

            return text.strip()

    except (httpx.HTTPError, httpx.InvalidURL, ValueError):
        return None


def verify_citation(url: str, claim: str, timeout: float = REQUEST_TIMEOUT) -> dict:
    """Verify that a URL contains information supporting a claim.

    Args:
        url: URL to check.
        claim: The factual claim to verify.
        timeout: Request timeout in seconds.

    Returns:
        Dict with keys: status, evidence, url.
        status is one of: VERIFIED, PARTIALLY_VERIFIED, UNVERIFIABLE, ERROR.
    """
    text = _fetch_url_text(url, timeout=timeout)

    if text is None:
        return {
            "status": "ERROR",
            "evidence": f"Could not fetch or parse URL: {url}",
            "url": url,
        }

    normalized_text = _normalize_text(text)
    key_terms = _extract_key_terms(claim)

    if not key_terms:
        return {
            "status": "UNVERIFIABLE",
            "evidence": "No key terms could be extracted from claim.",
            "url": url,
        }

    # Count how many key terms appear in the page
    found_terms: list[str] = []
    missing_terms: list[str] = []

    for term in key_terms:
        normalized_term = _normalize_text(term)
        if normalized_term and normalized_term in normalized_text:
            found_terms.append(term)
        else:
            missing_terms.append(term)

    total = len(key_terms)
    found = len(found_terms)

    # Find an evidence excerpt containing the first found term
    evidence_excerpt = ""
    if found_terms:
        first_term = _normalize_text(found_terms[0])
        idx = normalized_text.find(first_term)
        if idx >= 0:
            start = max(0, idx - 100)
            end = min(len(text), idx + 200)
            evidence_excerpt = text[start:end].strip()

    if found == 0:
        return {
            "status": "UNVERIFIABLE",
            "evidence": f"None of the key terms found in page. Searched for: {key_terms}",
            "url": url,
        }
    elif found / total >= 0.7:
        return {
            "status": "VERIFIED",
            "evidence": f"Found {found}/{total} key terms. Excerpt: ...{evidence_excerpt}...",
            "url": url,
        }
    else:
        return {
            "status": "PARTIALLY_VERIFIED",
            "evidence": (
                f"Found {found}/{total} key terms ({found_terms}). "
                f"Missing: {missing_terms}. Excerpt: ...{evidence_excerpt}..."
            ),
            "url": url,
        }


class CitationVerifierTool(BaseTool):
    """Verify that a cited URL actually contains the claimed information.

    Fetches the URL, extracts text, and checks for presence of key claim terms.
    Uses per-domain rate limiting: only sleeps when hitting the same domain
    within ``domain_delay`` seconds. Different domains are fetched immediately.
    """

    name: str = "citation_verifier"
    description: str = (
        "Verify that a cited URL actually contains the claimed information. "
        "Input: JSON with 'url' and 'claim' keys. "
        "Returns verification status (VERIFIED/PARTIALLY_VERIFIED/UNVERIFIABLE/ERROR) "
        "with evidence excerpt."
    )

    timeout: float = Field(default=REQUEST_TIMEOUT, description="Request timeout in seconds")
    domain_delay: float = Field(
        default=DEFAULT_DOMAIN_DELAY,
        description="Minimum delay in seconds between fetches to the same domain.",
    )

    # Internal state: last fetch timestamp per domain.
    # Using dict default_factory would conflict with Pydantic; we init in __init__.
    _domain_last_fetch: dict[str, float] = {}

    def model_post_init(self, __context: object) -> None:
        """Initialize per-domain tracking after Pydantic model init."""
        self._domain_last_fetch = {}

    def _rate_limit(self, url: str) -> None:
        """Sleep only if we've recently fetched from the same domain."""
        try:
            domain = urlparse(url).netloc.lower()
        except Exception:
            return

        if not domain:
            return

        now = time.monotonic()
        last = self._domain_last_fetch.get(domain, 0.0)
        elapsed = now - last

        if elapsed < self.domain_delay:
            time.sleep(self.domain_delay - elapsed)

        self._domain_last_fetch[domain] = time.monotonic()

    def _run(self, input_data: str) -> str:
        """Verify a citation.

        Args:
            input_data: JSON string with 'url' and 'claim' keys,
                       or a plain URL string with claim context.

        Returns:
            JSON string with verification result.
        """
        try:
            data = json.loads(input_data)
            url = data["url"]
            claim = data["claim"]
        except (json.JSONDecodeError, KeyError, TypeError):
            # Try to parse as "url: <url>, claim: <claim>" format
            parts = input_data.split(",", 1)
            if len(parts) == 2:
                url = parts[0].strip()
                claim = parts[1].strip()
            else:
                return json.dumps(
                    {
                        "status": "ERROR",
                        "evidence": "Invalid input. Expected JSON with 'url' and 'claim' keys.",
                        "url": "",
                    }
                )

        # Per-domain rate limiting
        self._rate_limit(url)

        result = verify_citation(url, claim, timeout=self.timeout)
        return json.dumps(result, indent=2)
