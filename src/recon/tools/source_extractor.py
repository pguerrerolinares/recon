"""Source extractor - extract and deduplicate URLs from research markdown.

Runs as a post-processing step after the investigation phase to produce
a ``sources.json`` summary file.  This file is useful for:

1. Quick visibility into how many unique sources the investigation found.
2. Feeding the verification crew a pre-built index of known citations.
3. Future migration to a SQLite ``sources`` table (v0.3).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

# Regex that matches http/https URLs in free text or markdown links.
_URL_RE = re.compile(r"https?://[^\s\)\]\>,\"']+")


def extract_sources(research_dir: str) -> dict:
    """Extract and deduplicate all URLs from research markdown files.

    Args:
        research_dir: Directory containing ``*.md`` research files.

    Returns:
        A dict with keys:
        - ``total_urls``: total URL occurrences (with duplicates)
        - ``unique_urls``: count of unique URLs
        - ``urls``: sorted list of ``{url, count, domain, documents}``
        - ``by_document``: mapping from filename to list of unique URLs
        - ``by_domain``: mapping from domain to count of unique URLs
    """
    research_path = Path(research_dir)
    if not research_path.exists():
        return _empty_result()

    md_files = sorted(research_path.glob("*.md"))
    if not md_files:
        return _empty_result()

    # url -> {count, documents set}
    url_info: dict[str, dict] = defaultdict(lambda: {"count": 0, "documents": set()})
    by_document: dict[str, list[str]] = {}

    for md_file in md_files:
        text = md_file.read_text(errors="replace")
        found_urls = _URL_RE.findall(text)
        # Clean trailing punctuation that regex may capture
        found_urls = [_clean_url(u) for u in found_urls]
        # Dedupe within document
        unique_in_doc = list(dict.fromkeys(found_urls))
        by_document[md_file.name] = unique_in_doc

        for url in found_urls:
            url_info[url]["count"] += 1
            url_info[url]["documents"].add(md_file.name)

    # Build sorted URL list (most cited first)
    urls_list = []
    for url, info in sorted(url_info.items(), key=lambda x: -x[1]["count"]):
        domain = _extract_domain(url)
        urls_list.append(
            {
                "url": url,
                "count": info["count"],
                "domain": domain,
                "documents": sorted(info["documents"]),
            }
        )

    # Domain summary
    domain_counts: dict[str, int] = defaultdict(int)
    for entry in urls_list:
        domain_counts[entry["domain"]] += 1

    total = sum(info["count"] for info in url_info.values())

    return {
        "total_urls": total,
        "unique_urls": len(url_info),
        "urls": urls_list,
        "by_document": {k: v for k, v in sorted(by_document.items())},
        "by_domain": dict(sorted(domain_counts.items(), key=lambda x: -x[1])),
    }


def write_sources_json(research_dir: str) -> dict:
    """Extract sources and write ``sources.json`` to the research directory.

    Args:
        research_dir: Directory containing ``*.md`` research files.

    Returns:
        The sources summary dict (same as :func:`extract_sources`).
    """
    result = extract_sources(research_dir)
    out_path = Path(research_dir) / "sources.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def _clean_url(url: str) -> str:
    """Strip trailing markdown/punctuation artefacts from a captured URL."""
    # Remove trailing punctuation that is not part of the URL
    while url and url[-1] in (".", ",", ";", ")", "]", ">", "'", '"'):
        url = url[:-1]
    return url


def _extract_domain(url: str) -> str:
    """Extract the domain from a URL, stripping ``www.``."""
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return "unknown"


def _empty_result() -> dict:
    return {
        "total_urls": 0,
        "unique_urls": 0,
        "urls": [],
        "by_document": {},
        "by_domain": {},
    }
