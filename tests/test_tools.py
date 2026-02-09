"""Tests for recon.tools module - custom verification tools."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003

from recon.tools.citation_verifier import CitationVerifierTool, verify_citation
from recon.tools.claim_extractor import (
    Claim,
    ClaimExtractorTool,
    _prioritize_claims,
    extract_claims,
)
from recon.tools.confidence_scorer import (
    ConfidenceScorerTool,
    score_confidence,
    score_label,
)
from recon.tools.contradiction_detector import (
    ContradictionDetectorTool,
    detect_contradiction,
)
from recon.tools.source_tracker import SourceTrackerTool, read_audit_trail, track_source

# --- ClaimExtractor tests ---


class TestClaimExtractor:
    def test_extract_from_nonexistent_file(self) -> None:
        claims = extract_claims("/nonexistent/path.md")
        assert claims == []

    def test_extract_statistics(self, tmp_path: Path) -> None:
        doc = tmp_path / "test.md"
        doc.write_text(
            "# Report\n\n"
            "CrewAI has 44K stars on GitHub.\n"
            "Revenue reached $1.5B in 2025.\n"
            "The market grew 45% year over year.\n"
        )
        claims = extract_claims(str(doc))
        assert len(claims) >= 2
        # Should detect numbers and dollar amounts
        claim_texts = [c.text for c in claims]
        assert any("44K" in t for t in claim_texts)

    def test_extract_pricing(self, tmp_path: Path) -> None:
        doc = tmp_path / "pricing.md"
        doc.write_text(
            "# Pricing\n\n"
            "The API costs $0.45/M input tokens.\n"
            "They offer a free tier with 1000 credits.\n"
        )
        claims = extract_claims(str(doc))
        assert len(claims) >= 1
        types = [c.claim_type for c in claims]
        assert "pricing" in types or "statistic" in types

    def test_extract_dates(self, tmp_path: Path) -> None:
        doc = tmp_path / "timeline.md"
        doc.write_text(
            "# Timeline\n\n"
            "The company was founded in 2020.\n"
            "They launched version 2.0 in January 2025.\n"
        )
        claims = extract_claims(str(doc))
        assert len(claims) >= 1
        types = [c.claim_type for c in claims]
        assert "date" in types

    def test_extract_with_sources(self, tmp_path: Path) -> None:
        doc = tmp_path / "sourced.md"
        doc.write_text(
            "# Analysis\n\nCrewAI has 44K stars [Source: https://github.com/crewAIInc/crewAI].\n"
        )
        claims = extract_claims(str(doc))
        sourced = [c for c in claims if c.cited_source]
        assert len(sourced) >= 1
        assert sourced[0].cited_source is not None and "github.com" in sourced[0].cited_source

    def test_tool_interface(self, tmp_path: Path) -> None:
        doc = tmp_path / "test.md"
        doc.write_text("Revenue reached $10M in 2025.\n")
        tool = ClaimExtractorTool()
        result = tool._run(str(doc))
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) >= 1
        assert "claim_id" in parsed[0]

    def test_deduplicate_claims(self, tmp_path: Path) -> None:
        doc = tmp_path / "dupe.md"
        doc.write_text(
            "Revenue reached $10M in 2025.\nRevenue reached $10M in 2025.\n"  # Duplicate line
        )
        claims = extract_claims(str(doc))
        texts = [c.text for c in claims]
        # Should not have duplicates
        assert len(texts) == len(set(texts))

    def test_max_claims_cap(self, tmp_path: Path) -> None:
        """extract_claims with max_claims should cap the output."""
        doc = tmp_path / "many.md"
        lines = [f"Revenue reached ${i}M in 2025.\n" for i in range(1, 51)]
        doc.write_text("# Report\n\n" + "".join(lines))
        all_claims = extract_claims(str(doc), max_claims=0)
        capped_claims = extract_claims(str(doc), max_claims=10)
        assert len(all_claims) > 10
        assert len(capped_claims) == 10

    def test_max_claims_no_cap_when_under_limit(self, tmp_path: Path) -> None:
        doc = tmp_path / "few.md"
        doc.write_text("# Report\n\nRevenue reached $10M in 2025.\n")
        claims = extract_claims(str(doc), max_claims=100)
        assert len(claims) >= 1  # should return all, not truncate

    def test_max_claims_renumbers(self, tmp_path: Path) -> None:
        """After prioritization, claim IDs should be renumbered sequentially."""
        doc = tmp_path / "many.md"
        lines = [f"Revenue reached ${i}M in 2025.\n" for i in range(1, 51)]
        doc.write_text("# Report\n\n" + "".join(lines))
        claims = extract_claims(str(doc), max_claims=5)
        ids = [c.claim_id for c in claims]
        assert ids == ["C1", "C2", "C3", "C4", "C5"]

    def test_prioritize_prefers_claims_with_sources(self) -> None:
        """Claims with cited_source should be prioritized over those without."""
        claims = [
            Claim("C1", "No source claim", "doc.md", "statistic", None),
            Claim("C2", "Has source claim", "doc.md", "statistic", "https://example.com"),
            Claim("C3", "Another no source", "doc.md", "date", None),
        ]
        result = _prioritize_claims(claims, max_claims=2)
        assert len(result) == 2
        # The claim with a source should come first
        assert result[0].text == "Has source claim"

    def test_prioritize_prefers_statistics_over_quotes(self) -> None:
        """Statistics should rank higher than quotes."""
        claims = [
            Claim("C1", "A quote claim", "doc.md", "quote", None),
            Claim("C2", "A stat claim", "doc.md", "statistic", None),
            Claim("C3", "A date claim", "doc.md", "date", None),
        ]
        result = _prioritize_claims(claims, max_claims=2)
        types = [c.claim_type for c in result]
        assert "statistic" in types
        assert "quote" not in types

    def test_tool_max_claims_field(self, tmp_path: Path) -> None:
        """ClaimExtractorTool should respect max_claims field."""
        doc = tmp_path / "many.md"
        lines = [f"Revenue reached ${i}M in 2025.\n" for i in range(1, 51)]
        doc.write_text("# Report\n\n" + "".join(lines))
        tool = ClaimExtractorTool(max_claims=5)
        result = json.loads(tool._run(str(doc)))
        assert len(result) == 5


# --- CitationVerifier tests ---


class TestCitationVerifier:
    def test_verify_unreachable_url(self) -> None:
        result = verify_citation(
            url="https://this-domain-does-not-exist-12345.com/page",
            claim="Some claim",
            timeout=3.0,
        )
        assert result["status"] == "ERROR"

    def test_verify_invalid_url(self) -> None:
        result = verify_citation(url="not-a-url", claim="test", timeout=3.0)
        assert result["status"] == "ERROR"

    def test_tool_interface_invalid_input(self) -> None:
        tool = CitationVerifierTool()
        result = tool._run("invalid json")
        parsed = json.loads(result)
        assert parsed["status"] == "ERROR"

    def test_tool_interface_valid_json(self) -> None:
        tool = CitationVerifierTool(timeout=3.0)
        result = tool._run(
            json.dumps(
                {
                    "url": "https://this-does-not-exist-99999.com",
                    "claim": "test claim",
                }
            )
        )
        parsed = json.loads(result)
        assert parsed["status"] in ("ERROR", "UNVERIFIABLE")

    def test_per_domain_rate_limit_same_domain(self) -> None:
        """Same domain should be rate-limited."""
        import time

        tool = CitationVerifierTool(timeout=3.0, domain_delay=0.3)
        # First call to this domain -- records timestamp
        tool._rate_limit("https://example.com/page1")
        start = time.monotonic()
        # Second call to same domain -- should sleep
        tool._rate_limit("https://example.com/page2")
        elapsed = time.monotonic() - start
        assert elapsed >= 0.2  # Should have slept ~0.3s

    def test_per_domain_rate_limit_different_domains(self) -> None:
        """Different domains should NOT be rate-limited."""
        import time

        tool = CitationVerifierTool(timeout=3.0, domain_delay=1.0)
        tool._rate_limit("https://example.com/page1")
        start = time.monotonic()
        # Different domain -- should not sleep
        tool._rate_limit("https://other-domain.com/page1")
        elapsed = time.monotonic() - start
        assert elapsed < 0.1  # Should be nearly instant

    def test_domain_delay_replaces_fetch_delay(self) -> None:
        """Tool should use domain_delay, not the old fetch_delay."""
        tool = CitationVerifierTool(timeout=3.0, domain_delay=0.5)
        assert hasattr(tool, "domain_delay")
        assert tool.domain_delay == 0.5
        # Old fetch_delay should not exist
        assert not hasattr(tool, "fetch_delay")


# --- ConfidenceScorer tests ---


class TestConfidenceScorer:
    def test_verified_single_source(self) -> None:
        score = score_confidence("VERIFIED", source_count=1)
        assert 0.7 <= score <= 1.0

    def test_verified_multiple_sources(self) -> None:
        score_single = score_confidence("VERIFIED", source_count=1)
        score_multi = score_confidence("VERIFIED", source_count=3)
        assert score_multi > score_single

    def test_verified_primary_source_bonus(self) -> None:
        score_no_primary = score_confidence("VERIFIED", source_count=1, has_primary_source=False)
        score_primary = score_confidence("VERIFIED", source_count=1, has_primary_source=True)
        assert score_primary > score_no_primary

    def test_partially_verified(self) -> None:
        score = score_confidence("PARTIALLY_VERIFIED", source_count=1)
        assert 0.4 <= score <= 0.7

    def test_unverifiable(self) -> None:
        score = score_confidence("UNVERIFIABLE", source_count=1)
        assert 0.2 <= score <= 0.4

    def test_contradicted(self) -> None:
        score = score_confidence("CONTRADICTED", source_count=1)
        assert 0.0 <= score <= 0.1

    def test_unknown_status_defaults(self) -> None:
        score = score_confidence("SOMETHING_ELSE", source_count=1)
        assert 0.2 <= score <= 0.4  # Defaults to UNVERIFIABLE range

    def test_score_label_high(self) -> None:
        assert score_label(0.85) == "HIGH"

    def test_score_label_medium(self) -> None:
        assert score_label(0.55) == "MEDIUM"

    def test_score_label_low(self) -> None:
        assert score_label(0.3) == "LOW"

    def test_score_label_very_low(self) -> None:
        assert score_label(0.05) == "VERY_LOW"

    def test_score_clamped_to_range(self) -> None:
        # Even with many bonuses, should not exceed 1.0
        score = score_confidence("VERIFIED", source_count=10, has_primary_source=True)
        assert score <= 1.0

    def test_tool_interface(self) -> None:
        tool = ConfidenceScorerTool()
        result = tool._run(
            json.dumps(
                {
                    "verification_status": "VERIFIED",
                    "source_count": 2,
                    "has_primary_source": True,
                }
            )
        )
        parsed = json.loads(result)
        assert "score" in parsed
        assert "label" in parsed
        assert parsed["label"] == "HIGH"

    def test_tool_interface_invalid_input(self) -> None:
        tool = ConfidenceScorerTool()
        result = tool._run("not json")
        parsed = json.loads(result)
        assert "error" in parsed
        assert parsed["label"] == "LOW"


# --- ContradictionDetector tests ---


class TestContradictionDetector:
    def test_consistent_numbers(self) -> None:
        result = detect_contradiction(
            claim_a="CrewAI has 44K stars",
            source_a="doc-a.md",
            claim_b="CrewAI has about 44,000 stars",
            source_b="doc-b.md",
        )
        assert result["status"] == "CONSISTENT"

    def test_contradicting_numbers(self) -> None:
        result = detect_contradiction(
            claim_a="CrewAI has 44K stars",
            source_a="doc-a.md",
            claim_b="CrewAI has 10K stars",
            source_b="doc-b.md",
        )
        assert result["status"] == "CONTRADICTED"

    def test_contradicting_years(self) -> None:
        result = detect_contradiction(
            claim_a="Founded in 2020",
            source_a="doc-a.md",
            claim_b="Founded in 2023",
            source_b="doc-b.md",
        )
        assert result["status"] == "CONTRADICTED"

    def test_ambiguous_no_numbers(self) -> None:
        result = detect_contradiction(
            claim_a="The product is well designed",
            source_a="doc-a.md",
            claim_b="The product is poorly designed",
            source_b="doc-b.md",
        )
        # Without numbers, should be ambiguous (negation detection may or may not trigger)
        assert result["status"] in ("CONTRADICTED", "AMBIGUOUS")

    def test_different_magnitude_ignored(self) -> None:
        # $10M and 44K stars are different metrics, should not contradict
        result = detect_contradiction(
            claim_a="Revenue is $10M",
            source_a="doc-a.md",
            claim_b="GitHub stars: 44000",
            source_b="doc-b.md",
        )
        # Numbers are very different orders of magnitude, should not compare
        assert result["status"] in ("CONSISTENT", "AMBIGUOUS")

    def test_tool_interface(self) -> None:
        tool = ContradictionDetectorTool()
        result = tool._run(
            json.dumps(
                {
                    "claim_a": "Revenue is $10M",
                    "source_a": "doc-a.md",
                    "claim_b": "Revenue is $50M",
                    "source_b": "doc-b.md",
                }
            )
        )
        parsed = json.loads(result)
        assert parsed["status"] == "CONTRADICTED"

    def test_tool_interface_invalid_input(self) -> None:
        tool = ContradictionDetectorTool()
        result = tool._run("not json")
        parsed = json.loads(result)
        assert parsed["status"] == "AMBIGUOUS"


# --- SourceTracker tests ---


class TestSourceTracker:
    def test_track_and_read(self, tmp_path: Path) -> None:
        output_dir = str(tmp_path / "verification")
        entry = track_source(
            claim_id="C1",
            claim_text="CrewAI has 44K stars",
            source_url="https://github.com/crewAIInc/crewAI",
            verification_status="VERIFIED",
            confidence_score=0.85,
            output_dir=output_dir,
        )
        assert entry["claim_id"] == "C1"
        assert entry["verification_status"] == "VERIFIED"

        # Read back
        entries = read_audit_trail(output_dir)
        assert len(entries) == 1
        assert entries[0]["claim_id"] == "C1"

    def test_multiple_entries(self, tmp_path: Path) -> None:
        output_dir = str(tmp_path / "verification")
        for i in range(3):
            track_source(
                claim_id=f"C{i}",
                claim_text=f"Claim {i}",
                source_url=f"https://example.com/{i}",
                verification_status="VERIFIED",
                output_dir=output_dir,
            )
        entries = read_audit_trail(output_dir)
        assert len(entries) == 3

    def test_evidence_truncation(self, tmp_path: Path) -> None:
        output_dir = str(tmp_path / "verification")
        long_excerpt = "x" * 1000
        entry = track_source(
            claim_id="C1",
            claim_text="test",
            source_url="https://example.com",
            verification_status="VERIFIED",
            evidence_excerpt=long_excerpt,
            output_dir=output_dir,
        )
        assert len(entry["evidence_excerpt"]) <= 500

    def test_tool_interface(self, tmp_path: Path) -> None:
        output_dir = str(tmp_path / "verification")
        tool = SourceTrackerTool(output_dir=output_dir)
        result = tool._run(
            json.dumps(
                {
                    "claim_id": "C1",
                    "claim_text": "test claim",
                    "source_url": "https://example.com",
                    "verification_status": "VERIFIED",
                }
            )
        )
        parsed = json.loads(result)
        assert parsed["tracked"] is True

    def test_tool_interface_invalid_input(self, tmp_path: Path) -> None:
        output_dir = str(tmp_path / "verification")
        tool = SourceTrackerTool(output_dir=output_dir)
        result = tool._run("not json")
        parsed = json.loads(result)
        assert "error" in parsed

    def test_empty_audit_trail(self, tmp_path: Path) -> None:
        entries = read_audit_trail(str(tmp_path / "nonexistent"))
        assert entries == []
