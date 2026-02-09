"""Tests for recon.tools module - custom verification tools."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from unittest.mock import MagicMock

from recon.tools.citation_verifier import CitationVerifierTool, verify_citation
from recon.tools.claim_extractor import (
    Claim,
    ClaimExtractorTool,
    _is_bibliography_entry,
    _is_incomplete_sentence,
    _is_markdown_noise,
    _llm_filter_claims,
    _passes_prefilter,
    _prioritize_claims,
    _strip_source_sections,
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
from recon.tools.semantic_verifier import (
    SemanticVerifierTool,
    verify_semantically,
)
from recon.tools.source_tracker import SourceTrackerTool, read_audit_trail, track_source


def _claim_json(
    text: str,
    typ: str = "statistic",
    keep: bool = True,
    reason: str = "valid",
) -> dict[str, object]:
    """Build a claim dict matching the LLM filter response schema."""
    return {"text": text, "type": typ, "keep": keep, "reason": reason}


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

    def test_tool_accepts_llm_parameter(self) -> None:
        """ClaimExtractorTool should accept an llm parameter."""
        tool = ClaimExtractorTool(max_claims=10, llm=None)
        assert tool.llm is None

    def test_tool_with_llm_calls_filter(self, tmp_path: Path) -> None:
        """ClaimExtractorTool with llm should call the LLM filter."""
        doc = tmp_path / "test.md"
        doc.write_text("Revenue reached $10M in 2025.\n")
        mock_llm = MagicMock()
        mock_llm.call.return_value = json.dumps([
            _claim_json("Revenue reached $10M in 2025."),
        ])
        tool = ClaimExtractorTool(llm=mock_llm)
        result = json.loads(tool._run(str(doc)))
        assert len(result) >= 1
        mock_llm.call.assert_called_once()


# --- ClaimExtractor Pre-filter tests ---


class TestClaimPrefilter:
    """Tests for the pre-filter heuristics that reject garbage claims."""

    # --- _is_markdown_noise ---

    def test_rejects_table_row_with_pipes(self) -> None:
        assert _is_markdown_noise("Donation Date** | December 2025 |") is True

    def test_rejects_double_pipe_table_fragment(self) -> None:
        assert _is_markdown_noise("Official** | Developed and maintained by (e.") is True

    def test_rejects_bold_label_with_colon(self) -> None:
        assert _is_markdown_noise("Type**: Open-source AI code assistant") is True

    def test_rejects_bold_label_with_bracket(self) -> None:
        assert _is_markdown_noise("Features**] Full MCP support with resources") is True

    def test_rejects_high_special_char_density(self) -> None:
        assert _is_markdown_noise("## **Title** | **Value** ##") is True

    def test_accepts_normal_sentence(self) -> None:
        assert _is_markdown_noise("CrewAI has 44K stars on GitHub.") is False

    def test_accepts_sentence_with_single_pipe(self) -> None:
        # A single pipe in text is OK (could be in a quote); 2+ pipes rejects
        assert _is_markdown_noise("Revenue is $10M with growth rate | stable.") is False

    def test_accepts_normal_dollar_claim(self) -> None:
        assert _is_markdown_noise("Revenue reached $1.5B in 2025.") is False

    # --- _is_bibliography_entry ---

    def test_rejects_numbered_bold_reference(self) -> None:
        assert _is_bibliography_entry('1. **Anthropic - "Introducing the MCP"** (Nov 2024)') is True

    def test_rejects_author_title_pattern(self) -> None:
        text = 'Anthropic - "Introducing the Model Context Protocol"'
        assert _is_bibliography_entry(text) is True

    def test_rejects_url_line(self) -> None:
        assert _is_bibliography_entry("URL: https://www.anthropic.com/news/mcp") is True

    def test_rejects_access_date(self) -> None:
        assert _is_bibliography_entry("Access Date: January 2026") is True

    def test_rejects_bare_url(self) -> None:
        assert _is_bibliography_entry("https://www.anthropic.com/news/mcp") is True

    def test_accepts_normal_claim(self) -> None:
        assert _is_bibliography_entry("The company was founded in 2020.") is False

    def test_accepts_quoted_claim(self) -> None:
        assert _is_bibliography_entry('According to the CEO: "We raised $50M in funding."') is False

    # --- _is_incomplete_sentence ---

    def test_rejects_truncated_parenthetical(self) -> None:
        text = "Developed and maintained by the service provider (e."
        assert _is_incomplete_sentence(text) is True

    def test_rejects_too_few_words(self) -> None:
        assert _is_incomplete_sentence("Revenue $10M") is True

    def test_rejects_continuation_word_start(self) -> None:
        text = "and Model Context Protocol to New Agentic AI Foundation."
        assert _is_incomplete_sentence(text) is True

    def test_rejects_md_continuation(self) -> None:
        assert _is_incomplete_sentence("md and Model Context Protocol to New Foundation.") is True

    def test_accepts_full_sentence(self) -> None:
        assert _is_incomplete_sentence("The company was founded in 2020.") is False

    def test_accepts_sentence_ending_with_exclamation(self) -> None:
        assert _is_incomplete_sentence("Revenue reached $10M in 2025!") is False

    def test_accepts_sentence_ending_with_question(self) -> None:
        assert _is_incomplete_sentence("Did revenue reach $10M in 2025?") is False

    # --- _passes_prefilter (combined) ---

    def test_rejects_table_fragment_combined(self) -> None:
        assert _passes_prefilter("Type**: Open-source AI code assistant") is False

    def test_rejects_bibliography_combined(self) -> None:
        assert _passes_prefilter('1. **Anthropic - "Introducing MCP"**') is False

    def test_rejects_incomplete_combined(self) -> None:
        assert _passes_prefilter("Official maintained by (e.") is False

    def test_accepts_real_claim(self) -> None:
        assert _passes_prefilter("CrewAI has 44K stars on GitHub.") is True

    def test_accepts_attribution_claim(self) -> None:
        assert _passes_prefilter("The protocol was created by Anthropic in November 2024.") is True


# --- Source Section Stripping tests ---


class TestSourceSectionStripping:
    """Tests for _strip_source_sections that removes bibliography."""

    def test_strips_sources_section(self) -> None:
        text = (
            "# Report\n\nSome content here.\n\n"
            "## Sources\n\n"
            '1. **Anthropic - "Introducing MCP"** (Nov 2024)\n'
            "   URL: https://example.com\n"
        )
        result = _strip_source_sections(text)
        assert "Sources" not in result
        assert "Anthropic" not in result
        assert "Some content here." in result

    def test_strips_references_section(self) -> None:
        text = "# Report\n\nContent.\n\n## References\n\nRef 1.\n"
        result = _strip_source_sections(text)
        assert "References" not in result
        assert "Content." in result

    def test_strips_bibliography_section(self) -> None:
        text = "# Report\n\nContent.\n\n# Bibliography\n\nEntry 1.\n"
        result = _strip_source_sections(text)
        assert "Bibliography" not in result

    def test_preserves_doc_without_sources(self) -> None:
        text = "# Report\n\nContent with $10M revenue.\n"
        assert _strip_source_sections(text) == text

    def test_strips_at_h3_level(self) -> None:
        text = "# Report\n\nContent.\n\n### Sources\n\nRef.\n"
        result = _strip_source_sections(text)
        assert "Sources" not in result


# --- Integration: Pre-filter with real document ---


class TestPrefilterIntegration:
    """Integration tests using the real e2e document patterns."""

    def test_rejects_garbage_from_mcp_report(self, tmp_path: Path) -> None:
        """Claims from the MCP e2e report that were garbage should now be filtered."""
        doc = tmp_path / "report.md"
        doc.write_text(
            "# MCP Report\n\n"
            "Created by Anthropic and released in November 2024, MCP has rapidly evolved.\n"
            "In December 2025, Anthropic donated MCP to the Agentic AI Foundation.\n\n"
            "| Attribute | Details |\n"
            "|-----------|--------|\n"
            "| **Donation Date** | December 2025 |\n"
            "| **Type** | Open-source protocol |\n\n"
            "#### Continue\n"
            "- **Provider**: Continue.dev\n"
            "- **Type**: Open-source AI code assistant\n"
            '- **Features**: Full MCP support with "@" mentions for resources.\n\n'
            "## Sources\n\n"
            '1. **Anthropic - "Introducing the Model Context Protocol"** (Nov 25, 2024)\n'
            "   - URL: https://www.anthropic.com/news/model-context-protocol\n"
            "   - Access Date: January 2026\n"
        )
        claims = extract_claims(str(doc))
        claim_texts = [c.text for c in claims]

        # These garbage patterns should be filtered out
        for text in claim_texts:
            assert "**" not in text or '"' in text, f"Bold label artifact survived: {text}"
            assert text.count("|") < 2, f"Table fragment survived: {text}"
            assert not text.startswith("Access Date"), f"Access date survived: {text}"

        # But real claims should survive
        assert any("Anthropic" in t and "November 2024" in t for t in claim_texts), (
            f"Real claim about MCP release was incorrectly filtered. Claims: {claim_texts}"
        )

    def test_source_section_excluded(self, tmp_path: Path) -> None:
        """Claims from Sources/References sections should not be extracted."""
        doc = tmp_path / "report.md"
        doc.write_text(
            "# Report\n\n"
            "The company raised $50M in 2025.\n\n"
            "## Sources\n\n"
            '1. **Anthropic - "Introducing MCP"** (Nov 25, 2024)\n'
            "   URL: https://www.anthropic.com/news/model-context-protocol\n"
            "   Access Date: January 2026\n"
            '2. **Wikipedia - "Model Context Protocol"**\n'
            "   URL: https://en.wikipedia.org/wiki/MCP\n"
        )
        claims = extract_claims(str(doc))
        claim_texts = [c.text for c in claims]

        # Nothing from the Sources section should appear
        assert not any("Wikipedia" in t for t in claim_texts)
        assert not any("Introducing MCP" in t for t in claim_texts)
        # The real claim should survive
        assert any("$50M" in t for t in claim_texts)

    def test_claim_count_reduced_on_noisy_doc(self, tmp_path: Path) -> None:
        """Noisy documents should produce fewer claims with pre-filtering."""
        doc = tmp_path / "noisy.md"
        # Mix of real claims and noise
        doc.write_text(
            "# Analysis\n\n"
            "Revenue reached $10M in 2025.\n"
            "The company has 5000 users globally.\n"
            "Growth rate was 45% year over year.\n\n"
            "| Feature | Status |\n"
            "| **Type** | Cloud platform |\n"
            "| **Users** | 5000 |\n"
            "| **Revenue** | $10M |\n\n"
            "## Sources\n\n"
            '1. **Company - "Annual Report"** (2025)\n'
            "   Access Date: January 2026\n"
        )
        claims = extract_claims(str(doc))
        # Should only have the real prose claims, not table/source entries
        assert len(claims) <= 5
        # All claims should be substantive sentences
        for c in claims:
            assert len(c.text.split()) >= 4, f"Claim too short: {c.text}"


# --- LLM Filter tests ---


class TestLLMFilter:
    """Tests for _llm_filter_claims LLM-based quality filter."""

    _REV = "Revenue reached $10M in 2025."

    def _make_claims(self, texts: list[str]) -> list[Claim]:
        """Helper to create Claim objects from text list."""
        return [
            Claim(f"C{i+1}", text, "doc.md", "statistic")
            for i, text in enumerate(texts)
        ]

    def test_keeps_valid_claims(self) -> None:
        """LLM marks claims as keep=True -> they survive."""
        claims = self._make_claims([self._REV])
        mock_llm = MagicMock()
        mock_llm.call.return_value = json.dumps([
            _claim_json(self._REV),
        ])
        result = _llm_filter_claims(claims, mock_llm)
        assert len(result) == 1
        assert "$10M" in result[0].text

    def test_rejects_garbage_claims(self) -> None:
        """LLM marks claims as keep=False -> they are removed."""
        garbage = "Type**: Open-source AI code assistant"
        claims = self._make_claims([self._REV, garbage])
        mock_llm = MagicMock()
        mock_llm.call.return_value = json.dumps([
            _claim_json(self._REV),
            _claim_json(garbage, keep=False, reason="fragment"),
        ])
        result = _llm_filter_claims(claims, mock_llm)
        assert len(result) == 1
        assert "Revenue" in result[0].text

    def test_decomposes_compound_claims(self) -> None:
        """LLM decomposes compound claim -> multiple atomic claims."""
        compound = "Founded in 2020 with $50M funding and 100 employees."
        claims = self._make_claims([compound])
        mock_llm = MagicMock()
        mock_llm.call.return_value = json.dumps([
            _claim_json(compound, keep=False, reason="compound"),
            _claim_json("The company was founded in 2020.", "date"),
            _claim_json("The company received $50M in funding."),
            _claim_json("The company had 100 employees."),
        ])
        result = _llm_filter_claims(claims, mock_llm)
        assert len(result) == 3
        assert result[0].claim_id == "C1"
        assert result[1].claim_id == "C2"
        assert result[2].claim_id == "C3"

    def test_fallback_on_llm_error(self) -> None:
        """LLM raises exception -> returns original claims unchanged."""
        claims = self._make_claims([self._REV])
        mock_llm = MagicMock()
        mock_llm.call.side_effect = RuntimeError("API error")
        result = _llm_filter_claims(claims, mock_llm)
        assert len(result) == 1
        assert result[0].text == self._REV

    def test_fallback_on_invalid_json(self) -> None:
        """LLM returns invalid JSON -> returns original claims."""
        claims = self._make_claims([self._REV])
        mock_llm = MagicMock()
        mock_llm.call.return_value = "This is not JSON at all"
        result = _llm_filter_claims(claims, mock_llm)
        assert len(result) == 1
        assert result[0].text == self._REV

    def test_fallback_on_non_list_response(self) -> None:
        """LLM returns a JSON object instead of array -> originals."""
        claims = self._make_claims([self._REV])
        mock_llm = MagicMock()
        mock_llm.call.return_value = json.dumps({"error": "unexpected"})
        result = _llm_filter_claims(claims, mock_llm)
        assert len(result) == 1

    def test_fallback_when_all_rejected(self) -> None:
        """LLM rejects everything -> returns original claims as fallback."""
        claims = self._make_claims([self._REV])
        mock_llm = MagicMock()
        mock_llm.call.return_value = json.dumps([
            _claim_json(self._REV, keep=False, reason="rejected"),
        ])
        result = _llm_filter_claims(claims, mock_llm)
        assert len(result) == 1  # fallback to originals
        assert result[0].text == self._REV

    def test_strips_markdown_fences_from_response(self) -> None:
        """LLM wraps response in ```json fences -> still parses."""
        claims = self._make_claims([self._REV])
        mock_llm = MagicMock()
        fenced = json.dumps([_claim_json(self._REV)])
        mock_llm.call.return_value = (
            "```json\n" + fenced + "\n```"
        )
        result = _llm_filter_claims(claims, mock_llm)
        assert len(result) == 1
        assert "$10M" in result[0].text

    def test_empty_claims_returns_empty(self) -> None:
        """Empty input -> empty output, no LLM call."""
        mock_llm = MagicMock()
        result = _llm_filter_claims([], mock_llm)
        assert result == []
        mock_llm.call.assert_not_called()

    def test_inherits_cited_source(self) -> None:
        """LLM-filtered claims should inherit cited_source."""
        claims = [
            Claim(
                "C1", self._REV, "doc.md",
                "statistic", "https://example.com",
            ),
        ]
        mock_llm = MagicMock()
        mock_llm.call.return_value = json.dumps([
            _claim_json(self._REV),
        ])
        result = _llm_filter_claims(claims, mock_llm)
        assert len(result) == 1
        assert result[0].cited_source == "https://example.com"

    def test_validates_claim_type(self) -> None:
        """Invalid claim types from LLM should default to 'statistic'."""
        claims = self._make_claims([self._REV])
        mock_llm = MagicMock()
        mock_llm.call.return_value = json.dumps([
            _claim_json(self._REV, typ="invalid_type"),
        ])
        result = _llm_filter_claims(claims, mock_llm)
        assert result[0].claim_type == "statistic"

    def test_rejects_short_text_from_llm(self) -> None:
        """Claims shorter than 15 chars from LLM should be rejected."""
        claims = self._make_claims([self._REV])
        mock_llm = MagicMock()
        mock_llm.call.return_value = json.dumps([
            _claim_json("short"),
            _claim_json(self._REV),
        ])
        result = _llm_filter_claims(claims, mock_llm)
        assert len(result) == 1
        assert "Revenue" in result[0].text


# --- Integration: extract_claims with LLM ---


class TestExtractClaimsWithLLM:
    """Integration tests for extract_claims with LLM filter."""

    _REV = "Revenue reached $10M in 2025."

    def test_extract_with_llm_filter(self, tmp_path: Path) -> None:
        """extract_claims with llm= should apply LLM filtering."""
        doc = tmp_path / "test.md"
        doc.write_text(
            "# Report\n\n"
            "Revenue reached $10M in 2025.\n"
            "The company has 5000 users globally.\n"
        )
        users = "The company has 5000 users globally."
        mock_llm = MagicMock()
        mock_llm.call.return_value = json.dumps([
            _claim_json(self._REV),
            _claim_json(users),
        ])
        claims = extract_claims(str(doc), llm=mock_llm)
        assert len(claims) >= 1
        mock_llm.call.assert_called_once()

    def test_extract_without_llm(self, tmp_path: Path) -> None:
        """extract_claims without llm= uses regex only."""
        doc = tmp_path / "test.md"
        doc.write_text("Revenue reached $10M in 2025.\n")
        claims = extract_claims(str(doc), llm=None)
        assert len(claims) >= 1

    def test_llm_filter_applied_before_prioritization(
        self, tmp_path: Path,
    ) -> None:
        """LLM filter runs before max_claims prioritization."""
        doc = tmp_path / "test.md"
        lines = [
            f"Revenue reached ${i}M in 2025.\n"
            for i in range(1, 21)
        ]
        doc.write_text("# Report\n\n" + "".join(lines))

        # LLM keeps only 5 claims
        mock_llm = MagicMock()
        kept = [
            _claim_json(f"Revenue reached ${i}M in 2025.")
            for i in range(1, 6)
        ]
        mock_llm.call.return_value = json.dumps(kept)

        claims = extract_claims(str(doc), max_claims=3, llm=mock_llm)
        # LLM returned 5, prioritization caps to 3
        assert len(claims) == 3


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


# --- SemanticVerifier tests ---


class TestSemanticVerifier:
    """Tests for the LLM-based semantic verification tool."""

    _CLAIM = "CrewAI has 44K GitHub stars."
    _EVIDENCE = (
        "CrewAI is a popular open-source framework. "
        "As of January 2026, the repository shows 44,000 stars."
    )

    def test_supports_verdict(self) -> None:
        mock_llm = MagicMock()
        mock_llm.call.return_value = json.dumps({
            "verdict": "SUPPORTS",
            "confidence": 0.95,
            "reasoning": "Evidence confirms 44K stars.",
        })
        result = verify_semantically(
            self._CLAIM, self._EVIDENCE, mock_llm,
        )
        assert result["verdict"] == "SUPPORTS"
        assert result["confidence"] == 0.95

    def test_contradicts_verdict(self) -> None:
        mock_llm = MagicMock()
        mock_llm.call.return_value = json.dumps({
            "verdict": "CONTRADICTS",
            "confidence": 0.9,
            "reasoning": "Page says 10K stars, not 44K.",
        })
        result = verify_semantically(
            self._CLAIM, "The repo has 10K stars.", mock_llm,
        )
        assert result["verdict"] == "CONTRADICTS"

    def test_insufficient_verdict(self) -> None:
        mock_llm = MagicMock()
        mock_llm.call.return_value = json.dumps({
            "verdict": "INSUFFICIENT",
            "confidence": 0.3,
            "reasoning": "Page mentions CrewAI but no star count.",
        })
        result = verify_semantically(
            self._CLAIM, "CrewAI is a framework.", mock_llm,
        )
        assert result["verdict"] == "INSUFFICIENT"

    def test_fallback_on_llm_error(self) -> None:
        mock_llm = MagicMock()
        mock_llm.call.side_effect = RuntimeError("API error")
        result = verify_semantically(
            self._CLAIM, self._EVIDENCE, mock_llm,
        )
        assert result["verdict"] == "INSUFFICIENT"
        assert result["confidence"] == 0.0

    def test_fallback_on_invalid_json(self) -> None:
        mock_llm = MagicMock()
        mock_llm.call.return_value = "Not JSON at all"
        result = verify_semantically(
            self._CLAIM, self._EVIDENCE, mock_llm,
        )
        assert result["verdict"] == "INSUFFICIENT"
        assert result["confidence"] == 0.0

    def test_empty_evidence(self) -> None:
        mock_llm = MagicMock()
        result = verify_semantically(
            self._CLAIM, "", mock_llm,
        )
        assert result["verdict"] == "INSUFFICIENT"
        mock_llm.call.assert_not_called()

    def test_strips_markdown_fences(self) -> None:
        mock_llm = MagicMock()
        fenced = (
            "```json\n"
            '{"verdict": "SUPPORTS", "confidence": 0.9, '
            '"reasoning": "Confirmed."}\n'
            "```"
        )
        mock_llm.call.return_value = fenced
        result = verify_semantically(
            self._CLAIM, self._EVIDENCE, mock_llm,
        )
        assert result["verdict"] == "SUPPORTS"

    def test_invalid_verdict_falls_back(self) -> None:
        mock_llm = MagicMock()
        mock_llm.call.return_value = json.dumps({
            "verdict": "MAYBE",
            "confidence": 0.5,
            "reasoning": "Unclear.",
        })
        result = verify_semantically(
            self._CLAIM, self._EVIDENCE, mock_llm,
        )
        assert result["verdict"] == "INSUFFICIENT"

    def test_confidence_clamped(self) -> None:
        mock_llm = MagicMock()
        mock_llm.call.return_value = json.dumps({
            "verdict": "SUPPORTS",
            "confidence": 5.0,
            "reasoning": "Very confident.",
        })
        result = verify_semantically(
            self._CLAIM, self._EVIDENCE, mock_llm,
        )
        assert result["confidence"] <= 1.0

    def test_url_passed_through(self) -> None:
        mock_llm = MagicMock()
        mock_llm.call.return_value = json.dumps({
            "verdict": "SUPPORTS",
            "confidence": 0.9,
            "reasoning": "OK.",
        })
        result = verify_semantically(
            self._CLAIM, self._EVIDENCE, mock_llm,
            url="https://github.com/crewAIInc/crewAI",
        )
        assert result["url"] == "https://github.com/crewAIInc/crewAI"

    # --- Tool interface tests ---

    def test_tool_interface(self) -> None:
        mock_llm = MagicMock()
        mock_llm.call.return_value = json.dumps({
            "verdict": "SUPPORTS",
            "confidence": 0.9,
            "reasoning": "OK.",
        })
        tool = SemanticVerifierTool(llm=mock_llm)
        result = tool._run(json.dumps({
            "claim": self._CLAIM,
            "evidence": self._EVIDENCE,
        }))
        parsed = json.loads(result)
        assert parsed["verdict"] == "SUPPORTS"

    def test_tool_no_llm(self) -> None:
        tool = SemanticVerifierTool(llm=None)
        result = tool._run(json.dumps({
            "claim": self._CLAIM,
            "evidence": self._EVIDENCE,
        }))
        parsed = json.loads(result)
        assert parsed["verdict"] == "INSUFFICIENT"
        assert "No LLM" in parsed["reasoning"]

    def test_tool_invalid_input(self) -> None:
        tool = SemanticVerifierTool(llm=MagicMock())
        result = tool._run("not json")
        parsed = json.loads(result)
        assert "error" in parsed
