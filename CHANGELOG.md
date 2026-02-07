# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-02-08

### Added

- 3-phase research pipeline: Investigation, Verification, Synthesis.
- 6 CLI commands: `run`, `init`, `templates`, `verify`, `status`, `rerun`.
- 9 LLM providers (OpenRouter, Anthropic, OpenAI, Gemini, Groq, Kimi, Ollama, Copilot, custom).
- 4 search providers (Tavily, Brave, Serper, Exa).
- 5 custom verification tools:
  - `claim_extractor` -- regex-based claim extraction (zero LLM cost).
  - `citation_verifier` -- fetch + match claims against source URLs.
  - `confidence_scorer` -- deterministic 0.0-1.0 scoring.
  - `source_tracker` -- JSONL audit trail per claim.
  - `contradiction_detector` -- cross-source consistency check.
- Rich Live TUI with real-time progress display during pipeline execution.
- JSONL audit logging for pipeline-level and claim-level provenance.
- Context window management with auto strategy selection (direct/summarize/map_reduce).
- 4 plan templates: market-research, competitive-analysis, technical-landscape, opportunity-finder.
- Inline topic mode (`recon run --topic "..."`) and plan file mode.
- Dry-run validation (`recon run --dry-run`).
- Standalone verification (`recon verify ./research/`).
- Phase-level re-execution (`recon rerun plan.yaml --phase verification`).
- Pipeline status viewer (`recon status`).
- Docker and Docker Compose support.
- GitHub Actions CI (lint, format, test).
- 154 tests across 7 test files.
