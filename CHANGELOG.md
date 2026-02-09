# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.3.0] - 2026-02-09

### Added

- **Knowledge database** (`db.py`): SQLite-backed persistence with 8 tables
  (`runs`, `phase_metrics`, `token_usage`, `claims`, `claim_sources`,
  `claim_history`, `sources`, `events`), FTS5 virtual table with auto-sync
  triggers, WAL mode, schema versioning with migration support.
- **Claim extraction guardrails**: Two-stage pipeline with pre-filter heuristics
  (rejects markdown noise, bibliography entries, incomplete sentences) and LLM
  batch filter (validates quality, decomposes compound claims).
- **Semantic verifier tool** (`semantic_verifier.py`): LLM-based evidence
  evaluation returning SUPPORTS/CONTRADICTS/INSUFFICIENT/UNRELATED with
  confidence and reasoning.
- **Dual-write strategy**: AuditLogger, SourceTracker, and SourceExtractor
  write to both SQLite and legacy JSONL files. DB failures never break the pipeline.
- **Full DB integration in flow_builder**: Opens DB at pipeline start, records
  run/phase metrics/token usage, queries prior knowledge via FTS5, closes DB
  at the end.
- **CrewAI features unlocked**:
  - `memory=True` with ONNX embedder on all crews (local, no API key).
  - `Process.hierarchical` with Research Director for DEEP depth.
  - `reasoning=True` and `allow_delegation=True` for DEEP agents.
  - Report guardrail on verification task.
  - `step_callback` and `task_callback` forwarding.
  - Token tracking via `CrewOutput.token_usage`.
- **Verification crew enhancements**: SemanticVerifierTool, prior claims
  context from FTS5, stale claims context for re-verification.
- **Synthesis crew enhancements**: Claims context from DB, Perplexity-style
  inline citations (`[1]`, `[2]`), numbered Sources section.
- **4 new CLI commands**:
  - `recon claims` -- Browse verified claims (--run, --status, --search, --limit).
  - `recon history <id>` -- Show verification history for a claim.
  - `recon stats` -- Global or per-run statistics (--run).
  - `recon reverify` -- List stale claims needing re-verification.
- **Token tracking + cost estimation**: `PROVIDER_PRICING` dict and
  `estimate_cost()` function in config.py.
- **Progress UI enhancements**: Live verification metrics during verification
  phase, knowledge DB summary at pipeline end (claims, tokens, cost, sources).
- **KnowledgeConfig**: Replaces MemoryConfig with backward-compatible migration
  from legacy `memory:` YAML key. Always-on by default.
- 401 tests across 10 test files (up from 154).

### Changed

- `config.py`: `KnowledgeConfig` replaces `MemoryConfig` (alias kept).
  New fields: `db_path`, `embedder`, `stale_after_days`.
- `flow_builder.py`: All `plan.memory.*` references changed to `plan.knowledge.*`.
  Prior knowledge query uses FTS5 instead of memvid.
- Investigation crew: Sequential for QUICK/STANDARD, hierarchical for DEEP.
- Verification crew: Added SemanticVerifierTool to verification tools.

### Removed

- `memory/` module (memvid `.mv2` wrapper) -- replaced by knowledge database.
- Legacy `recon memory stats` and `recon memory query` CLI subcommands.
- `_ingest_to_memory()` function from flow_builder.
- memvid fallback in `_query_prior_knowledge()`.

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
