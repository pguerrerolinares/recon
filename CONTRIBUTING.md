# Contributing to Recon

Thanks for your interest in contributing. This document covers how to set up
a development environment, run tests, and submit changes.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/paulofcardoso/recon.git
cd recon

# Create a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys (needed only for e2e tests)
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_cli.py -v

# Run with output
pytest tests/ -v -s
```

All tests mock external API calls. No API keys are needed to run the test suite.

## Linting and Formatting

```bash
# Check for lint errors
ruff check src/ tests/

# Auto-fix import sorting and safe fixes
ruff check --fix src/ tests/

# Check formatting
ruff format --check src/ tests/

# Apply formatting
ruff format src/ tests/
```

CI runs `ruff check`, `ruff format --check`, and `pytest` on every push
and pull request.

## Code Structure

```
src/recon/
  __init__.py          # Version
  config.py            # Pydantic models: ReconPlan, Depth, Investigation
  cli.py               # Typer CLI: run, init, templates, verify, status, rerun
  flow_builder.py      # Pipeline orchestrator (3-phase execution)
  providers/
    llm.py             # LLM factory (9 OpenAI-compatible providers)
    search.py           # Search tool factory (4 providers)
  crews/
    investigation/      # Parallel researcher agents + config YAMLs
    verification/       # Fact-checking agent + 5 custom tools
    synthesis/          # Director agent for final report
  tools/
    claim_extractor.py  # Regex-based claim extraction (zero LLM cost)
    citation_verifier.py # Fetch + match claims against source URLs
    confidence_scorer.py # Deterministic 0.0-1.0 scoring
    source_tracker.py    # JSONL audit trail per claim
    contradiction_detector.py # Cross-source consistency check
  callbacks/
    progress.py         # Rich Live TUI with spinners
    audit.py            # JSONL pipeline audit logger
  context/
    strategy.py         # Token counting + context window management
  flows/
    research_flow.py    # CrewAI Flow with state (alternative to flow_builder)
  templates/            # 4 YAML plan templates
tests/
  conftest.py           # Shared fixtures
  test_config.py        # Plan models and YAML loading
  test_cli.py           # All 6 CLI commands
  test_providers.py     # LLM + search provider factories
  test_templates.py     # Template validation
  test_tools.py         # All 5 verification tools
  test_flow.py          # Audit logger, progress tracker, flow builder
  test_context.py       # Token counting and strategy selection
```

## Making Changes

1. Create a branch from `main`.
2. Make your changes.
3. Run `ruff check src/ tests/` and `ruff format src/ tests/`.
4. Run `pytest tests/ -v` and ensure all tests pass.
5. Open a pull request with a clear description of what changed and why.

## Adding a New Provider

LLM providers: edit `src/recon/providers/llm.py`. Add the base URL to
`PROVIDER_BASE_URLS` and the env var name to `PROVIDER_API_KEY_ENV`.

Search providers: edit `src/recon/providers/search.py`. Add the env var to
`SEARCH_API_KEY_ENV` and handle the provider in `create_search_tools()`.

## Adding a New Tool

Create a new file in `src/recon/tools/`. Follow the pattern of existing tools
(class with a `run()` method). Add it to `src/recon/tools/__init__.py` and
write tests in `tests/test_tools.py`.

## Adding a Template

Add a `.yaml` file to `src/recon/templates/`. Start with a `#` comment
explaining the template. Include `topic:` (placeholder), `depth:`, and either
`questions:` or `investigations:`. The template tests will automatically
pick it up.

## Commit Messages

Use concise messages that explain **why**, not just what:

```
Add contradiction detection for cross-source claims
Fix Kimi provider base URL (was pointing to China endpoint)
```

## Questions

Open an issue on GitHub for bugs, feature requests, or questions.
