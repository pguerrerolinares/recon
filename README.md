# Recon

[![CI](https://github.com/paulofcardoso/recon/actions/workflows/ci.yaml/badge.svg)](https://github.com/paulofcardoso/recon/actions)
[![PyPI](https://img.shields.io/pypi/v/recon-ai)](https://pypi.org/project/recon-ai/)
[![Python](https://img.shields.io/pypi/pyversions/recon-ai)](https://pypi.org/project/recon-ai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Verified research pipelines powered by AI agents.**

Recon turns any topic into a multi-perspective, fact-checked research report.
It runs specialized AI agents in parallel to investigate different angles,
verifies every factual claim against independent sources, and synthesizes
everything into a trustworthy final report.

Built on [CrewAI](https://crewai.com). No agent configuration needed.

## Quick Start

Requires **Python 3.12+**.

```bash
pip install recon-ai
```

```bash
# Research a topic with built-in fact-checking
recon run --topic "AI agent frameworks in 2026" --depth deep --verify

# Or use a plan file
recon init --template market-research
# Edit plan.yaml with your topic, then:
recon run plan.yaml --verbose
```

You need at least one LLM API key and one search API key. Copy `.env.example`
to `.env` and fill in your keys:

```bash
cp .env.example .env
# Edit .env -- at minimum set OPENROUTER_API_KEY and TAVILY_API_KEY
```

## What Makes Recon Different

Every deep research tool generates impressive reports. None of them tell you
which claims are actually verified.

Recon produces a **verification report** alongside the research, where every
factual claim is marked as:

- **VERIFIED** -- confirmed by an independent source
- **PARTIALLY VERIFIED** -- related evidence found, not exact match
- **UNVERIFIABLE** -- no confirming or contradicting evidence
- **CONTRADICTED** -- evidence contradicts the claim

Each claim gets a confidence score (0.0-1.0) based on verification status,
source count, and whether a primary source was found.

## How It Works

```
plan.yaml --> Phase 1: Investigation (parallel agents search the web)
          --> Phase 2: Verification  (claims extracted, checked against sources)
          --> Phase 3: Synthesis     (confidence-weighted final report)
```

**Phase 1 -- Investigation**: Parallel CrewAI researcher agents run web
searches, read pages, and produce structured markdown reports in `research/`.
The number of agents depends on depth: quick (1), standard (3), deep (5).

**Phase 2 -- Verification**: A fact-checker agent extracts verifiable claims
(statistics, dates, pricing, attributions) using regex heuristics (zero LLM
cost for extraction). Each claim with a cited URL is verified by fetching the
source and matching key terms. Claims are scored and marked with a status.
Output: `verification/report.md`.

**Phase 3 -- Synthesis**: A director agent reads all research files plus the
verification report and produces `output/final-report.md` with
confidence-weighted findings, areas of uncertainty, and ranked recommendations.

Output structure:
```
research/          <- individual investigation reports
verification/      <- fact-check report with claim-level status
  report.md        <- verification summary + detailed claim table
  audit-trail.jsonl <- per-claim provenance (source URL, status, score)
output/            <- final report + pipeline audit
  final-report.md  <- synthesized, confidence-weighted report
  audit-log.jsonl  <- pipeline-level provenance (phase timings)
```

## CLI Reference

### `recon run`

Run the full 3-phase research pipeline.

```bash
recon run plan.yaml                          # from a plan file
recon run --topic "My research topic"        # inline mode
recon run plan.yaml --verbose                # show detailed CrewAI output
recon run plan.yaml --dry-run                # validate plan, don't execute
recon run plan.yaml --no-verify              # skip verification phase
recon run --topic "X" --depth deep           # quick | standard | deep
recon run --topic "X" --provider anthropic --model claude-sonnet-4
```

### `recon init`

Create a new plan file from a template.

```bash
recon init                                   # default: market-research
recon init --template competitive-analysis
recon init --template technical-landscape
recon init --template opportunity-finder
recon init --output my-plan.yaml             # custom output path
```

### `recon templates`

List all available plan templates.

```bash
recon templates
```

### `recon verify`

Run standalone fact-checking on existing research files.

```bash
recon verify ./research/                     # verify files in ./research/
recon verify ./research/ --output ./checks/  # custom output directory
```

### `recon status`

Show the status of a previous pipeline execution.

```bash
recon status                                 # reads ./output/audit-log.jsonl
recon status ./my-output/                    # custom output directory
```

### `recon rerun`

Re-run a specific phase using an existing plan.

```bash
recon rerun plan.yaml --phase verification   # re-run just verification
recon rerun plan.yaml --phase synthesis      # re-generate the final report
recon rerun plan.yaml --phase investigation  # re-run all researchers
```

## Plan File Format

### Simple Mode

```yaml
topic: "AI agent frameworks and tooling in 2026"
questions:
  - "What frameworks exist and what is their adoption?"
  - "What business models work in this space?"
  - "What gaps and opportunities remain?"
focus: "Open-source projects viable for a solo developer"
depth: deep       # quick (1 agent) | standard (3) | deep (5)
verify: true      # enable fact-checking phase
```

### Advanced Mode

```yaml
topic: "Competitive analysis of vector databases"
depth: standard
verify: true
provider: kimi
model: kimi-k2.5

search:
  provider: tavily

investigations:
  - id: features
    name: "Feature Comparison"
    questions:
      - "What features does each database offer?"
    instructions: "Create a comparison table with benchmarks."

  - id: pricing
    name: "Pricing Analysis"
    questions:
      - "What are the pricing models?"
    instructions: "Include free tier details."

verification:
  min_confidence: 0.6               # flag claims below this threshold (0.0-1.0)
  require_primary_source: false      # flag claims without primary sources
  max_queries_per_claim: 2           # max search queries per claim (1-10)
  max_fetches_per_claim: 2           # max URL fetches per claim (1-10)
  timeout_per_fetch: 10              # seconds per URL fetch (1-60)

synthesis:
  instructions: "Rank databases by developer experience."

output_dir: ./output
research_dir: ./research
verification_dir: ./verification
```

See `examples/` for more plan files.

## Templates

```bash
recon templates                              # list available templates
recon init --template market-research        # competitive/market analysis
recon init --template competitive-analysis   # deep competitor comparison
recon init --template technical-landscape    # technology/framework survey
recon init --template opportunity-finder     # gap analysis + ideas
```

## LLM Providers

Recon works with any OpenAI-compatible provider:

| Provider | Config | API Key Env Var | Free tier |
|----------|--------|-----------------|-----------|
| OpenRouter (default) | `provider: openrouter` | `OPENROUTER_API_KEY` | 25+ free models |
| Google Gemini | `provider: gemini` | `GEMINI_API_KEY` | Generous free tier |
| Groq | `provider: groq` | `GROQ_API_KEY` | Free, fast inference |
| Kimi K2.5 | `provider: kimi` | `KIMI_API_KEY` | Free tier |
| Anthropic Claude | `provider: anthropic` | `ANTHROPIC_API_KEY` | No |
| OpenAI | `provider: openai` | `OPENAI_API_KEY` | No |
| Ollama (local) | `provider: ollama` | -- | Unlimited |
| Custom | `provider: custom` | `RECON_API_KEY` | -- |

## Search Providers

| Provider | Config | API Key Env Var | Free tier |
|----------|--------|-----------------|-----------|
| Tavily (default) | `search.provider: tavily` | `TAVILY_API_KEY` | 1000 calls/month |
| Brave Search | `search.provider: brave` | `BRAVE_API_KEY` | 2000 calls/month |
| Serper | `search.provider: serper` | `SERPER_API_KEY` | 2500 calls/month |
| Exa | `search.provider: exa` | `EXA_API_KEY` | Limited free |

## Environment Variables

Copy `.env.example` to `.env` and set the keys for your chosen providers.
At minimum you need one LLM key and one search key.

```bash
cp .env.example .env
```

You can also override defaults without editing plan files:

```bash
export RECON_PROVIDER=anthropic
export RECON_MODEL=claude-sonnet-4
```

See `.env.example` for all available variables and signup URLs.

## Docker

### Docker Compose (recommended)

```bash
# Place your plan file in ./plans/plan.yaml
mkdir -p plans output research verification
cp plan.yaml plans/plan.yaml

docker compose up
```

### Docker Run

```bash
docker run -v $(pwd):/workspace \
  -e OPENROUTER_API_KEY=sk-... \
  -e TAVILY_API_KEY=tvly-... \
  recon-ai run /workspace/plan.yaml
```

## Examples

The `examples/` directory contains ready-to-use plan files:

| File | Description |
|------|-------------|
| `simple.yaml` | Basic topic research with standard depth |
| `advanced.yaml` | Custom investigations, provider override, verification tuning |
| `verify-only.yaml` | Standalone verification on existing research files |

```bash
# Try the simple example
recon run examples/simple.yaml --verbose
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and
PR guidelines.

## License

[MIT](LICENSE)
