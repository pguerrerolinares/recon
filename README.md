# Recon

[![CI](https://github.com/pguerrerolinares/recon/actions/workflows/ci.yaml/badge.svg)](https://github.com/paulofcardoso/recon/actions)
[![PyPI](https://img.shields.io/pypi/v/recon-ai)](https://pypi.org/project/recon-ai/)
[![Python](https://img.shields.io/pypi/pyversions/recon-ai)](https://pypi.org/project/recon-ai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**AI research that tells you what it got wrong.**

Most AI research tools generate impressive-looking reports. None of them tell
you which parts are actually true. Recon does.

Recon runs parallel AI agents to investigate a topic from multiple angles, then
**verifies every factual claim** against its cited source. The output is a
research report where each claim is marked as verified, partially verified,
unverifiable, or contradicted -- with a confidence score.

## What you get

```
recon run --topic "AI agent frameworks in 2026" --depth deep
```

```
research/               <- raw investigation reports (one per angle)
verification/
  report.md             <- claim-level fact-check: status + confidence
  audit-trail.jsonl     <- machine-readable provenance per claim
output/
  final-report.md       <- synthesized report, confidence-weighted
```

From a real run (MCP ecosystem research):
- 19 claims extracted, 13 verified, 0 contradicted
- Overall reliability score: 71.1%
- Each claim linked to its source URL + evidence excerpt

## Install

Python 3.12+.

```bash
pip install recon-ai
```

## Setup

You need one LLM API key and one search API key:

```bash
cp .env.example .env
# Set at minimum: OPENROUTER_API_KEY + TAVILY_API_KEY
```

## Usage

```bash
# Research a topic directly
recon run --topic "Your research question" --depth standard

# Use a plan file for more control
recon init --template market-research
# Edit plan.yaml, then:
recon run plan.yaml

# Fact-check existing research
recon verify ./research/

# Check what happened in a previous run
recon status
```

### Depth levels

| Depth | Agents | Best for |
|-------|--------|----------|
| `quick` | 1 | Fast answers, single-angle |
| `standard` | 3 | Balanced research |
| `deep` | 5 | Thorough multi-perspective analysis |

## How it works

```mermaid
graph LR
    P[plan.yaml] --> I

    subgraph Pipeline
        I["Investigation\n1-5 parallel agents"] --> V["Verification\n5 tools, zero LLM cost"]
        V --> S["Synthesis\nconfidence-weighted report"]
    end

    I --> R["research/*.md"]
    V --> VR["verification/report.md"]
    S --> F["output/final-report.md"]
```

The verification phase uses 5 deterministic tools (regex extraction, HTTP
fetching, term matching) -- no LLM calls. Fact-checking adds compute time
but zero additional API cost.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical breakdown.

## Providers

Works with any OpenAI-compatible LLM and 4 search APIs:

| LLM | Search |
|-----|--------|
| OpenRouter (default), Gemini, Groq, Kimi, Anthropic, OpenAI, Ollama, Custom | Tavily (default), Brave, Serper, Exa |

```bash
recon run --topic "X" --provider anthropic --model claude-sonnet-4
```

## Cross-run memory

Optional. Recon can carry knowledge between runs using [memvid](https://github.com/memvid/memvid):

```bash
pip install recon-ai[memory]
recon run --topic "AI frameworks" --memory ./memory/recon.mv2
```

Second run on a related topic starts with context from previous research.
Everything is still verified independently.

## Plan file

Simple:
```yaml
topic: "AI agent frameworks in 2026"
questions:
  - "What frameworks exist and what is their adoption?"
  - "What gaps and opportunities remain?"
depth: deep
verify: true
```

Advanced plans support custom investigations, provider overrides, verification
thresholds, and synthesis instructions. See `examples/` and
[ARCHITECTURE.md](ARCHITECTURE.md#plan-file-format).

## Templates

```bash
recon templates                              # list available
recon init --template market-research        # market/competitive analysis
recon init --template competitive-analysis   # deep competitor comparison
recon init --template technical-landscape    # technology survey
recon init --template opportunity-finder     # gap analysis + ideas
```

## Docker

```bash
docker compose up
# or
docker run -v $(pwd):/workspace \
  -e OPENROUTER_API_KEY=sk-... \
  -e TAVILY_API_KEY=tvly-... \
  recon-ai run /workspace/plan.yaml
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
