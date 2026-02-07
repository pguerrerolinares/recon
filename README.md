# Recon

**Verified research pipelines powered by AI agents.**

Recon turns any topic into a multi-perspective, fact-checked research report.
It runs specialized AI agents in parallel to investigate different angles,
verifies every factual claim against independent sources, and synthesizes
everything into a trustworthy final report.

Built on [CrewAI](https://crewai.com). No agent configuration needed.

## Quick Start

```bash
pip install recon-ai
```

```bash
# Research a topic with built-in fact-checking
recon run --topic "AI agent frameworks in 2026" --depth deep --verify

# Or use a plan file
recon init --template market-research
# Edit plan.yaml with your topic
recon run plan.yaml
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

## How It Works

```
plan.yaml --> Phase 1: Investigation (parallel agents)
          --> Phase 2: Verification (fact-checking)
          --> Phase 3: Synthesis (verified final report)
```

Output:
```
research/          <- individual investigation reports
verification/      <- fact-check report with claim-level status
output/            <- final synthesized report
```

## Plan File (Simple Mode)

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

| Provider | Config | Free tier |
|----------|--------|-----------|
| OpenRouter (default) | `provider: openrouter` | 25+ free models |
| Google Gemini | `provider: gemini` | Generous free tier |
| Groq | `provider: groq` | Free, fast inference |
| Anthropic Claude | `provider: anthropic` | No |
| OpenAI | `provider: openai` | No |
| Kimi K2.5 | `provider: kimi` | Not verified |
| Ollama (local) | `provider: ollama` | Unlimited |

## Docker

```bash
docker run -v $(pwd):/workspace \
  -e OPENROUTER_API_KEY=sk-... \
  -e TAVILY_API_KEY=tvly-... \
  recon-ai run /workspace/plan.yaml
```

## License

MIT
