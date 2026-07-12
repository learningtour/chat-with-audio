# Chat with Audio — Documentation

Chat with Audio is a chat-driven audio post-production toolkit: you talk to
Claude (or any MCP client) about a recording, and the toolkit analyzes,
repairs, enhances, masters and QC-checks it. A local A/B viewer shows and
plays every intervention; every operation is a fully documented session.

| Guide | What it covers |
|---|---|
| [Getting started](getting-started.md) | Install, register with Claude/Codex, first session |
| [Tool reference](tools.md) | All 30 MCP tools: parameters, returns, examples |
| [Workflows cookbook](workflows.md) | Podcast, broadcast delivery, film dialogue, music, restoration, batch |
| [Delivery compliance](compliance.md) | EBU R128, ATSC A/85, Netflix, streaming and ACX specs — how checking and mastering work |
| [Smart regions](smart-regions.md) | How the AI finds problem regions and treats only those |
| [Recipes](recipes.md) | Saving, applying and sharing processing chains |
| [Architecture](architecture.md) | DSP core, analysis, session model, viewer, MCP layer |
| [Roadmap & gap analysis](roadmap.md) | Full post-production taxonomy vs current tools, phased build plan |

## The one-paragraph pitch

Most audio tools make you choose between a DAW (powerful, manual) and an
online "enhancer" (automatic, opaque). Chat with Audio is a third thing: you
describe the outcome in plain language — *"fix it only where something is
wrong"*, *"make this broadcast-compliant"*, *"save this as my podcast
preset"* — and the toolkit runs measurable, explainable DSP: every step has a
written rationale, every result is A/B-comparable with a residual listen
(hear exactly what changed), and every claim is checkable in the session log.

## Design principles

1. **Do no harm** — surgical tools leave everything outside their target
   bit-for-bit untouched; conservative defaults everywhere.
2. **Explain everything** — every chain ships a rationale; every session has
   a full provenance log (`log.md`).
3. **Measure, don't vibe** — decisions are driven by BS.1770 loudness, SNR,
   spectral analysis and Whisper word-retention, not by "sounds about right".
4. **Interoperate** — sessions export to Audition, markers to any DAW,
   recipes as shareable JSON, delivery masters as 48 kHz/24-bit WAV.
