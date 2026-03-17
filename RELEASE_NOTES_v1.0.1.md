# semantic-memory-mcp-server v1.0.1

Follow-up release aligning the public tag snapshot with the current `main` branch.

## What Changed

- Realigns the tagged source snapshot with the current public landing-page state
- Includes the tightened release-note copy now present on `main`
- Preserves the same runtime surface, benchmark artifacts, and federated MCP server behavior as `v1.0.0`

## Scope

This is a release-coherence update, not a feature expansion.

Runtime API, benchmark artifacts, and federated store behavior are unchanged from `v1.0.0`.

## Background

After the initial `v1.0.0` tag was pushed, two additional commits landed on `main`:

1. `Merge remote bootstrap commit` — repository history reconciliation
2. `Tighten v1.0.0 release notes` — release-copy polish
3. `Add v1.0.1 tag plan` — release engineering documentation

None of these commits introduced a runtime feature or changed any benchmark artifact.
The `v1.0.1` tag exists solely to give the public release snapshot a clean match with the current `main` branch state.

## Benchmark Snapshot (unchanged from v1.0.0)

- Storage-shaped RWIF vs dense cosine benchmark: `100.0%` top-1 parity, RWIF size ratio `0.547` versus dense float32 and `0.273` versus dense float64
- Small real text array benchmark: `100.0%` RWIF top-1 accuracy, `100.0%` dense cosine top-1 accuracy, `100.0%` full-ranking agreement
- Larger shipped text corpus benchmark: `24` records, `12` queries, `100.0%` top-1 parity, mean Spearman rank correlation `0.813`, top-10 overlap `82.5%`
