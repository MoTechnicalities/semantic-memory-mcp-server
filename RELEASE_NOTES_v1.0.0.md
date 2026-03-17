# semantic-memory-mcp-server v1.0.0

RWIF turns semantic memory into compact, portable files that stay on disk instead of requiring a separate always-on vector database tier. This first public export release packages that storage model behind an MCP server with federated routing, governed write policy, Docker examples, and published benchmark artifacts.

## Highlights

- RWIF-backed MCP semantic-memory server export with a focused public package layout
- Federated multi-store config with `store_roots`, portable relative paths, and per-store `read-only` or `read-write` access policy
- MCP tools for semantic retrieval, grounded answers, reasoning, store inspection, and governed proposal workflows
- Docker, Docker Compose, and Docker Desktop examples for read-only and selectively writable deployments
- Published benchmark artifacts that cover:
  - a storage-shaped RWIF vs dense cosine comparison
  - a small workload-shaped real text array comparison
  - a larger shipped text-corpus comparison with deterministic text-derived activations

## Benchmark Record

The shipped benchmark record supports these claims:

- RWIF retrieval is operational and measurable on the shipped micro-benchmark.
- RWIF stays in top-1 agreement with dense cosine on the shipped real text-derived array benchmark.
- RWIF also stays in top-1 agreement with dense cosine on the larger shipped 24-record, 12-query text corpus benchmark, though deeper rankings diverge and are reported explicitly.
- The repository includes a grounded semantic-memory demo and a broader verifier showcase snapshot from development validation.

Key published measurements from the comparative benchmarks:

- Synthetic storage-shaped benchmark:
  - RWIF top-1 accuracy: `100.0%`
  - Dense cosine top-1 accuracy: `100.0%`
  - RWIF size ratio versus dense float32: `0.547`
  - RWIF size ratio versus dense float64: `0.273`
- Small real text array benchmark:
  - RWIF top-1 accuracy: `100.0%`
  - Dense cosine top-1 accuracy: `100.0%`
  - Full-ranking agreement rate: `100.0%`
- Larger shipped text corpus benchmark:
  - Records: `24`
  - Queries: `12`
  - RWIF top-1 accuracy: `100.0%`
  - Dense cosine top-1 accuracy: `100.0%`
  - Mean Spearman rank correlation: `0.813`
  - Top-10 overlap rate: `82.5%`

## Operational Notes

- Write access is enforced as a real local capability. Stores marked `read-only` reject mutations, and multiple writable stores require explicit targeting.
- The Docker Desktop examples separate read-only mounting from selectively writable mounting so storage policy is visible in deployment configuration rather than hidden in convention.
- Benchmark claims are intentionally conservative. This release does not claim universal superiority over vector databases or broad latency wins across arbitrary corpora.

## Included Assets

- Standalone source tree under `src/`
- Focused federation tests under `tests/`
- Sample configs and sample data under `sample_data/`
- Benchmark generation scripts under `scripts/`
- Published benchmark outputs under `benchmark_results/`
- CI, Docker assets, security guidance, release checklist, and tag notes

## Suggested GitHub Release Title

`semantic-memory-mcp-server v1.0.0`

## Suggested Short Release Summary

First public export release of the RWIF-backed semantic-memory MCP server with federated config, governed write policy, Docker deployment examples, and published comparative benchmark artifacts.