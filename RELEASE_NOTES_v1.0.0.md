# semantic-memory-mcp-server v1.0.0

First public export release of the RWIF-backed semantic-memory MCP server.

RWIF keeps semantic memory in compact disk-native files and exposes it through one MCP surface, so retrieval can be deployed as a file-and-router problem instead of a heavyweight always-on vector database tier.

## Highlights

- RWIF-backed MCP semantic-memory server with a focused standalone export layout
- Federated multi-store config with portable roots, relative store paths, and per-store `read-only` or `read-write` policy
- Retrieval, reasoning, store-inspection, and governed proposal tools over one MCP surface
- Docker, Docker Compose, and Docker Desktop examples for read-only and selectively writable deployments
- Published benchmark artifacts included directly in the repository

## Benchmark Snapshot

- Storage-shaped RWIF vs dense cosine benchmark: `100.0%` top-1 parity, RWIF size ratio `0.547` versus dense float32 and `0.273` versus dense float64
- Small real text array benchmark: `100.0%` RWIF top-1 accuracy, `100.0%` dense cosine top-1 accuracy, `100.0%` full-ranking agreement
- Larger shipped text corpus benchmark: `24` records, `12` queries, `100.0%` top-1 parity, mean Spearman rank correlation `0.813`, top-10 overlap `82.5%`

## Release Notes

- Write access is enforced as a real capability, not a cosmetic flag
- Stores marked `read-only` reject mutations, and multiple writable stores require explicit targeting
- Benchmark claims remain conservative and do not claim universal superiority over vector databases or broad latency wins across arbitrary corpora

## Included In This Release

- Source tree under `src/`
- Focused federation tests under `tests/`
- Sample configs and sample data under `sample_data/`
- Benchmark scripts under `scripts/`
- Published benchmark outputs under `benchmark_results/`
- CI, Docker assets, security guidance, release checklist, and tag preparation notes