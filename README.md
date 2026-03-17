# Semantic Memory MCP Server

RWIF turns semantic memory into compact, portable files that can stay on disk instead of forcing a heavyweight vector database to stay resident in RAM or VRAM. This repository exposes that disk-native memory through a clean MCP server, so models and agents can retrieve grounded semantic evidence from RWIF files on SSDs, removable media, or mixed storage tiers without introducing a separate database platform.

The practical implication is straightforward: semantic retrieval becomes a file-and-router problem rather than a cluster-and-index problem. You can keep always-on knowledge on local SSD, mount private knowledge only when needed, and publish one stable MCP surface over all of it.

## Why This Repo Exists

- Replace the usual vector-database control plane with RWIF-backed semantic memory files.
- Route across multiple stores through one MCP tool surface.
- Keep storage policy explicit with per-store `read-only` and `read-write` modes.
- Allow semantic memory to live on normal filesystems, including removable or encrypted media.
- Preserve provenance, revisions, contradictions, and governed write workflows.

## What You Get

- MCP tools for `memory_query`, `memory_answer`, `memory_reason`, and store management.
- Federated routing across multiple RWIF semantic-memory stores.
- Store configuration via JSON with portable root directories and relative store paths.
- Per-store mutation policy enforced at the broker layer.
- Docker, Docker Compose, and Docker Desktop example configurations.
- A small demo RWIF artifact in [sample_data/demo_readme_semantic_store.rwif](sample_data/demo_readme_semantic_store.rwif).

## Repository Layout

See [REPOSITORY_LAYOUT.md](REPOSITORY_LAYOUT.md) for the cleaned export tree.

## Release Ops

Use [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) before pushing to a public remote or cutting a release. Use [TAG_PREPARATION.md](TAG_PREPARATION.md) when preparing the first version tag.

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for the initial release benchmark record. The current benchmark page includes:

- the RWIF retrieval micro-benchmark
- a dedicated RWIF vs dense cosine storage-method benchmark
- a dedicated RWIF vs dense cosine workload-shaped benchmark on a real text-derived array dataset
- a larger shipped text-corpus benchmark with deterministic text-derived activations
- the README-grounded semantic memory demo snapshot
- the broader semantic verifier showcase snapshot from development validation

Regenerate the published comparative benchmark artifacts with one command:

```bash
python scripts/regenerate_benchmark_artifacts.py
```

## Quick Start

Install the package with runtime dependencies:

```bash
pip install ".[runtime]"
```

Run the focused test suite:

```bash
python -m unittest tests.test_semantic_memory_federation
```

Run the server locally against the included demo store:

```bash
python scripts/semantic_memory_mcp_server.py \
  --federated-config sample_data/demo_local_federated_memory_config.json \
  --default-active-store repo-demo \
  --semantic-provider-warmup startup
```

The local config points at the included demo RWIF file and is intended for out-of-box validation. The Docker and Docker Desktop examples use [sample_data/demo_federated_memory_config.json](sample_data/demo_federated_memory_config.json), which is written for mounted container paths.

## Core MCP Tools

- `memory_list_stores`
- `memory_set_active_stores`
- `memory_status`
- `memory_get`
- `memory_query`
- `memory_answer`
- `memory_reason`

Write-governance tools are available as well:

- `memory_list_proposals`
- `memory_list_proposal_events`
- `memory_propose`
- `memory_review_proposal`
- `memory_commit_proposal`

## Storage Model

RWIF semantic memory stores are configured through JSON. Each store may be declared with:

- a root directory name in `store_roots`
- a `relative_path` under that root
- an `access_mode` of `read-only` or `read-write`
- optional removable-media metadata

Minimal pattern:

```json
{
  "store_roots": {
    "always-on": "/data/stores",
    "removable": "/data/removable"
  },
  "stores": [
    {
      "store_id": "repo-demo",
      "root": "always-on",
      "relative_path": "demo_readme_semantic_store.rwif",
      "access_mode": "read-write"
    },
    {
      "store_id": "personal-removable",
      "root": "removable",
      "relative_path": "personal_records_demo.rwif",
      "access_mode": "read-only"
    }
  ]
}
```

## Docker

Build the image:

```bash
docker build -t semantic-memory-mcp-server:latest .
```

Run with mounted config and store directories:

```bash
docker run --rm -i \
  -e SEMANTIC_MEMORY_CONFIG=/data/config/demo_federated_memory_config.json \
  -e DEFAULT_ACTIVE_STORES=repo-demo \
  -v "$PWD/sample_data:/data/config:ro" \
  -v "/ABSOLUTE/PATH/TO/stores:/data/stores:ro" \
  -v "/ABSOLUTE/PATH/TO/removable:/data/removable:ro" \
  semantic-memory-mcp-server:latest
```

The included [docker-compose.yml](docker-compose.yml) shows the same pattern.

## Docker Desktop Examples

Two example settings files are included:

- [docker-desktop-settings.example.json](docker-desktop-settings.example.json): config and stores mounted read-only
- [docker-desktop-settings.writable.example.json](docker-desktop-settings.writable.example.json): config mounted read-only, removable/private root mounted read-only, main store root mounted read-write

The writable example is intentionally narrow. Only stores marked `access_mode: read-write` can mutate, and only if the underlying mount is also writable.

## Security

Read [SECURITY.md](SECURITY.md) before exposing write tools to external clients. The short version is: write access is local capability, not a cosmetic flag.

## Included Sample Assets

- [sample_data/demo_local_federated_memory_config.json](sample_data/demo_local_federated_memory_config.json)
- [sample_data/demo_federated_memory_config.json](sample_data/demo_federated_memory_config.json)
- [sample_data/demo_readme_semantic_store.rwif](sample_data/demo_readme_semantic_store.rwif)
- [sample_data/benchmark_text_records_array.jsonl](sample_data/benchmark_text_records_array.jsonl)
- [sample_data/benchmark_text_queries_array.jsonl](sample_data/benchmark_text_queries_array.jsonl)
- [sample_data/benchmark_text_calibration_array.jsonl](sample_data/benchmark_text_calibration_array.jsonl)
- [sample_data/benchmark_text_large_records.jsonl](sample_data/benchmark_text_large_records.jsonl)
- [sample_data/benchmark_text_large_queries.jsonl](sample_data/benchmark_text_large_queries.jsonl)
- [sample_data/benchmark_text_large_calibration.jsonl](sample_data/benchmark_text_large_calibration.jsonl)
- [sample_data/research_notes_sample.jsonl](sample_data/research_notes_sample.jsonl)
- [sample_data/personal_records_sample.jsonl](sample_data/personal_records_sample.jsonl)

## Scope

This repository is intentionally narrow. It packages the MCP semantic-memory server and the RWIF runtime it depends on. It does not include the broader research and appliance scaffolding from the original monorepo.