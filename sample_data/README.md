# Sample Data

This folder contains both demo inputs and a small shipped RWIF artifact.

Included files:

- `demo_readme_semantic_store.rwif`: small validated semantic-memory store for immediate local or container demos
- `demo_local_federated_memory_config.json`: local config that points at the included RWIF file by relative path
- `demo_federated_memory_config.json`: container-oriented config that expects mounted `/data/config`, `/data/stores`, and `/data/removable` paths
- `benchmark_text_records_array.jsonl`: real text records with deterministic array activations for the shipped workload-shaped benchmark
- `benchmark_text_queries_array.jsonl`: benchmark queries with expected matches for the shipped workload-shaped benchmark
- `benchmark_text_calibration_array.jsonl`: calibration texts used to estimate the benchmark background vector
- `benchmark_text_large_records.jsonl`: larger shipped text corpus for the deterministic hashed-text benchmark
- `benchmark_text_large_queries.jsonl`: queries and expected matches for the larger shipped text corpus benchmark
- `benchmark_text_large_calibration.jsonl`: calibration texts for the larger shipped text corpus benchmark
- `research_notes_sample.jsonl`: sample ingestion input for a research-oriented store
- `personal_records_sample.jsonl`: sample ingestion input for a removable or private store

Suggested flow:

1. run the server locally with `demo_local_federated_memory_config.json` for an out-of-box smoke test
2. use `demo_federated_memory_config.json` for Docker or Docker Desktop examples
3. regenerate the shipped benchmark artifacts with `python scripts/regenerate_benchmark_artifacts.py`
4. replace the sample RWIF file or build your own stores from the shipped JSONL inputs