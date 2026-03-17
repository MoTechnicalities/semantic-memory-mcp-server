# Repository Layout

```text
semantic-memory-mcp-server/
├── .github/
│   └── workflows/
│       └── ci.yml
├── benchmark_results/
│   ├── rwif_vs_dense_cosine_baseline.json
│   ├── rwif_vs_dense_cosine_baseline.md
│   ├── rwif_vs_dense_text_corpus_baseline.json
│   ├── rwif_vs_dense_text_corpus_baseline.md
│   ├── rwif_vs_dense_text_array_baseline.json
│   └── rwif_vs_dense_text_array_baseline.md
├── sample_data/
│   ├── benchmark_text_large_calibration.jsonl
│   ├── benchmark_text_large_queries.jsonl
│   ├── benchmark_text_large_records.jsonl
│   ├── benchmark_text_calibration_array.jsonl
│   ├── benchmark_text_queries_array.jsonl
│   ├── benchmark_text_records_array.jsonl
│   ├── demo_federated_memory_config.json
│   ├── demo_local_federated_memory_config.json
│   ├── demo_readme_semantic_store.rwif
│   ├── personal_records_sample.jsonl
│   ├── research_notes_sample.jsonl
│   └── README.md
├── scripts/
│   ├── benchmark_rwif_vs_dense_baseline.py
│   ├── benchmark_rwif_vs_dense_text_corpus.py
│   ├── benchmark_rwif_vs_dense_text_array.py
│   ├── regenerate_benchmark_artifacts.py
│   └── semantic_memory_mcp_server.py
├── src/
│   ├── big_ai_brain/
│   │   ├── __init__.py
│   │   ├── federated_memory.py
│   │   ├── semantic_memory_mcp.py
│   │   └── semantic_memory_service.py
│   ├── rwif_activation_core/
│   ├── rwif_memory_store/
│   ├── rwif_retriever/
│   └── rwif_semantic_memory/
├── tests/
│   └── test_semantic_memory_federation.py
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── docker-desktop-settings.example.json
├── docker-desktop-settings.writable.example.json
├── Dockerfile
├── entrypoint.sh
├── LICENSE
├── pyproject.toml
├── README.md
├── RELEASE_CHECKLIST.md
├── SECURITY.md
└── TAG_PREPARATION.md
```

This export set intentionally excludes unrelated research, runtime, and model-appliance code from the original workspace.