from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rwif_activation_core import AtomicWaveUnit, WaveState, decode_wave_state, encode_activation
from rwif_memory_store import save_memory_store
from rwif_retriever import ArrayActivationProvider, QueryCase, RwifRetriever, TextMemorySeed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RWIF retrieval against a dense cosine baseline on a reproducible sparse-spectrum benchmark."
    )
    parser.add_argument("--record-count", type=int, default=64)
    parser.add_argument("--query-count", type=int, default=16)
    parser.add_argument("--vector-length", type=int, default=1024)
    parser.add_argument("--top-k-waves", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def _random_sparse_activation(rng: np.random.Generator, *, vector_length: int, active_units: int) -> np.ndarray:
    indices = rng.choice(vector_length, size=active_units, replace=False)
    amplitudes = rng.normal(loc=0.0, scale=1.0, size=active_units)
    units = tuple(
        AtomicWaveUnit(frequency_index=int(index), amplitude=float(amplitude))
        for index, amplitude in zip(indices.tolist(), amplitudes.tolist(), strict=True)
    )
    return decode_wave_state(WaveState(vector_length=vector_length, units=units, label="synthetic"))


def _build_dataset(
    *,
    record_count: int,
    query_count: int,
    vector_length: int,
    top_k_waves: int,
    seed: int,
) -> tuple[list[TextMemorySeed], list[QueryCase], dict[str, np.ndarray], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    activations: dict[str, np.ndarray] = {}
    records: list[TextMemorySeed] = []
    queries: list[QueryCase] = []

    for record_index in range(record_count):
        text = f"Synthetic memory record {record_index}"
        record_id = f"record-{record_index:04d}"
        activation = _random_sparse_activation(
            rng,
            vector_length=vector_length,
            active_units=top_k_waves,
        )
        activations[text] = activation
        records.append(
            TextMemorySeed(
                record_id=record_id,
                text=text,
                metadata={"kind": "synthetic-benchmark", "index": record_index},
            )
        )

    query_indices = rng.choice(record_count, size=min(query_count, record_count), replace=False)
    for query_number, record_index in enumerate(query_indices.tolist()):
        record = records[record_index]
        query_text = f"Synthetic query {query_number} for {record.record_id}"
        base_activation = activations[record.text]
        noise = rng.normal(loc=0.0, scale=0.0025, size=vector_length)
        query_activation = np.asarray(base_activation + noise, dtype=np.float64)
        activations[query_text] = query_activation
        queries.append(QueryCase(query_text=query_text, expected_record_id=record.record_id))

    metadata = {
        "dataset_kind": "synthetic-sparse-spectrum",
        "record_count": record_count,
        "query_count": len(queries),
        "vector_length": vector_length,
        "top_k_waves": top_k_waves,
        "noise_std": 0.0025,
        "seed": seed,
    }
    return records, queries, activations, metadata


def _render_markdown(payload: dict[str, Any]) -> str:
    comparison = payload["comparison"]
    dataset = payload["dataset"]
    return "\n".join(
        [
            "# RWIF vs Dense Cosine Baseline",
            "",
            "## Summary",
            "",
            f"- Dataset kind: `{dataset['dataset_kind']}`",
            f"- Records: `{dataset['record_count']}`",
            f"- Queries: `{dataset['query_count']}`",
            f"- Vector length: `{dataset['vector_length']}`",
            f"- Top-k wave units per record: `{dataset['top_k_waves']}`",
            f"- RWIF compressed top-1 accuracy: `{comparison['rwif_compressed_top1_accuracy'] * 100:.1f}%`",
            f"- Dense cosine top-1 accuracy: `{comparison['cosine_top1_accuracy'] * 100:.1f}%`",
            f"- Agreement rate: `{comparison['agreement_rate'] * 100:.1f}%`",
            f"- RWIF file size: `{comparison['rwif_file_bytes']:,}` bytes",
            f"- Dense float32 matrix estimate: `{comparison['dense_float32_bytes']:,}` bytes",
            f"- Dense float64 matrix estimate: `{comparison['dense_float64_bytes']:,}` bytes",
            f"- RWIF vs dense float32 size ratio: `{comparison['rwif_vs_dense_float32_ratio']:.3f}`",
            f"- RWIF vs dense float64 size ratio: `{comparison['rwif_vs_dense_float64_ratio']:.3f}`",
            "",
            "## Interpretation",
            "",
            "This benchmark compares one concrete vector baseline against the RWIF storage method: brute-force dense cosine retrieval on original activations versus cosine retrieval on RWIF-compressed reconstructions of those same activations.",
            "The storage comparison only models the embedding matrix itself. It does not include external database services, index structures, metadata services, or orchestration overhead.",
            "",
            "## Caveat",
            "",
            "This is a synthetic sparse-spectrum benchmark designed to isolate the file-storage implications of the RWIF encoding. It is not a claim about all embedding distributions or all vector database systems.",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    records, queries, activations, dataset_metadata = _build_dataset(
        record_count=args.record_count,
        query_count=args.query_count,
        vector_length=args.vector_length,
        top_k_waves=args.top_k_waves,
        seed=args.seed,
    )
    provider = ArrayActivationProvider(activations=activations)
    retriever = RwifRetriever.from_texts(
        provider=provider,
        records=records,
        top_k_waves=args.top_k_waves,
        metadata={"benchmark": "rwif-vs-dense-cosine", **dataset_metadata},
    )

    background = np.asarray(retriever.memory_store.background, dtype=np.float64)
    record_texts = [record.text for record in records]
    record_ids = [record.record_id for record in records]
    dense_record_matrix = np.stack([np.asarray(activations[text], dtype=np.float64) - background for text in record_texts], axis=0)
    rwif_record_matrix = np.stack([record.state.reconstruct() for record in retriever.memory_store.records], axis=0)

    dense_record_norms = np.linalg.norm(dense_record_matrix, axis=1)
    rwif_record_norms = np.linalg.norm(rwif_record_matrix, axis=1)

    rwif_correct = 0
    cosine_correct = 0
    agreements = 0
    per_query: list[dict[str, Any]] = []

    for query in queries:
        dense_query = np.asarray(activations[query.query_text], dtype=np.float64) - background
        rwif_query = decode_wave_state(
            encode_activation(
                np.asarray(activations[query.query_text], dtype=np.float64),
                background=background,
                top_k=args.top_k_waves,
                label=query.query_text,
            )
        )

        dense_query_norm = float(np.linalg.norm(dense_query))
        rwif_query_norm = float(np.linalg.norm(rwif_query))

        dense_scores = (dense_record_matrix @ dense_query) / np.clip(dense_record_norms * dense_query_norm, 1e-12, None)
        rwif_scores = (rwif_record_matrix @ rwif_query) / np.clip(rwif_record_norms * rwif_query_norm, 1e-12, None)

        dense_top_index = int(np.argmax(dense_scores))
        rwif_top_index = int(np.argmax(rwif_scores))
        dense_top = record_ids[dense_top_index]
        rwif_top = record_ids[rwif_top_index]

        if dense_top == query.expected_record_id:
            cosine_correct += 1
        if rwif_top == query.expected_record_id:
            rwif_correct += 1
        if dense_top == rwif_top:
            agreements += 1

        per_query.append(
            {
                "query_text": query.query_text,
                "expected_record_id": query.expected_record_id,
                "rwif_compressed_top": rwif_top,
                "cosine_top": dense_top,
                "rwif_compressed_score": float(rwif_scores[rwif_top_index]),
                "cosine_score": float(dense_scores[dense_top_index]),
            }
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "synthetic_benchmark_store.rwif"
        save_memory_store(temp_path, retriever.memory_store)
        rwif_file_bytes = temp_path.stat().st_size

    dense_float32_bytes = args.record_count * args.vector_length * 4
    dense_float64_bytes = args.record_count * args.vector_length * 8

    payload = {
        "dataset": dataset_metadata,
        "comparison": {
            "rwif_compressed_top1_accuracy": rwif_correct / len(queries),
            "cosine_top1_accuracy": cosine_correct / len(queries),
            "agreement_rate": agreements / len(queries),
            "rwif_file_bytes": rwif_file_bytes,
            "dense_float32_bytes": dense_float32_bytes,
            "dense_float64_bytes": dense_float64_bytes,
            "rwif_vs_dense_float32_ratio": rwif_file_bytes / dense_float32_bytes,
            "rwif_vs_dense_float64_ratio": rwif_file_bytes / dense_float64_bytes,
            "dense_float32_vs_rwif_ratio": dense_float32_bytes / rwif_file_bytes,
            "dense_float64_vs_rwif_ratio": dense_float64_bytes / rwif_file_bytes,
        },
        "per_query": per_query,
    }

    output_json_path = Path(args.output_json)
    output_md_path = Path(args.output_md)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output_md_path.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()