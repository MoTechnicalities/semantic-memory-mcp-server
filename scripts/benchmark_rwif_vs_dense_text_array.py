from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rwif_memory_store import save_memory_store
from rwif_retriever import RwifRetriever
from rwif_retriever.io import build_provider, build_query_cases, build_text_memory_seeds, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RWIF wave retrieval against dense cosine on a shipped real text-derived array dataset."
    )
    parser.add_argument(
        "--records",
        default=str(REPO_ROOT / "sample_data" / "benchmark_text_records_array.jsonl"),
        help="Path to the benchmark record JSONL.",
    )
    parser.add_argument(
        "--queries",
        default=str(REPO_ROOT / "sample_data" / "benchmark_text_queries_array.jsonl"),
        help="Path to the benchmark query JSONL.",
    )
    parser.add_argument(
        "--calibration",
        default=str(REPO_ROOT / "sample_data" / "benchmark_text_calibration_array.jsonl"),
        help="Path to the benchmark calibration JSONL.",
    )
    parser.add_argument("--top-k-waves", type=int, default=4)
    parser.add_argument("--rank-depth", type=int, default=6)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def _repo_relative(path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(REPO_ROOT))


def _render_markdown(payload: dict[str, Any]) -> str:
    dataset = payload["dataset"]
    summary = payload["summary"]
    overlap_rates = summary["overlap_rates_by_k"]
    lines = [
        "# RWIF vs Dense Cosine on a Real Text Array Dataset",
        "",
        "## Summary",
        "",
        f"- Dataset kind: `{dataset['dataset_kind']}`",
        f"- Records: `{dataset['record_count']}`",
        f"- Queries: `{dataset['query_count']}`",
        f"- Calibration texts: `{dataset['calibration_count']}`",
        f"- Vector length: `{dataset['vector_length']}`",
        f"- Top-k wave units per record: `{dataset['top_k_waves']}`",
        f"- RWIF wave top-1 accuracy: `{summary['wave_top1_accuracy'] * 100:.1f}%`",
        f"- Dense cosine top-1 accuracy: `{summary['cosine_top1_accuracy'] * 100:.1f}%`",
        f"- Top-1 agreement rate: `{summary['agreement_rate'] * 100:.1f}%`",
        f"- Full-ranking agreement rate: `{summary['full_ranking_agreement_rate'] * 100:.1f}%`",
        f"- Mean Spearman rank correlation: `{summary['mean_spearman_rank_correlation']:.3f}`",
        f"- Mean absolute rank shift: `{summary['mean_absolute_rank_shift']:.3f}`",
        f"- Top-1 overlap rate: `{overlap_rates['1'] * 100:.1f}%`",
        f"- Top-3 overlap rate: `{overlap_rates.get('3', 0.0) * 100:.1f}%`",
        f"- RWIF store artifact size: `{summary['rwif_file_bytes']:,}` bytes",
        "",
        "## Per-Query Snapshot",
        "",
        "| Query | Expected | RWIF top | Cosine top |",
        "|---|---|---|---|",
    ]
    for query in payload["per_query"]:
        lines.append(
            f"| {query['query_text']} | {query['expected_record_id']} | {query['wave_top']} | {query['cosine_top']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This benchmark is workload-shaped rather than storage-shaped. It uses real text records, real text queries, a shipped calibration set, the RWIF wave retrieval path, and the dense cosine baseline on the same text-derived activations.",
            "",
            "## Caveat",
            "",
            "The dataset is intentionally tiny and array-backed so it can ship in the public repository as a deterministic release check. It should be read as a reproducible parity benchmark, not as a large-corpus performance claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    record_rows = load_jsonl(args.records)
    query_rows = load_jsonl(args.queries)
    calibration_rows = load_jsonl(args.calibration)

    provider = build_provider(provider_kind="array", rows=record_rows, extra_rows=[*query_rows, *calibration_rows])
    records = build_text_memory_seeds(record_rows)
    queries = build_query_cases(query_rows)
    calibration_texts = [str(row["text"]).strip() for row in calibration_rows if str(row.get("text", "")).strip()]

    retriever = RwifRetriever.from_texts(
        provider=provider,
        records=records,
        calibration_texts=calibration_texts,
        top_k_waves=args.top_k_waves,
        metadata={
            "benchmark": "rwif-vs-dense-text-array",
            "dataset_kind": "real-text-array",
        },
    )

    benchmark = retriever.benchmark(queries, top_k=1)
    ranking = retriever.analyze_rankings(queries, rank_depth=args.rank_depth, overlap_cutoffs=(1, 3, 5))

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "text_array_benchmark_store.rwif"
        save_memory_store(temp_path, retriever.memory_store)
        rwif_file_bytes = temp_path.stat().st_size

    payload = {
        "dataset": {
            "dataset_kind": "real-text-array",
            "record_count": len(records),
            "query_count": len(queries),
            "calibration_count": len(calibration_texts),
            "vector_length": int(retriever.memory_store.vector_length or 0),
            "top_k_waves": args.top_k_waves,
            "rank_depth": ranking.rank_depth,
            "records_path": _repo_relative(args.records),
            "queries_path": _repo_relative(args.queries),
            "calibration_path": _repo_relative(args.calibration),
        },
        "summary": {
            "wave_top1_accuracy": benchmark.wave_top1_accuracy,
            "cosine_top1_accuracy": benchmark.cosine_top1_accuracy,
            "agreement_rate": benchmark.agreement_rate,
            "full_ranking_agreement_rate": ranking.full_ranking_agreement_rate,
            "mean_spearman_rank_correlation": ranking.mean_spearman_rank_correlation,
            "mean_absolute_rank_shift": ranking.mean_absolute_rank_shift,
            "overlap_rates_by_k": ranking.overlap_rates_by_k,
            "rwif_file_bytes": rwif_file_bytes,
        },
        "per_query": list(benchmark.per_query),
        "ranking_analysis": list(ranking.per_query),
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