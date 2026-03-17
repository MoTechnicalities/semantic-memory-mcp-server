from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rwif_memory_store import save_memory_store
from rwif_retriever import RwifRetriever
from rwif_retriever.io import build_query_cases, build_text_memory_seeds, load_jsonl


TOKEN_RE = re.compile(r"[a-z0-9]+")


class DeterministicHashedTextProvider:
    def __init__(self, *, vector_length: int = 384) -> None:
        self.vector_length = int(vector_length)
        if self.vector_length <= 0:
            raise ValueError("vector_length must be positive")

    def encode_text(self, text: str) -> np.ndarray:
        normalized = self._normalize(text)
        vector = np.zeros(self.vector_length, dtype=np.float64)

        tokens = TOKEN_RE.findall(normalized)
        bigrams = [f"{left}_{right}" for left, right in zip(tokens, tokens[1:], strict=False)]
        char_trigrams = [normalized[index : index + 3] for index in range(max(0, len(normalized) - 2))]

        for feature, weight in self._iter_features(tokens, bigrams, char_trigrams):
            index = self._hash_index(feature)
            sign = self._hash_sign(feature)
            vector[index] += sign * weight

        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("texts must not be empty")
        return np.stack([self.encode_text(text) for text in texts], axis=0)

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split())

    def _hash_index(self, feature: str) -> int:
        digest = hashlib.blake2b(f"idx:{feature}".encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) % self.vector_length

    @staticmethod
    def _hash_sign(feature: str) -> float:
        digest = hashlib.blake2b(f"sgn:{feature}".encode("utf-8"), digest_size=1).digest()
        return 1.0 if digest[0] % 2 == 0 else -1.0

    @staticmethod
    def _iter_features(tokens: list[str], bigrams: list[str], char_trigrams: list[str]):
        for token in tokens:
            yield f"tok:{token}", 1.0
        for bigram in bigrams:
            yield f"big:{bigram}", 0.85
        for trigram in char_trigrams:
            if " " in trigram:
                continue
            yield f"tri:{trigram}", 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RWIF wave retrieval against dense cosine on a larger shipped text corpus using deterministic hashed activations."
    )
    parser.add_argument(
        "--records",
        default=str(REPO_ROOT / "sample_data" / "benchmark_text_large_records.jsonl"),
        help="Path to the benchmark record JSONL.",
    )
    parser.add_argument(
        "--queries",
        default=str(REPO_ROOT / "sample_data" / "benchmark_text_large_queries.jsonl"),
        help="Path to the benchmark query JSONL.",
    )
    parser.add_argument(
        "--calibration",
        default=str(REPO_ROOT / "sample_data" / "benchmark_text_large_calibration.jsonl"),
        help="Path to the benchmark calibration JSONL.",
    )
    parser.add_argument("--vector-length", type=int, default=384)
    parser.add_argument("--top-k-waves", type=int, default=96)
    parser.add_argument("--rank-depth", type=int, default=12)
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
        "# RWIF vs Dense Cosine on a Larger Shipped Text Corpus",
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
        f"- Top-1 overlap rate: `{overlap_rates.get('1', 0.0) * 100:.1f}%`",
        f"- Top-3 overlap rate: `{overlap_rates.get('3', 0.0) * 100:.1f}%`",
        f"- Top-5 overlap rate: `{overlap_rates.get('5', 0.0) * 100:.1f}%`",
        f"- RWIF store artifact size: `{summary['rwif_file_bytes']:,}` bytes",
        f"- Dense float32 matrix estimate: `{summary['dense_float32_bytes']:,}` bytes",
        f"- Dense float64 matrix estimate: `{summary['dense_float64_bytes']:,}` bytes",
        f"- RWIF vs dense float32 size ratio: `{summary['rwif_vs_dense_float32_ratio']:.3f}`",
        f"- RWIF vs dense float64 size ratio: `{summary['rwif_vs_dense_float64_ratio']:.3f}`",
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
            "This benchmark scales the shipped workload comparison up to a larger text corpus and uses deterministic hashed-text activations so the full benchmark remains reproducible with the base runtime dependencies only.",
            "",
            "## Caveat",
            "",
            "The activations here are deterministic text hashes, not a learned embedding model. This benchmark is intended as a larger reproducible release check for RWIF-versus-dense behavior, not as a substitute for transformer-backed evaluation.",
        ]
    )
    return "\n".join(lines) + "\n"


def _safe_ranking_analysis(
    retriever: RwifRetriever,
    queries: list[Any],
    *,
    rank_depth: int,
    overlap_cutoffs: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, Any]:
    record_count = len(retriever.record_ids)
    depth = max(1, min(rank_depth, record_count))
    cutoffs = tuple(sorted({min(max(1, cutoff), depth) for cutoff in overlap_cutoffs if cutoff > 0}))

    full_agreements = 0
    spearman_total = 0.0
    absolute_shift_total = 0.0
    overlap_totals = {str(cutoff): 0.0 for cutoff in cutoffs}
    per_query: list[dict[str, Any]] = []

    for query in queries:
        wave_hits = retriever.query_text_wave(query.query_text, top_k=record_count)
        cosine_hits = retriever.query_text_cosine(query.query_text, top_k=record_count)

        wave_ids = [hit.record.record_id for hit in wave_hits]
        cosine_ids = [hit.record.record_id for hit in cosine_hits]
        wave_ranks = {record_id: index + 1 for index, record_id in enumerate(wave_ids)}
        cosine_ranks = {record_id: index + 1 for index, record_id in enumerate(cosine_ids)}

        truncated_wave_ids = wave_ids[:depth]
        truncated_cosine_ids = cosine_ids[:depth]
        if truncated_wave_ids == truncated_cosine_ids:
            full_agreements += 1

        compared_ids = retriever.record_ids
        rank_deltas = {record_id: wave_ranks[record_id] - cosine_ranks[record_id] for record_id in compared_ids}
        squared_distance = sum(delta * delta for delta in rank_deltas.values())
        if record_count == 1:
            spearman = 1.0
        else:
            spearman = 1.0 - ((6.0 * squared_distance) / (record_count * ((record_count * record_count) - 1)))
        mean_absolute_shift = float(np.mean([abs(delta) for delta in rank_deltas.values()])) if rank_deltas else 0.0
        spearman_total += spearman
        absolute_shift_total += mean_absolute_shift

        overlap_at_k: dict[str, dict[str, Any]] = {}
        for cutoff in cutoffs:
            wave_top_ids = truncated_wave_ids[:cutoff]
            cosine_top_ids = truncated_cosine_ids[:cutoff]
            cosine_top_set = set(cosine_top_ids)
            shared = [record_id for record_id in wave_top_ids if record_id in cosine_top_set]
            rate = len(shared) / cutoff
            overlap_totals[str(cutoff)] += rate
            overlap_at_k[str(cutoff)] = {
                "shared_count": len(shared),
                "shared_rate": rate,
                "shared_record_ids": shared,
            }

        expected_record_id = query.expected_record_id
        per_query.append(
            {
                "query_text": query.query_text,
                "expected_record_id": expected_record_id,
                "expected_wave_rank": None if expected_record_id is None else wave_ranks.get(expected_record_id),
                "expected_cosine_rank": None if expected_record_id is None else cosine_ranks.get(expected_record_id),
                "spearman_rank_correlation": spearman,
                "mean_absolute_rank_shift": mean_absolute_shift,
                "overlap_at_k": overlap_at_k,
                "largest_rank_shifts": [
                    {
                        "record_id": record_id,
                        "wave_rank": wave_ranks[record_id],
                        "cosine_rank": cosine_ranks[record_id],
                        "rank_delta": rank_deltas[record_id],
                    }
                    for record_id in sorted(rank_deltas, key=lambda item: (abs(rank_deltas[item]), item), reverse=True)[:5]
                ],
                "wave_ranking": [
                    {
                        "record_id": hit.record.record_id,
                        "score": hit.score,
                    }
                    for hit in wave_hits[:depth]
                ],
                "cosine_ranking": [
                    {
                        "record_id": hit.record.record_id,
                        "score": hit.score,
                    }
                    for hit in cosine_hits[:depth]
                ],
            }
        )

    query_count = len(queries)
    return {
        "record_count": record_count,
        "rank_depth": depth,
        "full_ranking_agreement_rate": full_agreements / query_count,
        "mean_spearman_rank_correlation": spearman_total / query_count,
        "mean_absolute_rank_shift": absolute_shift_total / query_count,
        "overlap_rates_by_k": {key: value / query_count for key, value in overlap_totals.items()},
        "per_query": per_query,
    }


def main() -> None:
    args = parse_args()
    record_rows = load_jsonl(args.records)
    query_rows = load_jsonl(args.queries)
    calibration_rows = load_jsonl(args.calibration)

    provider = DeterministicHashedTextProvider(vector_length=args.vector_length)
    records = build_text_memory_seeds(record_rows)
    queries = build_query_cases(query_rows)
    calibration_texts = [str(row["text"]).strip() for row in calibration_rows if str(row.get("text", "")).strip()]

    retriever = RwifRetriever.from_texts(
        provider=provider,
        records=records,
        calibration_texts=calibration_texts,
        top_k_waves=args.top_k_waves,
        metadata={
            "benchmark": "rwif-vs-dense-text-corpus",
            "dataset_kind": "real-text-hashed-corpus",
        },
    )

    benchmark = retriever.benchmark(queries, top_k=1)
    ranking = _safe_ranking_analysis(retriever, queries, rank_depth=args.rank_depth, overlap_cutoffs=(1, 3, 5, 10))

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "text_corpus_benchmark_store.rwif"
        save_memory_store(temp_path, retriever.memory_store)
        rwif_file_bytes = temp_path.stat().st_size

    record_count = len(records)
    dense_float32_bytes = record_count * args.vector_length * 4
    dense_float64_bytes = record_count * args.vector_length * 8

    payload = {
        "dataset": {
            "dataset_kind": "real-text-hashed-corpus",
            "record_count": record_count,
            "query_count": len(queries),
            "calibration_count": len(calibration_texts),
            "vector_length": args.vector_length,
            "top_k_waves": args.top_k_waves,
            "rank_depth": ranking["rank_depth"],
            "records_path": _repo_relative(args.records),
            "queries_path": _repo_relative(args.queries),
            "calibration_path": _repo_relative(args.calibration),
            "provider_kind": "deterministic-hashed-text",
        },
        "summary": {
            "wave_top1_accuracy": benchmark.wave_top1_accuracy,
            "cosine_top1_accuracy": benchmark.cosine_top1_accuracy,
            "agreement_rate": benchmark.agreement_rate,
            "full_ranking_agreement_rate": ranking["full_ranking_agreement_rate"],
            "mean_spearman_rank_correlation": ranking["mean_spearman_rank_correlation"],
            "mean_absolute_rank_shift": ranking["mean_absolute_rank_shift"],
            "overlap_rates_by_k": ranking["overlap_rates_by_k"],
            "rwif_file_bytes": rwif_file_bytes,
            "dense_float32_bytes": dense_float32_bytes,
            "dense_float64_bytes": dense_float64_bytes,
            "rwif_vs_dense_float32_ratio": rwif_file_bytes / dense_float32_bytes,
            "rwif_vs_dense_float64_ratio": rwif_file_bytes / dense_float64_bytes,
            "dense_float32_vs_rwif_ratio": dense_float32_bytes / rwif_file_bytes,
            "dense_float64_vs_rwif_ratio": dense_float64_bytes / rwif_file_bytes,
        },
        "per_query": list(benchmark.per_query),
        "ranking_analysis": list(ranking["per_query"]),
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