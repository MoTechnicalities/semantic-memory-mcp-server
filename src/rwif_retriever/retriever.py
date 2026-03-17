from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rwif_activation_core import WaveState
from rwif_memory_store import MemoryQueryResult, MemoryRecord, MemoryStore, estimate_background

from .providers import ActivationProvider


@dataclass(frozen=True)
class TextMemorySeed:
    record_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str | None = None


@dataclass(frozen=True)
class RetrievalHit:
    record: MemoryRecord
    score: float
    method: str


@dataclass(frozen=True)
class QueryCase:
    query_text: str
    expected_record_id: str | None = None


@dataclass(frozen=True)
class RetrievalBenchmarkResult:
    query_count: int
    wave_top1_accuracy: float | None
    cosine_top1_accuracy: float | None
    agreement_rate: float
    per_query: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RetrievalRankingAnalysisResult:
    query_count: int
    record_count: int
    rank_depth: int
    full_ranking_agreement_rate: float
    mean_spearman_rank_correlation: float
    mean_absolute_rank_shift: float
    overlap_rates_by_k: dict[str, float]
    per_query: tuple[dict[str, Any], ...]


class RwifRetriever:
    def __init__(
        self,
        *,
        provider: ActivationProvider,
        memory_store: MemoryStore,
        indexed_centered_activations: np.ndarray,
        record_ids: list[str],
    ) -> None:
        self.provider = provider
        self.memory_store = memory_store
        self.indexed_centered_activations = np.asarray(indexed_centered_activations, dtype=np.float64)
        self.record_ids = list(record_ids)
        if self.indexed_centered_activations.ndim != 2:
            raise ValueError("indexed_centered_activations must be a 2D matrix")
        if self.indexed_centered_activations.shape[0] != len(self.record_ids):
            raise ValueError("record_ids must align with indexed_centered_activations")

    @classmethod
    def from_texts(
        cls,
        *,
        provider: ActivationProvider,
        records: list[TextMemorySeed],
        calibration_texts: list[str] | None = None,
        top_k_waves: int = 128,
        metadata: dict[str, Any] | None = None,
    ) -> RwifRetriever:
        if not records:
            raise ValueError("records must not be empty")
        if calibration_texts:
            background = estimate_background(provider.encode_texts(calibration_texts))
        else:
            background = estimate_background(provider.encode_texts([record.text for record in records]))

        memory_store = MemoryStore(background=background, metadata=dict(metadata or {}))
        centered_vectors: list[np.ndarray] = []
        record_ids: list[str] = []
        for record in records:
            activation = np.asarray(provider.encode_text(record.text), dtype=np.float64)
            memory_store.add_activation(
                activation=activation,
                record_id=record.record_id,
                text=record.text,
                top_k=top_k_waves,
                metadata=record.metadata,
                source=record.source,
            )
            centered_vectors.append(activation - background)
            record_ids.append(record.record_id)
        return cls(
            provider=provider,
            memory_store=memory_store,
            indexed_centered_activations=np.stack(centered_vectors, axis=0),
            record_ids=record_ids,
        )

    def query_text_wave(self, text: str, *, top_k: int = 5) -> list[RetrievalHit]:
        results = self.memory_store.query_activation(np.asarray(self.provider.encode_text(text), dtype=np.float64), top_k=top_k)
        return [RetrievalHit(record=result.record, score=result.score, method="wave") for result in results]

    def query_text_cosine(self, text: str, *, top_k: int = 5) -> list[RetrievalHit]:
        query_activation = np.asarray(self.provider.encode_text(text), dtype=np.float64)
        centered_query = self._center_activation(query_activation)
        scores = self._cosine_scores(centered_query)
        indices = np.argsort(scores)[::-1][: max(1, top_k)]
        record_map = {record.record_id: record for record in self.memory_store.records}
        return [
            RetrievalHit(record=record_map[self.record_ids[index]], score=float(scores[index]), method="cosine")
            for index in indices
        ]

    def benchmark(self, queries: list[QueryCase], *, top_k: int = 1) -> RetrievalBenchmarkResult:
        if not queries:
            raise ValueError("queries must not be empty")

        wave_correct = 0
        cosine_correct = 0
        agreements = 0
        per_query: list[dict[str, Any]] = []
        has_expected = any(query.expected_record_id is not None for query in queries)

        for query in queries:
            wave_hits = self.query_text_wave(query.query_text, top_k=top_k)
            cosine_hits = self.query_text_cosine(query.query_text, top_k=top_k)
            wave_top = wave_hits[0].record.record_id
            cosine_top = cosine_hits[0].record.record_id
            if wave_top == cosine_top:
                agreements += 1

            if query.expected_record_id is not None:
                if wave_top == query.expected_record_id:
                    wave_correct += 1
                if cosine_top == query.expected_record_id:
                    cosine_correct += 1

            per_query.append(
                {
                    "query_text": query.query_text,
                    "expected_record_id": query.expected_record_id,
                    "wave_top": wave_top,
                    "cosine_top": cosine_top,
                    "wave_score": wave_hits[0].score,
                    "cosine_score": cosine_hits[0].score,
                }
            )

        query_count = len(queries)
        return RetrievalBenchmarkResult(
            query_count=query_count,
            wave_top1_accuracy=(wave_correct / query_count) if has_expected else None,
            cosine_top1_accuracy=(cosine_correct / query_count) if has_expected else None,
            agreement_rate=agreements / query_count,
            per_query=tuple(per_query),
        )

    def analyze_rankings(
        self,
        queries: list[QueryCase],
        *,
        rank_depth: int | None = None,
        overlap_cutoffs: tuple[int, ...] = (1, 3, 5, 10),
    ) -> RetrievalRankingAnalysisResult:
        if not queries:
            raise ValueError("queries must not be empty")

        max_depth = len(self.record_ids)
        depth = max_depth if rank_depth is None else max(1, min(rank_depth, max_depth))
        cutoffs = tuple(sorted({min(max(1, cutoff), depth) for cutoff in overlap_cutoffs if cutoff > 0}))
        if not cutoffs:
            cutoffs = (min(1, depth),) if depth > 0 else (1,)

        full_agreements = 0
        spearman_total = 0.0
        absolute_shift_total = 0.0
        overlap_totals = {str(cutoff): 0.0 for cutoff in cutoffs}
        per_query: list[dict[str, Any]] = []

        for query in queries:
            wave_hits = self.query_text_wave(query.query_text, top_k=depth)
            cosine_hits = self.query_text_cosine(query.query_text, top_k=depth)

            wave_ids = [hit.record.record_id for hit in wave_hits]
            cosine_ids = [hit.record.record_id for hit in cosine_hits]
            wave_ranks = {record_id: index + 1 for index, record_id in enumerate(wave_ids)}
            cosine_ranks = {record_id: index + 1 for index, record_id in enumerate(cosine_ids)}

            if wave_ids == cosine_ids:
                full_agreements += 1

            rank_deltas = {record_id: wave_ranks[record_id] - cosine_ranks[record_id] for record_id in wave_ids}
            squared_distance = sum(delta * delta for delta in rank_deltas.values())
            if depth == 1:
                spearman = 1.0
            else:
                spearman = 1.0 - ((6.0 * squared_distance) / (depth * ((depth * depth) - 1)))
            mean_absolute_shift = float(np.mean([abs(delta) for delta in rank_deltas.values()])) if rank_deltas else 0.0
            spearman_total += spearman
            absolute_shift_total += mean_absolute_shift

            overlap_at_k: dict[str, dict[str, Any]] = {}
            for cutoff in cutoffs:
                wave_top_ids = wave_ids[:cutoff]
                cosine_top_ids = cosine_ids[:cutoff]
                shared = [record_id for record_id in wave_top_ids if record_id in set(cosine_top_ids)]
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
                        for hit in wave_hits
                    ],
                    "cosine_ranking": [
                        {
                            "record_id": hit.record.record_id,
                            "score": hit.score,
                        }
                        for hit in cosine_hits
                    ],
                }
            )

        query_count = len(queries)
        return RetrievalRankingAnalysisResult(
            query_count=query_count,
            record_count=max_depth,
            rank_depth=depth,
            full_ranking_agreement_rate=full_agreements / query_count,
            mean_spearman_rank_correlation=spearman_total / query_count,
            mean_absolute_rank_shift=absolute_shift_total / query_count,
            overlap_rates_by_k={key: value / query_count for key, value in overlap_totals.items()},
            per_query=tuple(per_query),
        )

    def _center_activation(self, activation: np.ndarray) -> np.ndarray:
        background = self.memory_store.background
        if background is None:
            return activation
        return activation - np.asarray(background, dtype=np.float64)

    def _cosine_scores(self, centered_query: np.ndarray) -> np.ndarray:
        matrix = self.indexed_centered_activations
        query_norm = float(np.linalg.norm(centered_query))
        matrix_norms = np.linalg.norm(matrix, axis=1)
        denom = matrix_norms * query_norm
        scores = np.zeros(matrix.shape[0], dtype=np.float64)
        valid = denom > 0.0
        if np.any(valid):
            scores[valid] = (matrix[valid] @ centered_query) / denom[valid]
        return scores