from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from rwif_activation_core import WaveLibrary, WaveState, encode_activation, interference_score, load_wave_library, save_wave_library


_MEMORY_METADATA_KEY = "rwif_memory_store"


def estimate_background(activations: np.ndarray) -> np.ndarray:
    matrix = np.asarray(activations, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("activations must be a 2D matrix of shape (samples, features)")
    if matrix.shape[0] == 0:
        raise ValueError("activations must contain at least one sample")
    return np.mean(matrix, axis=0)


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    text: str
    state: WaveState
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str | None = None


@dataclass(frozen=True)
class MemoryQueryResult:
    record: MemoryRecord
    score: float


@dataclass
class MemoryStore:
    background: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _records: list[MemoryRecord] = field(default_factory=list)

    @property
    def records(self) -> tuple[MemoryRecord, ...]:
        return tuple(self._records)

    @property
    def vector_length(self) -> int | None:
        if self.background is not None:
            return int(np.asarray(self.background).shape[0])
        if self._records:
            return self._records[0].state.vector_length
        return None

    def add_activation(
        self,
        *,
        activation: np.ndarray,
        record_id: str,
        text: str,
        top_k: int = 128,
        metadata: dict[str, Any] | None = None,
        source: str | None = None,
        label: str | None = None,
    ) -> MemoryRecord:
        vector = np.asarray(activation, dtype=np.float64)
        self._validate_vector_length(vector)
        state = encode_activation(
            vector,
            background=self.background,
            top_k=top_k,
            label=label or record_id,
            metadata=dict(metadata or {}),
        )
        record = MemoryRecord(
            record_id=record_id,
            text=text,
            state=state,
            metadata=dict(metadata or {}),
            source=source,
        )
        self._records.append(record)
        return record

    def add_state(
        self,
        *,
        state: WaveState,
        record_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> MemoryRecord:
        self._validate_state(state)
        record = MemoryRecord(
            record_id=record_id,
            text=text,
            state=state,
            metadata=dict(metadata or {}),
            source=source,
        )
        self._records.append(record)
        return record

    def query_activation(self, activation: np.ndarray, *, top_k: int = 5) -> list[MemoryQueryResult]:
        query_state = encode_activation(
            np.asarray(activation, dtype=np.float64),
            background=self.background,
            top_k=self._default_query_top_k(),
            label="query",
        )
        return self.query_state(query_state, top_k=top_k)

    def query_state(self, query_state: WaveState, *, top_k: int = 5) -> list[MemoryQueryResult]:
        self._validate_state(query_state)
        results = [
            MemoryQueryResult(record=record, score=interference_score(query_state, record.state))
            for record in self._records
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[: max(1, top_k)]

    def to_wave_library(self) -> WaveLibrary:
        states: list[WaveState] = []
        for record in self._records:
            state_metadata = dict(record.state.metadata)
            state_metadata[_MEMORY_METADATA_KEY] = {
                "record_id": record.record_id,
                "text": record.text,
                "metadata": record.metadata,
                "source": record.source,
            }
            states.append(replace(record.state, metadata=state_metadata))

        library_metadata = dict(self.metadata)
        if self.background is not None:
            library_metadata["background"] = np.asarray(self.background, dtype=np.float64).tolist()
        library_metadata["format"] = "rwif_memory_store"
        return WaveLibrary(states=tuple(states), metadata=library_metadata)

    @classmethod
    def from_wave_library(cls, library: WaveLibrary) -> MemoryStore:
        background_data = library.metadata.get("background")
        background = None if background_data is None else np.asarray(background_data, dtype=np.float64)
        store_metadata = {key: value for key, value in library.metadata.items() if key != "background"}
        store = cls(background=background, metadata=store_metadata)
        for state in library.states:
            memory_payload = dict(state.metadata.get(_MEMORY_METADATA_KEY, {}))
            record_metadata = dict(memory_payload.get("metadata", {}))
            record_id = str(memory_payload.get("record_id", state.label or f"record-{len(store._records)}"))
            text = str(memory_payload.get("text", state.label or record_id))
            source = memory_payload.get("source")
            stripped_state_metadata = {key: value for key, value in state.metadata.items() if key != _MEMORY_METADATA_KEY}
            store.add_state(
                state=replace(state, metadata=stripped_state_metadata),
                record_id=record_id,
                text=text,
                metadata=record_metadata,
                source=source,
            )
        return store

    def _default_query_top_k(self) -> int:
        if not self._records:
            return 1
        return max(record.state.top_k for record in self._records)

    def _validate_vector_length(self, vector: np.ndarray) -> None:
        if vector.ndim != 1:
            raise ValueError("activation must be a 1D vector")
        expected = self.vector_length
        if expected is not None and vector.shape[0] != expected:
            raise ValueError(f"vector length mismatch: expected {expected}, got {vector.shape[0]}")

    def _validate_state(self, state: WaveState) -> None:
        expected = self.vector_length
        if expected is not None and state.vector_length != expected:
            raise ValueError(f"state vector length mismatch: expected {expected}, got {state.vector_length}")


def save_memory_store(path: str | Path, store: MemoryStore) -> None:
    save_wave_library(path, store.to_wave_library())


def load_memory_store(path: str | Path) -> MemoryStore:
    library = load_wave_library(path)
    return MemoryStore.from_wave_library(library)