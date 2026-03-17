from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np


RWIF_MAGIC = b"RWIFACT1"
RWIF_HEADER_STRUCT = struct.Struct("<8sI")
RWIF_UNIT_STRUCT = struct.Struct("<Id")


@lru_cache(maxsize=32)
def _dct_basis(vector_length: int) -> np.ndarray:
    if vector_length <= 0:
        raise ValueError("vector_length must be positive")
    positions = np.arange(vector_length, dtype=np.float64)
    frequencies = np.arange(vector_length, dtype=np.float64)[:, None]
    basis = np.cos(np.pi / vector_length * (positions + 0.5) * frequencies)
    basis[0, :] *= np.sqrt(1.0 / vector_length)
    if vector_length > 1:
        basis[1:, :] *= np.sqrt(2.0 / vector_length)
    return basis


def _dct(vector: np.ndarray) -> np.ndarray:
    basis = _dct_basis(int(vector.shape[0]))
    return basis @ vector


def _idct(coefficients: np.ndarray) -> np.ndarray:
    basis = _dct_basis(int(coefficients.shape[0]))
    return basis.T @ coefficients


@dataclass(frozen=True)
class AtomicWaveUnit:
    frequency_index: int
    amplitude: float

    def omega(self, vector_length: int) -> float:
        return float(np.pi * self.frequency_index / vector_length)

    def phi(self, vector_length: int) -> float:
        return float((np.pi * self.frequency_index / (2.0 * vector_length)) + (np.pi / 2.0))

    def evaluate(self, positions: np.ndarray, vector_length: int) -> np.ndarray:
        return self.amplitude * np.sin((self.omega(vector_length) * positions) + self.phi(vector_length))


@dataclass(frozen=True)
class WaveState:
    vector_length: int
    units: tuple[AtomicWaveUnit, ...]
    label: str | None = None
    centered_norm: float = 0.0
    original_norm: float = 0.0
    top_k: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def reconstruct(self) -> np.ndarray:
        return decode_wave_state(self)

    def to_header_entry(self, awu_offset: int) -> dict[str, Any]:
        return {
            "label": self.label,
            "vector_length": self.vector_length,
            "centered_norm": self.centered_norm,
            "original_norm": self.original_norm,
            "top_k": self.top_k,
            "metadata": self.metadata,
            "awu_offset": awu_offset,
            "awu_count": len(self.units),
        }


@dataclass(frozen=True)
class SearchMatch:
    label: str | None
    score: float
    state: WaveState


@dataclass(frozen=True)
class WaveLibrary:
    states: tuple[WaveState, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def search(self, query_state: WaveState, top_k: int = 5) -> list[SearchMatch]:
        matches = [
            SearchMatch(label=state.label, score=interference_score(query_state, state), state=state)
            for state in self.states
        ]
        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[: max(1, top_k)]


def encode_activation(
    activation: np.ndarray,
    *,
    background: np.ndarray | None = None,
    top_k: int = 128,
    label: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> WaveState:
    vector = np.asarray(activation, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("activation must be a 1D vector")

    centered = vector.copy()
    if background is not None:
        background_vector = np.asarray(background, dtype=np.float64)
        if background_vector.shape != vector.shape:
            raise ValueError("background must have the same shape as activation")
        centered = centered - background_vector

    coefficients = _dct(centered)
    keep_count = min(max(1, top_k), coefficients.shape[0])
    top_indices = np.argpartition(np.abs(coefficients), -keep_count)[-keep_count:]
    ordered_indices = sorted(top_indices.tolist(), key=lambda index: abs(coefficients[index]), reverse=True)
    units = tuple(AtomicWaveUnit(frequency_index=index, amplitude=float(coefficients[index])) for index in ordered_indices)
    return WaveState(
        vector_length=int(vector.shape[0]),
        units=units,
        label=label,
        centered_norm=float(np.linalg.norm(centered)),
        original_norm=float(np.linalg.norm(vector)),
        top_k=keep_count,
        metadata=dict(metadata or {}),
    )


def decode_wave_state(state: WaveState) -> np.ndarray:
    coefficients = np.zeros(state.vector_length, dtype=np.float64)
    for unit in state.units:
        coefficients[unit.frequency_index] = unit.amplitude
    return _idct(coefficients)


def interference_score(left: WaveState, right: WaveState) -> float:
    left_vector = left.reconstruct()
    right_vector = right.reconstruct()
    left_norm = float(np.linalg.norm(left_vector))
    right_norm = float(np.linalg.norm(right_vector))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    score = float(np.dot(left_vector, right_vector) / (left_norm * right_norm))
    return max(-1.0, min(1.0, score))


def save_wave_library(path: str | Path, library: WaveLibrary) -> None:
    output_path = Path(path)
    awu_offset = 0
    state_headers: list[dict[str, Any]] = []
    for state in library.states:
        state_headers.append(state.to_header_entry(awu_offset))
        awu_offset += len(state.units)

    header = {
        "version": 1,
        "library_metadata": library.metadata,
        "states": state_headers,
    }
    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

    with output_path.open("wb") as handle:
        handle.write(RWIF_HEADER_STRUCT.pack(RWIF_MAGIC, len(header_bytes)))
        handle.write(header_bytes)
        for state in library.states:
            for unit in state.units:
                handle.write(RWIF_UNIT_STRUCT.pack(unit.frequency_index, unit.amplitude))


def load_wave_library(path: str | Path) -> WaveLibrary:
    input_path = Path(path)
    with input_path.open("rb") as handle:
        prefix = handle.read(RWIF_HEADER_STRUCT.size)
        magic, header_size = RWIF_HEADER_STRUCT.unpack(prefix)
        if magic != RWIF_MAGIC:
            raise ValueError("Not a RWIF activation-core file")
        header = json.loads(handle.read(header_size).decode("utf-8"))
        states: list[WaveState] = []
        for state_header in header["states"]:
            units: list[AtomicWaveUnit] = []
            for _ in range(state_header["awu_count"]):
                index, amplitude = RWIF_UNIT_STRUCT.unpack(handle.read(RWIF_UNIT_STRUCT.size))
                units.append(AtomicWaveUnit(frequency_index=index, amplitude=amplitude))
            states.append(
                WaveState(
                    vector_length=state_header["vector_length"],
                    units=tuple(units),
                    label=state_header.get("label"),
                    centered_norm=state_header.get("centered_norm", 0.0),
                    original_norm=state_header.get("original_norm", 0.0),
                    top_k=state_header.get("top_k", len(units)),
                    metadata=state_header.get("metadata", {}),
                )
            )
    return WaveLibrary(states=tuple(states), metadata=header.get("library_metadata", {}))