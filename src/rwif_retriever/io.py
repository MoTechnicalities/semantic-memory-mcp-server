from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .providers import ArrayActivationProvider, TransformersActivationProvider
from .retriever import QueryCase, TextMemorySeed


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {input_path}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object on line {line_number} of {input_path}")
            rows.append(payload)
    return rows


def build_text_memory_seeds(rows: list[dict[str, Any]]) -> list[TextMemorySeed]:
    seeds: list[TextMemorySeed] = []
    for index, row in enumerate(rows):
        text = str(row.get("text", "")).strip()
        if not text:
            raise ValueError(f"Record at index {index} is missing a non-empty 'text' field")
        record_id = str(row.get("record_id", f"record-{index:05d}"))
        metadata = dict(row.get("metadata", {})) if isinstance(row.get("metadata", {}), dict) else {}
        source = row.get("source")
        seeds.append(TextMemorySeed(record_id=record_id, text=text, metadata=metadata, source=source))
    return seeds


def build_query_cases(rows: list[dict[str, Any]]) -> list[QueryCase]:
    cases: list[QueryCase] = []
    for index, row in enumerate(rows):
        query_text = str(row.get("query_text", row.get("text", ""))).strip()
        if not query_text:
            raise ValueError(f"Query at index {index} is missing 'query_text' or 'text'")
        expected = row.get("expected_record_id")
        cases.append(QueryCase(query_text=query_text, expected_record_id=None if expected is None else str(expected)))
    return cases


def build_provider(
    *,
    provider_kind: str,
    rows: list[dict[str, Any]],
    extra_rows: list[dict[str, Any]] | None = None,
    model_id: str | None = None,
    layer_index: int = -1,
    pooling: str = "last_token",
    device: str | None = None,
    max_length: int = 256,
):
    if provider_kind == "array":
        activations: dict[str, np.ndarray] = {}
        for payload in [*rows, *(extra_rows or [])]:
            text = str(payload.get("text", payload.get("query_text", ""))).strip()
            if not text:
                continue
            activation = payload.get("activation")
            if activation is None:
                raise ValueError("Array provider requires an 'activation' field for every record and query")
            activations[text] = np.asarray(activation, dtype=np.float64)
        return ArrayActivationProvider(activations=activations)

    if provider_kind == "transformers":
        if not model_id:
            raise ValueError("Transformers provider requires --model-id")
        return TransformersActivationProvider(
            model_id=model_id,
            layer_index=layer_index,
            pooling=pooling,
            device=device,
            max_length=max_length,
        )

    raise ValueError(f"Unsupported provider kind: {provider_kind}")