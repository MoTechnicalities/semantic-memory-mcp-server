from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np

from rwif_retriever.io import load_jsonl

from .store import ProvenanceRef, SemanticMemoryObject, SemanticRelation

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAIM_VERB_RE = re.compile(r"\b(is|are|was|were|be|can|uses|use|supports|means|contains|stores|works|combines|passes|flows|equals|includes|becomes|remains)\b", re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def load_semantic_jsonl(path: str) -> list[dict[str, Any]]:
    return load_jsonl(path)


def build_semantic_memory_objects(rows: list[dict[str, Any]]) -> list[SemanticMemoryObject]:
    objects: list[SemanticMemoryObject] = []
    for index, row in enumerate(rows):
        memory_id = str(row.get("memory_id", f"memory-{index:05d}")).strip()
        title = str(row.get("title", memory_id)).strip()
        canonical_text = str(row.get("canonical_text", row.get("text", ""))).strip()
        if not canonical_text:
            raise ValueError(f"Semantic memory row {index} is missing canonical_text or text")
        relations = tuple(
            SemanticRelation.from_payload(item)
            for item in row.get("relations", [])
            if isinstance(item, dict)
        )
        provenance = tuple(
            ProvenanceRef.from_payload(item)
            for item in row.get("provenance", [])
            if isinstance(item, dict)
        )
        payload: dict[str, Any] = {
            "memory_id": memory_id,
            "revision": int(row.get("revision", 1)),
            "title": title,
            "canonical_text": canonical_text,
            "kind": str(row.get("kind", "concept")),
            "status": str(row.get("status", "active")),
            "facts": [str(item) for item in row.get("facts", [])],
            "tags": [str(item) for item in row.get("tags", [])],
            "relations": [relation.to_payload() for relation in relations],
            "provenance": [reference.to_payload() for reference in provenance],
            "metadata": dict(row.get("metadata", {})) if isinstance(row.get("metadata", {}), dict) else {},
        }
        if row.get("summary") is not None:
            payload["summary"] = str(row["summary"])
        if row.get("source_model") is not None:
            payload["source_model"] = str(row["source_model"])
        if row.get("created_at") is not None:
            payload["created_at"] = str(row["created_at"])
        if row.get("updated_at") is not None:
            payload["updated_at"] = str(row["updated_at"])
        objects.append(SemanticMemoryObject.from_payload(payload))
    return objects


def activation_rows_from_semantic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("canonical_text", row.get("text", row.get("query_text", "")))).strip()
        payload: dict[str, Any] = {"text": text}
        if row.get("activation") is not None:
            payload["activation"] = np.asarray(row["activation"], dtype=np.float64).tolist()
        payloads.append(payload)
    return payloads


def build_semantic_update_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        operation = str(row.get("operation", "")).strip().lower()
        if operation not in {"revise", "merge", "deprecate", "contradict"}:
            raise ValueError(f"Semantic update row {index} has unsupported operation: {operation!r}")
        updates.append(dict(row))
    return updates


def build_semantic_object_rows_from_corpus(
    rows: list[dict[str, Any]],
    *,
    max_sentences_per_object: int = 3,
    max_chars_per_object: int = 420,
    max_facts_per_object: int = 3,
    max_tags: int = 6,
    create_document_root: bool = True,
    max_objects_per_document: int | None = None,
    deduplicate_claims: bool = True,
    min_claim_token_count: int = 5,
    dedup_similarity_threshold: float = 0.72,
) -> list[dict[str, Any]]:
    semantic_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source_id = str(row.get("source_id", row.get("doc_id", row.get("memory_id", f"document-{index:05d}")))).strip()
        if not source_id:
            raise ValueError(f"Corpus row {index} is missing a usable source id")
        title = str(row.get("title", source_id)).strip() or source_id
        text = _normalize_whitespace(str(row.get("text", row.get("canonical_text", ""))))
        if not text:
            raise ValueError(f"Corpus row {index} is missing text")

        sentences = _split_sentences(text)
        if not sentences:
            sentences = [text]
        claims = _extract_claims(
            title=title,
            sentences=sentences,
            max_chars=max_chars_per_object,
            min_claim_token_count=min_claim_token_count,
        )
        if not claims:
            claims = [_summarize_sentences(sentences, max_sentences=max(1, max_sentences_per_object), max_chars=max_chars_per_object)]
        if max_objects_per_document is not None:
            claims = claims[: max(1, max_objects_per_document)]

        source_type = str(row.get("source_type", "document"))
        metadata = dict(row.get("metadata", {})) if isinstance(row.get("metadata", {}), dict) else {}
        base_tags = _extract_keywords(f"{title} {text}", limit=max_tags)
        activation = row.get("activation")
        document_root_id = str(row.get("document_memory_id", source_id)).strip() or source_id

        if create_document_root:
            document_summary = _summarize_sentences(sentences, max_sentences=max_facts_per_object, max_chars=max_chars_per_object)
            semantic_rows.append(
                {
                    "memory_id": document_root_id,
                    "title": title,
                    "canonical_text": document_summary,
                    "kind": str(row.get("document_kind", "document")),
                    "summary": document_summary,
                    "facts": sentences[: max(1, max_facts_per_object)],
                    "tags": list(base_tags),
                    "provenance": [
                        {
                            "source_id": source_id,
                            "source_type": source_type,
                            "locator": str(row.get("locator", "document")),
                            "quoted_text": document_summary,
                        }
                    ],
                    "metadata": {
                        **metadata,
                        "ingest_stage": "document_root",
                        "source_document_id": source_id,
                        "auto_ingested": True,
                    },
                    **({"activation": np.asarray(activation, dtype=np.float64).tolist()} if activation is not None else {}),
                }
            )

        for claim_index, claim_text in enumerate(claims):
            claim_text = _normalize_whitespace(claim_text)
            if not claim_text:
                continue
            claim_id = f"{source_id}-claim-{claim_index:03d}"
            claim_tags = _extract_keywords(f"{title} {claim_text}", limit=max_tags)
            relations: list[dict[str, Any]] = []
            if create_document_root:
                relations.append(
                    {
                        "relation_type": "source_document",
                        "target_memory_id": document_root_id,
                        "weight": 1.0,
                    }
                )
            claim_rows.append(
                {
                    "memory_id": claim_id,
                    "title": f"{title} claim {claim_index + 1}",
                    "canonical_text": claim_text,
                    "kind": str(row.get("claim_kind", "claim")),
                    "summary": claim_text,
                    "facts": [claim_text][: max(1, max_facts_per_object)],
                    "tags": list(_merge_tags(base_tags, claim_tags, max_tags=max_tags)),
                    "relations": relations,
                    "provenance": [
                        {
                            "source_id": source_id,
                            "source_type": source_type,
                            "locator": f"claim:{claim_index}",
                            "quoted_text": claim_text,
                        }
                    ],
                    "metadata": {
                        **metadata,
                        "ingest_stage": "claim",
                        "source_document_id": source_id,
                        "claim_index": claim_index,
                        "claim_signature": _claim_signature(claim_text),
                        "auto_ingested": True,
                    },
                    **({"activation": np.asarray(activation, dtype=np.float64).tolist()} if activation is not None else {}),
                }
            )

    if deduplicate_claims:
        claim_rows = _deduplicate_claim_rows(
            claim_rows,
            max_tags=max_tags,
            max_facts_per_object=max_facts_per_object,
            similarity_threshold=dedup_similarity_threshold,
        )

    semantic_rows.extend(claim_rows)
    return semantic_rows


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _split_sentences(text: str) -> list[str]:
    normalized = _normalize_whitespace(text)
    if not normalized:
        return []
    return [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(normalized) if segment.strip()]


def _extract_claims(
    *,
    title: str,
    sentences: list[str],
    max_chars: int,
    min_claim_token_count: int,
) -> list[str]:
    claims: list[str] = []
    for sentence in sentences:
        normalized = _normalize_whitespace(sentence)
        tokens = [token.lower() for token in _TOKEN_RE.findall(normalized)]
        if len(tokens) < min_claim_token_count:
            continue
        if not _CLAIM_VERB_RE.search(normalized) and not any(char.isdigit() for char in normalized):
            continue
        claims.append(_summarize_sentences([normalized], max_sentences=1, max_chars=max_chars))
    if claims:
        return claims
    fallback = _summarize_sentences(sentences, max_sentences=1, max_chars=max_chars)
    return [] if not fallback else [fallback]


def _extract_keywords(text: str, *, limit: int) -> tuple[str, ...]:
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    filtered = [token for token in tokens if token not in _STOPWORDS]
    if not filtered:
        return ()
    counts = Counter(filtered)
    first_index: dict[str, int] = {}
    for index, token in enumerate(filtered):
        first_index.setdefault(token, index)
    ordered = sorted(counts, key=lambda token: (-counts[token], first_index[token], token))
    return tuple(ordered[: max(0, limit)])


def _merge_tags(left: tuple[str, ...], right: tuple[str, ...], *, max_tags: int) -> tuple[str, ...]:
    merged: list[str] = []
    for tag in [*left, *right]:
        if tag in merged:
            continue
        merged.append(tag)
        if len(merged) >= max_tags:
            break
    return tuple(merged)


def _claim_signature(text: str) -> str:
    keywords = _extract_keywords(text, limit=12)
    return " ".join(keywords)


def _claim_token_set(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text) if token.lower() not in _STOPWORDS}


def _normalized_claim_text(text: str) -> str:
    return " ".join(sorted(_claim_token_set(text)))


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left.union(right)
    if not union:
        return 0.0
    return len(left.intersection(right)) / len(union)


def _deduplicate_claim_rows(
    claim_rows: list[dict[str, Any]],
    *,
    max_tags: int,
    max_facts_per_object: int,
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    if len(claim_rows) < 2:
        return claim_rows

    parent = list(range(len(claim_rows)))
    token_sets = [_claim_token_set(str(row.get("canonical_text", ""))) for row in claim_rows]
    normalized_claims = [_normalized_claim_text(str(row.get("canonical_text", ""))) for row in claim_rows]
    source_ids = [str(row.get("metadata", {}).get("source_document_id", "")) for row in claim_rows]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left_row in enumerate(claim_rows):
        for right_index in range(left_index + 1, len(claim_rows)):
            if source_ids[left_index] == source_ids[right_index]:
                continue
            if normalized_claims[left_index] and normalized_claims[left_index] == normalized_claims[right_index]:
                union(left_index, right_index)
                continue
            similarity = _jaccard_similarity(token_sets[left_index], token_sets[right_index])
            if similarity >= similarity_threshold:
                union(left_index, right_index)

    groups: dict[int, list[int]] = {}
    for index in range(len(claim_rows)):
        groups.setdefault(find(index), []).append(index)

    merged_rows: list[dict[str, Any]] = []
    for indices in groups.values():
        if len(indices) == 1:
            merged_rows.append(claim_rows[indices[0]])
            continue
        grouped_rows = [claim_rows[index] for index in indices]
        anchor = min(grouped_rows, key=lambda row: (len(str(row.get("canonical_text", ""))), str(row.get("memory_id", ""))))
        canonical_texts = [str(row.get("canonical_text", "")).strip() for row in grouped_rows if str(row.get("canonical_text", "")).strip()]
        unique_facts = _unique_texts(canonical_texts)
        merged_tags = _unique_texts(tag for row in grouped_rows for tag in row.get("tags", []))
        merged_relations = _merge_relation_payloads(tuple(row for row in grouped_rows for row in row.get("relations", [])))
        merged_provenance = _merge_provenance_payloads(tuple(row for row in grouped_rows for row in row.get("provenance", [])))
        source_document_ids = _unique_texts(row.get("metadata", {}).get("source_document_id", "") for row in grouped_rows)
        merged_metadata = dict(anchor.get("metadata", {}))
        merged_metadata.update(
            {
                "ingest_stage": "claim_deduplicated",
                "deduplicated_claim_count": len(grouped_rows),
                "deduplicated_claim_ids": [str(row.get("memory_id", "")) for row in grouped_rows],
                "source_document_ids": list(source_document_ids),
                "claim_signature": _claim_signature(anchor.get("canonical_text", "")),
                "auto_ingested": True,
            }
        )
        merged_row = {
            **anchor,
            "canonical_text": min(canonical_texts, key=len),
            "summary": min(canonical_texts, key=len),
            "facts": list(unique_facts[: max(1, max_facts_per_object)]),
            "tags": list(merged_tags[:max_tags]),
            "relations": list(merged_relations),
            "provenance": list(merged_provenance),
            "metadata": merged_metadata,
        }
        averaged_activation = _average_activations(grouped_rows)
        if averaged_activation is not None:
            merged_row["activation"] = averaged_activation
        merged_rows.append(merged_row)

    merged_rows.sort(key=lambda row: str(row.get("memory_id", "")))
    return merged_rows


def _unique_texts(values) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return tuple(result)


def _merge_relation_payloads(relations: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float]] = set()
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        relation_type = str(relation.get("relation_type", "")).strip()
        target_memory_id = str(relation.get("target_memory_id", "")).strip()
        weight = float(relation.get("weight", 1.0))
        key = (relation_type, target_memory_id, weight)
        if not relation_type or not target_memory_id or key in seen:
            continue
        seen.add(key)
        payload = {
            "relation_type": relation_type,
            "target_memory_id": target_memory_id,
            "weight": weight,
        }
        if isinstance(relation.get("metadata"), dict) and relation.get("metadata"):
            payload["metadata"] = dict(relation["metadata"])
        merged.append(payload)
    return tuple(merged)


def _merge_provenance_payloads(provenance: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for reference in provenance:
        if not isinstance(reference, dict):
            continue
        source_id = str(reference.get("source_id", "")).strip()
        source_type = str(reference.get("source_type", "document")).strip()
        locator = str(reference.get("locator", "")).strip()
        quoted_text = str(reference.get("quoted_text", "")).strip()
        key = (source_id, source_type, locator, quoted_text)
        if not source_id or key in seen:
            continue
        seen.add(key)
        payload = {
            "source_id": source_id,
            "source_type": source_type,
        }
        if locator:
            payload["locator"] = locator
        if quoted_text:
            payload["quoted_text"] = quoted_text
        merged.append(payload)
    return tuple(merged)


def _average_activations(rows: list[dict[str, Any]]) -> list[float] | None:
    activations = [np.asarray(row["activation"], dtype=np.float64) for row in rows if row.get("activation") is not None]
    if not activations:
        return None
    first_shape = activations[0].shape
    if any(vector.shape != first_shape for vector in activations[1:]):
        return None
    return np.mean(np.stack(activations, axis=0), axis=0).tolist()


def _summarize_sentences(sentences: list[str], *, max_sentences: int, max_chars: int) -> str:
    summary = " ".join(sentences[: max(1, max_sentences)]).strip()
    if len(summary) <= max_chars:
        return summary
    trimmed = summary[: max_chars].rstrip()
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0]
    return trimmed + "..."