from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

import numpy as np

from rwif_activation_core import WaveState, encode_activation, interference_score
from rwif_memory_store import MemoryStore, estimate_background, load_memory_store, save_memory_store

_SEMANTIC_MEMORY_METADATA_KEY = "rwif_semantic_memory"
_CONFLICT_RELATION_TYPES = {"contradicts"}
_SUPPORT_RELATION_TYPES = {"supports", "supported_by", "evidence_for", "same_as", "restates"}
_QUERY_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")
_QUERY_STOPWORDS = {"and", "are", "for", "from", "how", "its", "the", "this", "what", "when", "where", "which", "who", "why", "with"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _unique_strings(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _merge_relations(existing: tuple[SemanticRelation, ...], incoming: tuple[SemanticRelation, ...]) -> tuple[SemanticRelation, ...]:
    merged: list[SemanticRelation] = []
    seen: set[tuple[str, str, float]] = set()
    for relation in [*existing, *incoming]:
        key = (relation.relation_type, relation.target_memory_id, relation.weight)
        if key in seen:
            continue
        seen.add(key)
        merged.append(relation)
    return tuple(merged)


def _merge_provenance(existing: tuple[ProvenanceRef, ...], incoming: tuple[ProvenanceRef, ...]) -> tuple[ProvenanceRef, ...]:
    merged: list[ProvenanceRef] = []
    seen: set[tuple[str, str, str | None, str | None]] = set()
    for reference in [*existing, *incoming]:
        key = (reference.source_id, reference.source_type, reference.locator, reference.quoted_text)
        if key in seen:
            continue
        seen.add(key)
        merged.append(reference)
    return tuple(merged)


@dataclass(frozen=True)
class ProvenanceRef:
    source_id: str
    source_type: str = "document"
    locator: str | None = None
    quoted_text: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_id": self.source_id,
            "source_type": self.source_type,
        }
        if self.locator is not None:
            payload["locator"] = self.locator
        if self.quoted_text is not None:
            payload["quoted_text"] = self.quoted_text
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ProvenanceRef:
        return cls(
            source_id=str(payload["source_id"]),
            source_type=str(payload.get("source_type", "document")),
            locator=None if payload.get("locator") is None else str(payload["locator"]),
            quoted_text=None if payload.get("quoted_text") is None else str(payload["quoted_text"]),
            confidence=None if payload.get("confidence") is None else float(payload["confidence"]),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
        )


@dataclass(frozen=True)
class SemanticRelation:
    relation_type: str
    target_memory_id: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "relation_type": self.relation_type,
            "target_memory_id": self.target_memory_id,
            "weight": self.weight,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> SemanticRelation:
        return cls(
            relation_type=str(payload["relation_type"]),
            target_memory_id=str(payload["target_memory_id"]),
            weight=float(payload.get("weight", 1.0)),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
        )


@dataclass(frozen=True)
class SemanticMemoryObject:
    memory_id: str
    revision: int
    title: str
    canonical_text: str
    kind: str = "concept"
    summary: str | None = None
    facts: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    relations: tuple[SemanticRelation, ...] = ()
    provenance: tuple[ProvenanceRef, ...] = ()
    status: str = "active"
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    source_model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "memory_id": self.memory_id,
            "revision": self.revision,
            "title": self.title,
            "canonical_text": self.canonical_text,
            "kind": self.kind,
            "facts": list(self.facts),
            "tags": list(self.tags),
            "relations": [relation.to_payload() for relation in self.relations],
            "provenance": [reference.to_payload() for reference in self.provenance],
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.summary is not None:
            payload["summary"] = self.summary
        if self.source_model is not None:
            payload["source_model"] = self.source_model
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> SemanticMemoryObject:
        return cls(
            memory_id=str(payload["memory_id"]),
            revision=int(payload.get("revision", 1)),
            title=str(payload.get("title", payload.get("memory_id", "memory"))),
            canonical_text=str(payload["canonical_text"]),
            kind=str(payload.get("kind", "concept")),
            summary=None if payload.get("summary") is None else str(payload["summary"]),
            facts=tuple(str(item) for item in payload.get("facts", [])),
            tags=tuple(str(item) for item in payload.get("tags", [])),
            relations=tuple(
                SemanticRelation.from_payload(item)
                for item in payload.get("relations", [])
                if isinstance(item, dict)
            ),
            provenance=tuple(
                ProvenanceRef.from_payload(item)
                for item in payload.get("provenance", [])
                if isinstance(item, dict)
            ),
            status=str(payload.get("status", "active")),
            created_at=str(payload.get("created_at", _utc_now())),
            updated_at=str(payload.get("updated_at", payload.get("created_at", _utc_now()))),
            source_model=None if payload.get("source_model") is None else str(payload["source_model"]),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
        )

    def revised(
        self,
        *,
        title: str | None = None,
        canonical_text: str | None = None,
        kind: str | None = None,
        summary: str | None = None,
        facts: tuple[str, ...] | None = None,
        tags: tuple[str, ...] | None = None,
        relations: tuple[SemanticRelation, ...] | None = None,
        provenance: tuple[ProvenanceRef, ...] | None = None,
        status: str | None = None,
        source_model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticMemoryObject:
        return replace(
            self,
            revision=self.revision + 1,
            title=self.title if title is None else title,
            canonical_text=self.canonical_text if canonical_text is None else canonical_text,
            kind=self.kind if kind is None else kind,
            summary=self.summary if summary is None else summary,
            facts=self.facts if facts is None else facts,
            tags=self.tags if tags is None else tags,
            relations=self.relations if relations is None else relations,
            provenance=self.provenance if provenance is None else provenance,
            status=self.status if status is None else status,
            updated_at=_utc_now(),
            source_model=self.source_model if source_model is None else source_model,
            metadata=dict(self.metadata if metadata is None else metadata),
        )


@dataclass(frozen=True)
class SemanticQueryResult:
    memory: SemanticMemoryObject
    score: float
    state: WaveState


@dataclass(frozen=True)
class SemanticAnswerResult:
    question: str
    answer_text: str
    primary_memory: SemanticMemoryObject
    supporting_memories: tuple[SemanticQueryResult, ...]
    conflicting_memories: tuple[SemanticMemoryObject, ...]
    relation_links: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SemanticEvidenceRoute:
    memory: SemanticMemoryObject
    score: float
    route: str
    reasons: tuple[str, ...] = ()
    origin: str = "retrieval"
    relation_types: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "memory": self.memory.to_payload(),
            "score": self.score,
            "route": self.route,
            "reasons": list(self.reasons),
            "origin": self.origin,
            "relation_types": list(self.relation_types),
        }


@dataclass(frozen=True)
class SemanticReasoningPacket:
    question: str
    primary_memory: SemanticMemoryObject
    primary_score: float
    supporting_evidence: tuple[SemanticEvidenceRoute, ...]
    neutral_evidence: tuple[SemanticEvidenceRoute, ...]
    conflicting_evidence: tuple[SemanticEvidenceRoute, ...]
    relation_links: tuple[dict[str, Any], ...]
    routing_summary: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "primary_memory": self.primary_memory.to_payload(),
            "primary_score": self.primary_score,
            "supporting_evidence": [item.to_payload() for item in self.supporting_evidence],
            "neutral_evidence": [item.to_payload() for item in self.neutral_evidence],
            "conflicting_evidence": [item.to_payload() for item in self.conflicting_evidence],
            "relation_links": list(self.relation_links),
            "routing_summary": dict(self.routing_summary),
        }


@dataclass(frozen=True)
class ConsolidationCandidate:
    memory_ids: tuple[str, ...]
    similarity: float
    shared_tags: tuple[str, ...]
    kind: str | None = None


@dataclass
class SemanticMemoryStore:
    memory_store: MemoryStore

    @property
    def metadata(self) -> dict[str, Any]:
        return self.memory_store.metadata

    @property
    def background(self) -> np.ndarray | None:
        return self.memory_store.background

    @property
    def vector_length(self) -> int | None:
        return self.memory_store.vector_length

    @property
    def revision_count(self) -> int:
        return len(self.memory_store.records)

    @property
    def active_memories(self) -> tuple[SemanticMemoryObject, ...]:
        return tuple(payload for _, payload, _ in self._iter_active_records())

    @classmethod
    def empty(cls, *, background: np.ndarray | None = None, metadata: dict[str, Any] | None = None) -> SemanticMemoryStore:
        store = MemoryStore(background=background, metadata=dict(metadata or {}))
        store.metadata.setdefault("format", "rwif_semantic_memory")
        store.metadata.setdefault("semantic_memory_version", 1)
        return cls(memory_store=store)

    @classmethod
    def from_objects(
        cls,
        *,
        provider,
        objects: list[SemanticMemoryObject],
        calibration_texts: list[str] | None = None,
        top_k_waves: int = 128,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticMemoryStore:
        if not objects:
            raise ValueError("objects must not be empty")
        if calibration_texts:
            background = estimate_background(provider.encode_texts(calibration_texts))
        else:
            background = estimate_background(provider.encode_texts([item.canonical_text for item in objects]))
        semantic_store = cls.empty(background=background, metadata=metadata)
        for item in objects:
            activation = np.asarray(provider.encode_text(item.canonical_text), dtype=np.float64)
            semantic_store.add_object_activation(item, activation=activation, top_k=top_k_waves)
        return semantic_store

    @classmethod
    def from_memory_store(cls, memory_store: MemoryStore) -> SemanticMemoryStore:
        memory_store.metadata.setdefault("format", "rwif_semantic_memory")
        memory_store.metadata.setdefault("semantic_memory_version", 1)
        return cls(memory_store=memory_store)

    def add_object_activation(
        self,
        memory: SemanticMemoryObject,
        *,
        activation: np.ndarray,
        top_k: int = 128,
    ) -> SemanticMemoryObject:
        semantic_payload = memory.to_payload()
        record_metadata = dict(memory.metadata)
        record_metadata[_SEMANTIC_MEMORY_METADATA_KEY] = semantic_payload
        self.memory_store.add_activation(
            activation=np.asarray(activation, dtype=np.float64),
            record_id=self._record_id_for(memory.memory_id, memory.revision),
            text=memory.canonical_text,
            top_k=top_k,
            metadata=record_metadata,
            source=memory.source_model,
            label=f"{memory.memory_id}@r{memory.revision}",
        )
        return memory

    def revise_object(
        self,
        memory_id: str,
        *,
        activation: np.ndarray | None = None,
        top_k: int | None = None,
        title: str | None = None,
        canonical_text: str | None = None,
        kind: str | None = None,
        summary: str | None = None,
        facts: tuple[str, ...] | None = None,
        tags: tuple[str, ...] | None = None,
        relations: tuple[SemanticRelation, ...] | None = None,
        provenance: tuple[ProvenanceRef, ...] | None = None,
        status: str | None = None,
        source_model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticMemoryObject:
        current_record, current_memory, _ = self._latest_record_for(memory_id)
        revised = current_memory.revised(
            title=title,
            canonical_text=canonical_text,
            kind=kind,
            summary=summary,
            facts=facts,
            tags=tags,
            relations=relations,
            provenance=provenance,
            status=status,
            source_model=source_model,
            metadata=metadata,
        )
        if activation is None:
            activation_vector = current_record.state.reconstruct()
            if self.background is not None:
                activation_vector = activation_vector + np.asarray(self.background, dtype=np.float64)
        else:
            activation_vector = np.asarray(activation, dtype=np.float64)
        self.add_object_activation(revised, activation=activation_vector, top_k=current_record.state.top_k if top_k is None else top_k)
        return revised

    def merge_update(
        self,
        memory_id: str,
        *,
        activation: np.ndarray | None = None,
        top_k: int | None = None,
        title: str | None = None,
        canonical_text: str | None = None,
        kind: str | None = None,
        summary: str | None = None,
        facts: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        relations: tuple[SemanticRelation, ...] = (),
        provenance: tuple[ProvenanceRef, ...] = (),
        source_model: str | None = None,
        metadata_patch: dict[str, Any] | None = None,
    ) -> SemanticMemoryObject:
        current = self.get_object(memory_id)
        merged_metadata = dict(current.metadata)
        if metadata_patch:
            merged_metadata.update(metadata_patch)
        return self.revise_object(
            memory_id,
            activation=activation,
            top_k=top_k,
            title=title,
            canonical_text=canonical_text,
            kind=kind,
            summary=current.summary if summary is None else summary,
            facts=_unique_strings([*current.facts, *facts]),
            tags=_unique_strings([*current.tags, *tags]),
            relations=_merge_relations(current.relations, relations),
            provenance=_merge_provenance(current.provenance, provenance),
            source_model=source_model,
            metadata=merged_metadata,
        )

    def deprecate_object(
        self,
        memory_id: str,
        *,
        reason: str,
        replacement_memory_id: str | None = None,
        activation: np.ndarray | None = None,
        top_k: int | None = None,
        source_model: str | None = None,
    ) -> SemanticMemoryObject:
        current = self.get_object(memory_id)
        new_relations: tuple[SemanticRelation, ...] = ()
        if replacement_memory_id:
            new_relations = (SemanticRelation(relation_type="superseded_by", target_memory_id=replacement_memory_id),)
        metadata = dict(current.metadata)
        metadata["deprecation_reason"] = str(reason)
        return self.revise_object(
            memory_id,
            activation=activation,
            top_k=top_k,
            status="deprecated",
            relations=_merge_relations(current.relations, new_relations),
            source_model=source_model,
            metadata=metadata,
        )

    def register_contradiction(
        self,
        memory_id: str,
        conflicting_memory_id: str,
        *,
        reason: str | None = None,
        weight: float = 1.0,
        source_model: str | None = None,
    ) -> tuple[SemanticMemoryObject, SemanticMemoryObject]:
        left = self.get_object(memory_id)
        right = self.get_object(conflicting_memory_id)
        relation_metadata = {} if reason is None else {"reason": str(reason)}

        left_meta = dict(left.metadata)
        right_meta = dict(right.metadata)
        left_conflicts = list(left_meta.get("conflicts", [])) if isinstance(left_meta.get("conflicts", []), list) else []
        right_conflicts = list(right_meta.get("conflicts", [])) if isinstance(right_meta.get("conflicts", []), list) else []
        left_conflicts.append({"memory_id": conflicting_memory_id, "reason": reason})
        right_conflicts.append({"memory_id": memory_id, "reason": reason})
        left_meta["conflicts"] = left_conflicts
        right_meta["conflicts"] = right_conflicts

        left_revision = self.revise_object(
            memory_id,
            relations=_merge_relations(
                left.relations,
                (SemanticRelation(relation_type="contradicts", target_memory_id=conflicting_memory_id, weight=weight, metadata=relation_metadata),),
            ),
            source_model=source_model,
            metadata=left_meta,
        )
        right_revision = self.revise_object(
            conflicting_memory_id,
            relations=_merge_relations(
                right.relations,
                (SemanticRelation(relation_type="contradicts", target_memory_id=memory_id, weight=weight, metadata=relation_metadata),),
            ),
            source_model=source_model,
            metadata=right_meta,
        )
        return left_revision, right_revision

    def get_object(self, memory_id: str, revision: int | None = None) -> SemanticMemoryObject:
        if revision is None:
            _, payload, _ = self._latest_record_for(memory_id)
            return payload
        target_record_id = self._record_id_for(memory_id, revision)
        for record in self.memory_store.records:
            if record.record_id == target_record_id:
                return self._payload_from_record(record)
        raise KeyError(f"No semantic memory revision {revision} for {memory_id}")

    def history(self, memory_id: str) -> tuple[SemanticMemoryObject, ...]:
        records = [
            self._payload_from_record(record)
            for record in self.memory_store.records
            if self._payload_from_record(record).memory_id == memory_id
        ]
        records.sort(key=lambda item: item.revision)
        if not records:
            raise KeyError(f"No semantic memory found for {memory_id}")
        return tuple(records)

    def query_activation(self, activation: np.ndarray, *, top_k: int = 5, include_inactive: bool = False) -> list[SemanticQueryResult]:
        query_state = encode_activation(
            np.asarray(activation, dtype=np.float64),
            background=self.background,
            top_k=self._default_query_top_k(),
            label="query",
        )
        return self.query_state(query_state, top_k=top_k, include_inactive=include_inactive)

    def query_text(self, provider, text: str, *, top_k: int = 5, include_inactive: bool = False) -> list[SemanticQueryResult]:
        activation = np.asarray(provider.encode_text(text), dtype=np.float64)
        return self.query_activation(activation, top_k=top_k, include_inactive=include_inactive)

    def answer_question(
        self,
        provider,
        question: str,
        *,
        top_k: int = 5,
        support_limit: int = 3,
        include_inactive: bool = False,
    ) -> SemanticAnswerResult:
        packet = self.route_evidence(
            provider,
            question,
            top_k=top_k,
            support_limit=support_limit,
            neutral_limit=0,
            conflict_limit=max(1, top_k),
            include_inactive=include_inactive,
        )
        primary = packet.primary_memory
        supporting = tuple(
            SemanticQueryResult(
                memory=item.memory,
                score=item.score,
                state=self._latest_record_for(item.memory.memory_id)[0].state,
            )
            for item in packet.supporting_evidence
        )
        conflicting_memories = tuple(item.memory for item in packet.conflicting_evidence)
        answer_text = primary.summary or (primary.facts[0] if primary.facts else primary.canonical_text)
        if conflicting_memories:
            answer_text = f"{answer_text} Conflicting memories exist and should be reviewed."
        return SemanticAnswerResult(
            question=question,
            answer_text=answer_text,
            primary_memory=primary,
            supporting_memories=supporting,
            conflicting_memories=conflicting_memories,
            relation_links=packet.relation_links,
        )

    def route_evidence(
        self,
        provider,
        question: str,
        *,
        top_k: int = 8,
        support_limit: int = 3,
        neutral_limit: int = 3,
        conflict_limit: int = 3,
        support_threshold: float = 0.15,
        neutral_threshold: float = 0.0,
        include_inactive: bool = False,
        preferred_primary_memory_id: str | None = None,
    ) -> SemanticReasoningPacket:
        activation = np.asarray(provider.encode_text(question), dtype=np.float64)
        query_state = encode_activation(
            activation,
            background=self.background,
            top_k=self._default_query_top_k(),
            label="query",
        )
        results = self.query_state(query_state, top_k=top_k, include_inactive=include_inactive)
        if not results:
            raise ValueError("No semantic memories are available to route evidence")

        primary_result = results[0]
        if preferred_primary_memory_id is not None:
            for result in results:
                if result.memory.memory_id == preferred_primary_memory_id:
                    primary_result = result
                    break
            else:
                record, payload, _ = self._latest_record_for(preferred_primary_memory_id)
                if include_inactive or payload.status == "active":
                    primary_result = SemanticQueryResult(
                        memory=payload,
                        score=interference_score(query_state, record.state),
                        state=record.state,
                    )
        primary = primary_result.memory
        query_keywords = self._query_keywords(question)
        route_priority = {"neutral": 0, "support": 1, "conflict": 2}
        evidence_map: dict[str, dict[str, Any]] = {}

        def add_candidate(
            memory: SemanticMemoryObject,
            *,
            score: float,
            origin: str,
            relation_types: tuple[str, ...] = (),
        ) -> None:
            classification = self._classify_evidence_candidate(
                primary=primary,
                candidate=memory,
                score=score,
                relation_types=relation_types,
                query_keywords=query_keywords,
                support_threshold=support_threshold,
                neutral_threshold=neutral_threshold,
            )
            if classification is None:
                return
            route, reasons = classification
            bucket = evidence_map.setdefault(
                memory.memory_id,
                {
                    "memory": memory,
                    "score": score,
                    "route": route,
                    "reasons": [],
                    "origins": set(),
                    "relation_types": set(),
                },
            )
            bucket["memory"] = memory
            bucket["score"] = max(float(bucket["score"]), float(score))
            if route_priority[route] > route_priority[bucket["route"]]:
                bucket["route"] = route
            bucket["reasons"].extend(reasons)
            bucket["origins"].add(origin)
            bucket["relation_types"].update(relation_types)

        add_candidate(primary, score=primary_result.score, origin="retrieval")
        for result in results[1:]:
            relation_types = self._relation_types_between(primary.memory_id, result.memory.memory_id)
            add_candidate(result.memory, score=result.score, origin="retrieval", relation_types=relation_types)

        for relation in primary.relations:
            try:
                target_record, target_memory, _ = self._latest_record_for(relation.target_memory_id)
            except KeyError:
                continue
            if not include_inactive and target_memory.status != "active":
                continue
            relation_types = self._relation_types_between(primary.memory_id, target_memory.memory_id)
            relation_score = interference_score(query_state, target_record.state)
            add_candidate(target_memory, score=relation_score, origin="relation", relation_types=relation_types)

        routed_items = [
            SemanticEvidenceRoute(
                memory=item["memory"],
                score=float(item["score"]),
                route=str(item["route"]),
                reasons=_unique_strings(item["reasons"]),
                origin="|".join(sorted(str(value) for value in item["origins"])),
                relation_types=tuple(sorted(str(value) for value in item["relation_types"])),
            )
            for item in evidence_map.values()
        ]
        primary_support = [item for item in routed_items if item.memory.memory_id == primary.memory_id]
        supporting_rest = [item for item in routed_items if item.route == "support" and item.memory.memory_id != primary.memory_id]
        supporting_rest.sort(key=lambda item: item.score, reverse=True)
        supporting = tuple([*primary_support, *supporting_rest][: max(1, support_limit)])

        neutral = [item for item in routed_items if item.route == "neutral"]
        neutral.sort(key=lambda item: item.score, reverse=True)
        neutral = tuple(neutral[: max(0, neutral_limit)])

        conflicting = [item for item in routed_items if item.route == "conflict"]
        conflicting.sort(key=lambda item: item.score, reverse=True)
        conflicting = tuple(conflicting[: max(0, conflict_limit)])

        relation_links = self._build_relation_links(primary, evidence_map)
        return SemanticReasoningPacket(
            question=question,
            primary_memory=primary,
            primary_score=primary_result.score,
            supporting_evidence=supporting,
            neutral_evidence=neutral,
            conflicting_evidence=conflicting,
            relation_links=relation_links,
            routing_summary={
                "top_k": top_k,
                "support_threshold": support_threshold,
                "neutral_threshold": neutral_threshold,
                "preferred_primary_memory_id": preferred_primary_memory_id,
                "supporting_count": len(supporting),
                "neutral_count": len(neutral),
                "conflicting_count": len(conflicting),
            },
        )

    def suggest_consolidation_candidates(
        self,
        *,
        min_similarity: float = 0.92,
        min_shared_tags: int = 1,
        require_same_kind: bool = True,
    ) -> tuple[ConsolidationCandidate, ...]:
        active_records = list(self._iter_active_records())
        if len(active_records) < 2:
            return ()

        parent = {payload.memory_id: payload.memory_id for _, payload, _ in active_records}

        def find(memory_id: str) -> str:
            while parent[memory_id] != memory_id:
                parent[memory_id] = parent[parent[memory_id]]
                memory_id = parent[memory_id]
            return memory_id

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        pair_scores: dict[tuple[str, str], tuple[float, tuple[str, ...]]] = {}
        for index, (left_record, left_payload, _) in enumerate(active_records):
            left_tags = set(left_payload.tags)
            for right_record, right_payload, _ in active_records[index + 1 :]:
                if require_same_kind and left_payload.kind != right_payload.kind:
                    continue
                shared_tags = tuple(sorted(left_tags.intersection(right_payload.tags)))
                if len(shared_tags) < min_shared_tags:
                    continue
                similarity = interference_score(left_record.state, right_record.state)
                if similarity < min_similarity:
                    continue
                pair_scores[(left_payload.memory_id, right_payload.memory_id)] = (similarity, shared_tags)
                union(left_payload.memory_id, right_payload.memory_id)

        groups: dict[str, list[str]] = {}
        for _, payload, _ in active_records:
            root = find(payload.memory_id)
            groups.setdefault(root, []).append(payload.memory_id)

        active_map = {payload.memory_id: payload for _, payload, _ in active_records}
        candidates: list[ConsolidationCandidate] = []
        for memory_ids in groups.values():
            if len(memory_ids) < 2:
                continue
            ids = tuple(sorted(memory_ids))
            similarities: list[float] = []
            shared_tag_values: set[str] = set(active_map[ids[0]].tags)
            for left_index, left_id in enumerate(ids):
                shared_tag_values.intersection_update(active_map[left_id].tags)
                for right_id in ids[left_index + 1 :]:
                    key = (left_id, right_id) if (left_id, right_id) in pair_scores else (right_id, left_id)
                    if key in pair_scores:
                        similarities.append(pair_scores[key][0])
            candidates.append(
                ConsolidationCandidate(
                    memory_ids=ids,
                    similarity=float(sum(similarities) / len(similarities)) if similarities else 1.0,
                    shared_tags=tuple(sorted(shared_tag_values)),
                    kind=active_map[ids[0]].kind,
                )
            )
        candidates.sort(key=lambda item: (item.similarity, len(item.memory_ids)), reverse=True)
        return tuple(candidates)

    def consolidate_candidate(
        self,
        candidate: ConsolidationCandidate,
        *,
        top_k: int | None = None,
        source_model: str | None = None,
    ) -> SemanticMemoryObject:
        if len(candidate.memory_ids) < 2:
            raise ValueError("Consolidation requires at least two memory ids")
        anchor_id = candidate.memory_ids[0]
        anchor = self.get_object(anchor_id)
        latest_records = {payload.memory_id: (record, payload) for record, payload, _ in self._iter_active_records()}
        merged_facts = list(anchor.facts)
        merged_tags = list(anchor.tags)
        merged_relations = anchor.relations
        merged_provenance = anchor.provenance
        metadata_patch = dict(anchor.metadata)
        metadata_patch["consolidated_from"] = list(candidate.memory_ids)

        activation_vectors: list[np.ndarray] = []
        for memory_id in candidate.memory_ids:
            record, payload = latest_records[memory_id]
            activation_vector = record.state.reconstruct()
            if self.background is not None:
                activation_vector = activation_vector + np.asarray(self.background, dtype=np.float64)
            activation_vectors.append(activation_vector)
            if memory_id == anchor_id:
                continue
            merged_facts.extend(payload.facts)
            merged_tags.extend(payload.tags)
            merged_relations = _merge_relations(merged_relations, payload.relations)
            merged_provenance = _merge_provenance(merged_provenance, payload.provenance)

        merged_activation = np.mean(np.stack(activation_vectors, axis=0), axis=0)
        merged = self.merge_update(
            anchor_id,
            activation=merged_activation,
            top_k=top_k,
            facts=tuple(merged_facts),
            tags=tuple(merged_tags),
            relations=merged_relations,
            provenance=merged_provenance,
            source_model=source_model,
            metadata_patch=metadata_patch,
        )
        for memory_id in candidate.memory_ids[1:]:
            self.deprecate_object(
                memory_id,
                reason=f"Consolidated into {anchor_id}",
                replacement_memory_id=anchor_id,
                source_model=source_model,
            )
        return merged

    def query_state(self, query_state: WaveState, *, top_k: int = 5, include_inactive: bool = False) -> list[SemanticQueryResult]:
        candidates = self._iter_all_records() if include_inactive else self._iter_active_records()
        results = [
            SemanticQueryResult(memory=payload, score=interference_score(query_state, record.state), state=record.state)
            for record, payload, _ in candidates
        ]
        results.sort(key=lambda item: item.score, reverse=True)
        return results[: max(1, top_k)]

    def _default_query_top_k(self) -> int:
        if not self.memory_store.records:
            return 1
        return max(record.state.top_k for record in self.memory_store.records)

    def _latest_record_for(self, memory_id: str):
        latest: tuple[Any, Any, Any] | None = None
        for record, payload, rank in self._iter_all_records():
            if payload.memory_id != memory_id:
                continue
            if latest is None or payload.revision > latest[1].revision:
                latest = (record, payload, rank)
        if latest is None:
            raise KeyError(f"No semantic memory found for {memory_id}")
        return latest

    def _iter_all_records(self):
        for index, record in enumerate(self.memory_store.records):
            yield record, self._payload_from_record(record), index

    def _iter_active_records(self):
        latest_by_id: dict[str, tuple[Any, Any, Any]] = {}
        for record, payload, rank in self._iter_all_records():
            current = latest_by_id.get(payload.memory_id)
            if current is None or payload.revision > current[1].revision:
                latest_by_id[payload.memory_id] = (record, payload, rank)
        active_records = [value for value in latest_by_id.values() if value[1].status == "active"]
        active_records.sort(key=lambda item: item[2])
        return tuple(active_records)

    def _safe_title_for(self, memory_id: str) -> str | None:
        try:
            return self.get_object(memory_id).title
        except KeyError:
            return None

    def _build_relation_links(self, primary: SemanticMemoryObject, evidence_map: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], ...]:
        links: list[dict[str, Any]] = []
        for relation in primary.relations:
            bucket = evidence_map.get(relation.target_memory_id)
            target_status = None
            try:
                target_status = self.get_object(relation.target_memory_id).status
            except KeyError:
                target_status = None
            links.append(
                {
                    "relation_type": relation.relation_type,
                    "target_memory_id": relation.target_memory_id,
                    "weight": relation.weight,
                    "target_title": self._safe_title_for(relation.target_memory_id),
                    "target_status": target_status,
                    "route": None if bucket is None else bucket.get("route"),
                    "query_score": None if bucket is None else float(bucket.get("score", 0.0)),
                    "metadata": dict(relation.metadata),
                }
            )
        return tuple(links)

    def _classify_evidence_candidate(
        self,
        *,
        primary: SemanticMemoryObject,
        candidate: SemanticMemoryObject,
        score: float,
        relation_types: tuple[str, ...],
        query_keywords: set[str],
        support_threshold: float,
        neutral_threshold: float,
    ) -> tuple[str, tuple[str, ...]] | None:
        reasons: list[str] = []
        relation_type_set = set(relation_types)
        shared_tags = tuple(sorted(set(primary.tags).intersection(candidate.tags)))
        candidate_keywords = self._query_keywords(candidate.canonical_text)
        shared_query_keywords = tuple(sorted(query_keywords.intersection(candidate_keywords)))

        if candidate.memory_id == primary.memory_id:
            return "support", ("primary retrieval match",)
        if relation_type_set.intersection(_CONFLICT_RELATION_TYPES):
            reasons.append("direct contradiction relation")
            if candidate.metadata.get("conflicts"):
                reasons.append("conflict metadata present")
            return "conflict", tuple(reasons)
        if relation_type_set.intersection(_SUPPORT_RELATION_TYPES):
            reasons.append("supportive relation link")
            return "support", tuple(reasons)
        if shared_tags:
            reasons.append(f"shared tags: {', '.join(shared_tags[:3])}")
        if shared_query_keywords:
            reasons.append(f"shared query terms: {', '.join(shared_query_keywords[:3])}")
        if score >= support_threshold and len(shared_query_keywords) >= 2:
            reasons.append(f"score {score:.3f} reached support threshold")
            return "support", tuple(reasons)
        if relation_types:
            reasons.append(f"related via {', '.join(sorted(relation_type_set))}")
            return "neutral", tuple(reasons)
        if score >= neutral_threshold:
            reasons.append(f"score {score:.3f} reached neutral threshold")
            return "neutral", tuple(reasons)
        return None

    def _relation_types_between(self, primary_memory_id: str, candidate_memory_id: str) -> tuple[str, ...]:
        relation_types: list[str] = []
        try:
            primary = self.get_object(primary_memory_id)
            relation_types.extend(
                relation.relation_type
                for relation in primary.relations
                if relation.target_memory_id == candidate_memory_id
            )
        except KeyError:
            pass
        try:
            candidate = self.get_object(candidate_memory_id)
            relation_types.extend(
                relation.relation_type
                for relation in candidate.relations
                if relation.target_memory_id == primary_memory_id
            )
        except KeyError:
            pass
        return _unique_strings(relation_types)

    def _query_keywords(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in _QUERY_TOKEN_RE.findall(text)
            if token.lower() not in _QUERY_STOPWORDS
        }

    def _payload_from_record(self, record) -> SemanticMemoryObject:
        payload = record.metadata.get(_SEMANTIC_MEMORY_METADATA_KEY)
        if not isinstance(payload, dict):
            raise ValueError(f"Record {record.record_id} is missing semantic memory metadata")
        return SemanticMemoryObject.from_payload(payload)

    @staticmethod
    def _record_id_for(memory_id: str, revision: int) -> str:
        return f"{memory_id}@r{revision:06d}"


def save_semantic_memory_store(path: str | Path, store: SemanticMemoryStore) -> None:
    save_memory_store(path, store.memory_store)


def load_semantic_memory_store(path: str | Path) -> SemanticMemoryStore:
    return SemanticMemoryStore.from_memory_store(load_memory_store(path))