from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any

from rwif_retriever import ArrayActivationProvider, TransformersActivationProvider
from rwif_semantic_memory import load_semantic_memory_store

from .semantic_memory_service import ProposalReviewPolicy, SemanticMemoryService

_CONFLICT_RELATION_TYPES = {"contradicts"}
_SUPPORT_RELATION_TYPES = {"supports", "supported_by", "evidence_for", "same_as", "restates"}
_TEXT_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
_ACCESS_MODES = {"read-only", "read-write"}


def _expand_config_path(raw_path: str, *, base_directory: Path | None = None) -> str:
    expanded = os.path.expanduser(os.path.expandvars(str(raw_path)))
    path = Path(expanded)
    if not path.is_absolute() and base_directory is not None:
        path = base_directory / path
    return str(path.resolve())


def _normalize_claim_signature(text: str) -> str:
    return _TEXT_NORMALIZE_RE.sub(" ", text.lower()).strip()


@dataclass(frozen=True)
class FederatedStoreSpec:
    store_id: str
    label: str
    path: str
    access_mode: str = "read-write"
    domain_tags: tuple[str, ...] = ()
    trust_weight: float = 1.0
    enabled: bool = True
    removable: bool = False
    required_mount_path: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        store_roots: dict[str, str] | None = None,
        base_directory: Path | None = None,
    ) -> FederatedStoreSpec:
        raw_path = payload.get("path")
        if raw_path is None:
            root_name = None if payload.get("root") is None else str(payload.get("root"))
            relative_path = payload.get("relative_path")
            if not root_name or not str(relative_path or "").strip():
                raise ValueError("store entries require either path or root plus relative_path")
            root_path = None if store_roots is None else store_roots.get(root_name)
            if root_path is None:
                raise KeyError(f"Unknown store root: {root_name}")
            raw_path = str(Path(root_path) / str(relative_path))
        access_mode = str(payload.get("access_mode", "read-write")).strip().lower()
        if access_mode not in _ACCESS_MODES:
            raise ValueError(f"Unsupported access_mode: {access_mode}")
        return cls(
            store_id=str(payload["store_id"]),
            label=str(payload.get("label") or payload["store_id"]),
            path=_expand_config_path(str(raw_path), base_directory=base_directory),
            access_mode=access_mode,
            domain_tags=tuple(str(item) for item in payload.get("domain_tags", []) if str(item).strip()),
            trust_weight=float(payload.get("trust_weight", 1.0)),
            enabled=bool(payload.get("enabled", True)),
            removable=bool(payload.get("removable", False)),
            required_mount_path=(
                None
                if payload.get("required_mount_path") is None
                else _expand_config_path(str(payload["required_mount_path"]), base_directory=base_directory)
            ),
            description=None if payload.get("description") is None else str(payload["description"]),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
        )


@dataclass
class FederatedStore:
    spec: FederatedStoreSpec
    service: SemanticMemoryService | None = None
    load_error: str | None = None

    @property
    def store_id(self) -> str:
        return self.spec.store_id

    @property
    def is_writable(self) -> bool:
        return self.spec.access_mode == "read-write"

    @property
    def is_mounted(self) -> bool:
        if self.spec.required_mount_path is None:
            return True
        return Path(self.spec.required_mount_path).exists()

    @property
    def is_available(self) -> bool:
        if not self.spec.enabled or not self.is_mounted:
            return False
        if self.service is not None:
            return True
        return Path(self.spec.path).exists()

    def summary_payload(self, *, active: bool) -> dict[str, Any]:
        payload = {
            "store_id": self.spec.store_id,
            "label": self.spec.label,
            "path": self.spec.path,
            "access_mode": self.spec.access_mode,
            "writable": self.is_writable,
            "domain_tags": list(self.spec.domain_tags),
            "trust_weight": self.spec.trust_weight,
            "enabled": self.spec.enabled,
            "active": active,
            "available": self.is_available,
            "mounted": self.is_mounted,
            "removable": self.spec.removable,
            "description": self.spec.description,
            "metadata": dict(self.spec.metadata),
        }
        if self.spec.required_mount_path is not None:
            payload["required_mount_path"] = self.spec.required_mount_path
        if self.load_error is not None:
            payload["load_error"] = self.load_error
        if self.service is not None:
            payload["service"] = self.service.summary_payload()
        return payload


@dataclass(frozen=True)
class FederatedQueryHit:
    store_id: str
    store_label: str
    memory: Any
    raw_score: float
    merged_score: float
    rank: int
    trust_weight: float
    domain_tags: tuple[str, ...] = ()
    duplicate_store_ids: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "score": self.merged_score,
            "raw_score": self.raw_score,
            "rank": self.rank,
            "memory": self.memory.to_payload(),
            "store": {
                "store_id": self.store_id,
                "label": self.store_label,
                "trust_weight": self.trust_weight,
                "domain_tags": list(self.domain_tags),
            },
            "duplicate_store_ids": list(self.duplicate_store_ids),
        }


def build_provider_from_config(provider_config: dict[str, Any]) -> tuple[Any, str, dict[str, Any]]:
    provider_name = str(provider_config.get("name", "transformers"))
    if provider_name == "array":
        activations_payload = provider_config.get("activations", {})
        if not isinstance(activations_payload, dict):
            raise ValueError("array provider requires an activations mapping")
        return ArrayActivationProvider(activations=activations_payload), "array", {"activation_count": len(activations_payload)}
    normalized_config = {
        "model_id": str(provider_config.get("model_id", "sentence-transformers/all-MiniLM-L6-v2")),
        "layer_index": int(provider_config.get("layer_index", -1)),
        "pooling": str(provider_config.get("pooling", "mean")),
        "device": None if provider_config.get("device") is None else str(provider_config.get("device")),
        "max_length": int(provider_config.get("max_length", 128)),
    }
    return (
        TransformersActivationProvider(**normalized_config),
        "transformers",
        normalized_config,
    )


@dataclass
class FederatedSemanticMemoryBroker:
    stores: dict[str, FederatedStore]
    active_store_ids: list[str]
    registry_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def review_policy(self) -> ProposalReviewPolicy:
        for store in self.stores.values():
            if store.service is not None:
                return store.service.review_policy
        return ProposalReviewPolicy()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        default_provider_config: dict[str, Any] | None = None,
        review_policy: ProposalReviewPolicy | None = None,
        default_active_store_ids: list[str] | None = None,
    ) -> FederatedSemanticMemoryBroker:
        resolved_config_path = Path(config_path)
        payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        stores: dict[str, FederatedStore] = {}
        review = review_policy or ProposalReviewPolicy()
        base_directory = resolved_config_path.parent
        store_roots_payload = payload.get("store_roots") if isinstance(payload.get("store_roots"), dict) else {}
        store_roots = {
            str(name): _expand_config_path(str(root_path), base_directory=base_directory)
            for name, root_path in store_roots_payload.items()
        }
        registry_default_provider = payload.get("default_provider") if isinstance(payload.get("default_provider"), dict) else {}
        merged_default_provider = dict(registry_default_provider)
        merged_default_provider.update(default_provider_config or {})
        for item in payload.get("stores", []):
            if not isinstance(item, dict):
                continue
            spec = FederatedStoreSpec.from_payload(item, store_roots=store_roots, base_directory=base_directory)
            provider_payload = dict(merged_default_provider)
            provider_payload.update(item.get("provider", {}) if isinstance(item.get("provider"), dict) else {})
            store = FederatedStore(spec=spec)
            try:
                if spec.enabled and Path(spec.path).exists() and store.is_mounted:
                    provider, provider_name, provider_runtime_config = build_provider_from_config(provider_payload)
                    store.service = SemanticMemoryService(
                        semantic_store=load_semantic_memory_store(spec.path),
                        provider=provider,
                        store_path=spec.path,
                        provider_name=provider_name,
                        provider_config=provider_runtime_config,
                        review_policy=review,
                    )
            except Exception as exc:
                store.load_error = str(exc)
            stores[spec.store_id] = store
        default_active = list(default_active_store_ids or payload.get("default_active_store_ids", []))
        broker = cls(
            stores=stores,
            active_store_ids=[],
            registry_path=str(resolved_config_path),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
        )
        broker.set_active_store_ids(default_active or [store_id for store_id, store in stores.items() if store.is_available])
        return broker

    @classmethod
    def from_registry(
        cls,
        registry_path: str | Path,
        *,
        default_provider_config: dict[str, Any] | None = None,
        review_policy: ProposalReviewPolicy | None = None,
        default_active_store_ids: list[str] | None = None,
    ) -> FederatedSemanticMemoryBroker:
        return cls.from_config(
            registry_path,
            default_provider_config=default_provider_config,
            review_policy=review_policy,
            default_active_store_ids=default_active_store_ids,
        )

    def warmup(self, sample_text: str = "semantic memory warmup") -> None:
        for store in self._active_stores():
            provider = store.service.provider if store.service is not None else None
            warmup = getattr(provider, "warmup", None)
            if callable(warmup):
                warmup(sample_text=sample_text)
            elif store.service is not None:
                store.service._encode_text(sample_text)

    def summary_payload(self) -> dict[str, Any]:
        available_ids = [store_id for store_id, store in self.stores.items() if store.is_available]
        writable_ids = [store_id for store_id, store in self.stores.items() if store.is_available and store.is_writable]
        return {
            "enabled": True,
            "mode": "federated",
            "registry_path": self.registry_path,
            "store_count": len(self.stores),
            "available_store_count": len(available_ids),
            "active_store_ids": list(self.active_store_ids),
            "available_store_ids": available_ids,
            "writable_store_ids": writable_ids,
            "metadata": dict(self.metadata),
        }

    def status_payload(self) -> dict[str, Any]:
        return {
            "object": "semantic-memory-status",
            **self.summary_payload(),
            "stores": [store.summary_payload(active=store_id in self.active_store_ids) for store_id, store in sorted(self.stores.items())],
        }

    def list_stores_payload(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [store.summary_payload(active=store_id in self.active_store_ids) for store_id, store in sorted(self.stores.items())],
            "service": self.summary_payload(),
        }

    def set_active_store_ids(self, store_ids: list[str]) -> None:
        if not store_ids:
            self.active_store_ids = [store_id for store_id, store in self.stores.items() if store.is_available]
            return
        normalized: list[str] = []
        missing: list[str] = []
        unavailable: list[str] = []
        for raw_store_id in store_ids:
            store_id = str(raw_store_id)
            store = self.stores.get(store_id)
            if store is None:
                missing.append(store_id)
                continue
            if not store.is_available:
                unavailable.append(store_id)
                continue
            normalized.append(store_id)
        if missing:
            raise KeyError(f"Unknown store ids: {', '.join(sorted(missing))}")
        if unavailable:
            raise ValueError(f"Unavailable store ids: {', '.join(sorted(unavailable))}")
        self.active_store_ids = normalized

    def set_active_stores_payload(self, *, store_ids: list[str]) -> dict[str, Any]:
        self.set_active_store_ids(store_ids)
        return {
            "object": "semantic-memory-store-selection",
            "active_store_ids": list(self.active_store_ids),
            "service": self.summary_payload(),
        }

    def list_proposals_payload(self, *, status: str | None = None, store_id: str | None = None) -> dict[str, Any]:
        if store_id is not None:
            target_store = self._resolve_store(store_id)
            if target_store.service is None:
                raise ValueError(f"Store is unavailable: {store_id}")
            payload = target_store.service.list_proposals_payload(status=status)
            payload["store"] = {"store_id": target_store.store_id, "label": target_store.spec.label}
            payload["service"] = self.summary_payload()
            return payload
        proposals: list[dict[str, Any]] = []
        for candidate_store in self._stores_with_services():
            store_payload = candidate_store.service.list_proposals_payload(status=status)
            for row in store_payload.get("data", []):
                wrapped = dict(row)
                wrapped["store"] = {"store_id": candidate_store.store_id, "label": candidate_store.spec.label}
                proposals.append(wrapped)
        proposals.sort(key=lambda item: (str(item.get("created_at", "")), str(item.get("proposal_id", ""))))
        return {
            "object": "list",
            "status_filter": status,
            "data": proposals,
            "service": self.summary_payload(),
        }

    def list_proposal_events_payload(
        self,
        *,
        proposal_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
        store_id: str | None = None,
    ) -> dict[str, Any]:
        if store_id is not None:
            target_store = self._resolve_store(store_id)
            if target_store.service is None:
                raise ValueError(f"Store is unavailable: {store_id}")
            payload = target_store.service.list_proposal_events_payload(
                proposal_id=proposal_id,
                event_type=event_type,
                limit=limit,
            )
            payload["store"] = {"store_id": target_store.store_id, "label": target_store.spec.label}
            payload["service"] = self.summary_payload()
            return payload
        events: list[dict[str, Any]] = []
        for candidate_store in self._stores_with_services():
            store_payload = candidate_store.service.list_proposal_events_payload(
                proposal_id=proposal_id,
                event_type=event_type,
                limit=limit,
            )
            for row in store_payload.get("data", []):
                wrapped = dict(row)
                wrapped["store"] = {"store_id": candidate_store.store_id, "label": candidate_store.spec.label}
                events.append(wrapped)
        events.sort(key=lambda item: (str(item.get("timestamp", "")), str(item.get("event_id", ""))))
        return {
            "object": "list",
            "proposal_id_filter": proposal_id,
            "event_type_filter": event_type,
            "data": events[-max(1, min(int(limit), 1000)):],
            "service": self.summary_payload(),
        }

    def propose_change(self, *, store_id: str | None = None, **kwargs: Any) -> dict[str, Any]:
        target_store = self._resolve_mutation_store(store_id)
        payload = target_store.service.propose_change(**kwargs)
        payload["store"] = {"store_id": target_store.store_id, "label": target_store.spec.label}
        payload["service"] = self.summary_payload()
        return payload

    def review_proposal(self, *, store_id: str | None = None, **kwargs: Any) -> dict[str, Any]:
        target_store = self._resolve_mutation_store(store_id)
        payload = target_store.service.review_proposal(**kwargs)
        payload["store"] = {"store_id": target_store.store_id, "label": target_store.spec.label}
        payload["service"] = self.summary_payload()
        return payload

    def commit_proposal(self, *, store_id: str | None = None, **kwargs: Any) -> dict[str, Any]:
        target_store = self._resolve_mutation_store(store_id)
        payload = target_store.service.commit_proposal(**kwargs)
        payload["store"] = {"store_id": target_store.store_id, "label": target_store.spec.label}
        payload["service"] = self.summary_payload()
        return payload

    def get_memory_payload(self, memory_id: str, revision: int | None = None, store_id: str | None = None) -> dict[str, Any]:
        candidate_stores = self._resolve_store_ids([store_id] if store_id else None)
        matches: list[dict[str, Any]] = []
        for candidate_store_id in candidate_stores:
            store = self.stores[candidate_store_id]
            if store.service is None:
                continue
            try:
                payload = store.service.get_memory_payload(memory_id, revision=revision)
            except KeyError:
                continue
            matches.append({
                "store_id": candidate_store_id,
                "label": store.spec.label,
                "data": payload["data"],
            })
        if not matches:
            raise KeyError(f"Unknown semantic memory id: {memory_id}")
        if len(matches) > 1 and store_id is None:
            return {
                "object": "list",
                "memory_id": memory_id,
                "data": matches,
                "service": self.summary_payload(),
            }
        match = matches[0]
        return {
            "object": "semantic-memory",
            "store": {"store_id": match["store_id"], "label": match["label"]},
            "data": match["data"],
            "service": self.summary_payload(),
        }

    def query_payload(
        self,
        *,
        question: str,
        top_k: int = 5,
        include_inactive: bool = False,
        store_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        hits = self._federated_hits(question=question, top_k=top_k, include_inactive=include_inactive, store_ids=store_ids)
        return {
            "object": "list",
            "question": question,
            "top_k": top_k,
            "include_inactive": include_inactive,
            "data": [hit.to_payload() for hit in hits],
            "service": self.summary_payload(),
        }

    def answer_payload(
        self,
        *,
        question: str,
        top_k: int = 5,
        support_limit: int = 3,
        include_inactive: bool = False,
        store_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        hits = self._federated_hits(question=question, top_k=max(top_k, support_limit + 1), include_inactive=include_inactive, store_ids=store_ids)
        if not hits:
            raise ValueError("No active semantic-memory stores returned results for the question")
        primary = hits[0]
        conflicting = [hit.memory.to_payload() for hit in hits[1:] if self._is_conflicting(primary.memory, hit.memory)]
        supporting = [hit.to_payload() for hit in hits[1:] if not self._is_conflicting(primary.memory, hit.memory)][:support_limit]
        answer_text = primary.memory.summary or primary.memory.canonical_text
        return {
            "object": "semantic-answer",
            "question": question,
            "answer_text": answer_text,
            "primary_memory": primary.memory.to_payload(),
            "primary_store": {"store_id": primary.store_id, "label": primary.store_label},
            "supporting_memories": supporting,
            "conflicting_memories": conflicting,
            "relation_links": self._relation_links(primary.memory, hits),
            "service": self.summary_payload(),
        }

    def reasoning_payload(
        self,
        *,
        question: str,
        top_k: int = 8,
        support_limit: int = 3,
        neutral_limit: int = 3,
        conflict_limit: int = 3,
        include_inactive: bool = False,
        preferred_primary_memory_id: str | None = None,
        store_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        hits = self._federated_hits(question=question, top_k=top_k, include_inactive=include_inactive, store_ids=store_ids)
        if not hits:
            raise ValueError("No active semantic-memory stores returned results for the question")
        primary = hits[0]
        if preferred_primary_memory_id is not None:
            for candidate in hits:
                if candidate.memory.memory_id == preferred_primary_memory_id:
                    primary = candidate
                    break
        supporting: list[dict[str, Any]] = []
        neutral: list[dict[str, Any]] = []
        conflicting: list[dict[str, Any]] = []
        for hit in hits:
            if hit.memory.memory_id == primary.memory.memory_id and hit.store_id == primary.store_id:
                continue
            evidence = {
                "memory": hit.memory.to_payload(),
                "score": hit.merged_score,
                "route": "neutral",
                "reasons": ["high semantic similarity across federated stores"],
                "origin": "federated-retrieval",
                "relation_types": self._relation_types_between(primary.memory, hit.memory),
                "store": {"store_id": hit.store_id, "label": hit.store_label},
            }
            if self._is_conflicting(primary.memory, hit.memory):
                evidence["route"] = "conflict"
                evidence["reasons"] = ["explicit contradiction relation"]
                if len(conflicting) < conflict_limit:
                    conflicting.append(evidence)
                continue
            if self._is_supporting(primary.memory, hit.memory) or len(supporting) < support_limit:
                evidence["route"] = "support"
                if self._is_supporting(primary.memory, hit.memory):
                    evidence["reasons"] = ["explicit support relation"]
                if len(supporting) < support_limit:
                    supporting.append(evidence)
                    continue
            if len(neutral) < neutral_limit:
                neutral.append(evidence)
        return {
            "object": "semantic-reasoning-packet",
            "data": {
                "question": question,
                "primary_memory": primary.memory.to_payload(),
                "primary_score": primary.merged_score,
                "primary_store": {"store_id": primary.store_id, "label": primary.store_label},
                "supporting_evidence": supporting,
                "neutral_evidence": neutral,
                "conflicting_evidence": conflicting,
                "relation_links": self._relation_links(primary.memory, hits),
                "routing_summary": {
                    "mode": "federated",
                    "active_store_ids": self._resolve_store_ids(store_ids),
                    "candidate_count": len(hits),
                    "supporting_count": len(supporting),
                    "neutral_count": len(neutral),
                    "conflicting_count": len(conflicting),
                },
            },
            "service": self.summary_payload(),
        }

    def _active_stores(self) -> list[FederatedStore]:
        return [self.stores[store_id] for store_id in self.active_store_ids if self.stores[store_id].is_available]

    def _stores_with_services(self) -> list[FederatedStore]:
        return [store for _, store in sorted(self.stores.items()) if store.service is not None]

    def _resolve_store(self, store_id: str) -> FederatedStore:
        store = self.stores.get(str(store_id))
        if store is None:
            raise KeyError(f"Unknown store id: {store_id}")
        return store

    def _resolve_mutation_store(self, store_id: str | None) -> FederatedStore:
        if store_id is not None:
            store = self._resolve_store(store_id)
            if store.service is None or not store.is_available:
                raise ValueError(f"Store is unavailable: {store.store_id}")
            if not store.is_writable:
                raise PermissionError(f"Store is read-only: {store.store_id}")
            return store
        writable_active_stores = [store for store in self._active_stores() if store.service is not None and store.is_writable]
        if not writable_active_stores:
            raise PermissionError("No active read-write store is available; specify a writable store_id or change the config")
        if len(writable_active_stores) > 1:
            raise ValueError("Multiple active read-write stores are available; specify store_id for mutation tools")
        return writable_active_stores[0]

    def _resolve_store_ids(self, store_ids: list[str] | None) -> list[str]:
        if not store_ids:
            return list(self.active_store_ids)
        resolved: list[str] = []
        for raw_store_id in store_ids:
            store_id = str(raw_store_id)
            store = self.stores.get(store_id)
            if store is None:
                raise KeyError(f"Unknown store id: {store_id}")
            if not store.is_available:
                raise ValueError(f"Store is unavailable: {store_id}")
            resolved.append(store_id)
        return resolved

    def _federated_hits(
        self,
        *,
        question: str,
        top_k: int,
        include_inactive: bool,
        store_ids: list[str] | None,
    ) -> list[FederatedQueryHit]:
        per_store_hits: list[FederatedQueryHit] = []
        resolved_store_ids = self._resolve_store_ids(store_ids)
        for store_id in resolved_store_ids:
            store = self.stores[store_id]
            if store.service is None:
                continue
            results = store.service.semantic_store.query_text(
                store.service.provider,
                question,
                top_k=max(top_k, min(8, top_k * 2)),
                include_inactive=include_inactive,
            )
            if not results:
                continue
            scores = [item.score for item in results]
            top_score = max(scores)
            bottom_score = min(scores)
            score_range = top_score - bottom_score
            for index, item in enumerate(results, start=1):
                if score_range > 1e-9:
                    normalized_score = (item.score - bottom_score) / score_range
                elif top_score > 0:
                    normalized_score = 1.0
                else:
                    normalized_score = 0.0
                rank_bonus = 1.0 - ((index - 1) / max(len(results) - 1, 1))
                trust_weight = max(0.0, min(store.spec.trust_weight, 2.0)) / 2.0
                merged_score = (normalized_score * 0.65) + (rank_bonus * 0.20) + (trust_weight * 0.15)
                per_store_hits.append(
                    FederatedQueryHit(
                        store_id=store_id,
                        store_label=store.spec.label,
                        memory=item.memory,
                        raw_score=item.score,
                        merged_score=merged_score,
                        rank=index,
                        trust_weight=store.spec.trust_weight,
                        domain_tags=store.spec.domain_tags,
                    )
                )
        deduped: dict[str, FederatedQueryHit] = {}
        duplicate_sources: dict[str, set[str]] = {}
        for hit in sorted(per_store_hits, key=lambda item: (item.merged_score, item.raw_score), reverse=True):
            key = self._dedupe_key(hit.memory)
            duplicate_sources.setdefault(key, set()).add(hit.store_id)
            current = deduped.get(key)
            if current is None or (hit.merged_score, hit.raw_score) > (current.merged_score, current.raw_score):
                deduped[key] = hit
        merged: list[FederatedQueryHit] = []
        for key, hit in deduped.items():
            duplicate_store_ids = tuple(sorted(duplicate_sources.get(key, {hit.store_id})))
            merged.append(
                FederatedQueryHit(
                    store_id=hit.store_id,
                    store_label=hit.store_label,
                    memory=hit.memory,
                    raw_score=hit.raw_score,
                    merged_score=hit.merged_score,
                    rank=hit.rank,
                    trust_weight=hit.trust_weight,
                    domain_tags=hit.domain_tags,
                    duplicate_store_ids=duplicate_store_ids,
                )
            )
        merged.sort(key=lambda item: (item.merged_score, item.raw_score), reverse=True)
        return merged[:top_k]

    def _dedupe_key(self, memory: Any) -> str:
        signature = memory.metadata.get("claim_signature") if isinstance(getattr(memory, "metadata", {}), dict) else None
        if signature:
            return str(signature)
        return _normalize_claim_signature(memory.canonical_text)

    def _relation_types_between(self, left: Any, right: Any) -> list[str]:
        relation_types: list[str] = []
        relation_types.extend(relation.relation_type for relation in left.relations if relation.target_memory_id == right.memory_id)
        relation_types.extend(relation.relation_type for relation in right.relations if relation.target_memory_id == left.memory_id)
        return sorted(set(relation_types))

    def _is_conflicting(self, left: Any, right: Any) -> bool:
        return any(item in _CONFLICT_RELATION_TYPES for item in self._relation_types_between(left, right))

    def _is_supporting(self, left: Any, right: Any) -> bool:
        relation_types = self._relation_types_between(left, right)
        if any(item in _SUPPORT_RELATION_TYPES for item in relation_types):
            return True
        left_signature = self._dedupe_key(left)
        right_signature = self._dedupe_key(right)
        return left_signature == right_signature

    def _relation_links(self, primary_memory: Any, hits: list[FederatedQueryHit]) -> list[dict[str, Any]]:
        links: list[dict[str, Any]] = []
        for hit in hits:
            relation_types = self._relation_types_between(primary_memory, hit.memory)
            if not relation_types:
                continue
            links.append(
                {
                    "memory_id": hit.memory.memory_id,
                    "store_id": hit.store_id,
                    "relation_types": relation_types,
                }
            )
        return links