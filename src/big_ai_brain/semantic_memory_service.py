from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from typing import Any

from rwif_semantic_memory import ProvenanceRef, SemanticMemoryObject, SemanticMemoryStore, SemanticRelation, save_semantic_memory_store


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _local_path_or_none(path: str | None) -> Path | None:
    if path is None:
        return None
    if "://" in path:
        return None
    return Path(path)


@dataclass(frozen=True)
class ProposalReviewPolicy:
    min_provenance_count: int = 0
    min_provenance_confidence: float | None = None
    allowed_source_types: tuple[str, ...] = ()
    require_locator: bool = False
    require_quoted_text: bool = False
    require_review_notes: bool = False
    operation_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def summary_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_provenance_count": self.min_provenance_count,
            "min_provenance_confidence": self.min_provenance_confidence,
            "allowed_source_types": list(self.allowed_source_types),
            "require_locator": self.require_locator,
            "require_quoted_text": self.require_quoted_text,
            "require_review_notes": self.require_review_notes,
            "operation_overrides": {
                operation: self.resolve_for_operation(operation).base_summary_payload()
                for operation in sorted(self.operation_overrides)
            },
            "effective_operation_policies": self.effective_operation_policies_payload(),
        }

    def base_summary_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.base_enabled,
            "min_provenance_count": self.min_provenance_count,
            "min_provenance_confidence": self.min_provenance_confidence,
            "allowed_source_types": list(self.allowed_source_types),
            "require_locator": self.require_locator,
            "require_quoted_text": self.require_quoted_text,
            "require_review_notes": self.require_review_notes,
        }

    @property
    def enabled(self) -> bool:
        return self.base_enabled or any(self.resolve_for_operation(operation).base_enabled for operation in self.operation_overrides)

    @property
    def base_enabled(self) -> bool:
        return any(
            (
                self.min_provenance_count > 0,
                self.min_provenance_confidence is not None,
                bool(self.allowed_source_types),
                self.require_locator,
                self.require_quoted_text,
                self.require_review_notes,
            )
        )

    def resolve_for_operation(self, operation: str | None) -> ProposalReviewPolicy:
        normalized_operation = str(operation or "").strip().lower()
        if not normalized_operation:
            return self._without_operation_overrides()
        override = self.operation_overrides.get(normalized_operation)
        if not isinstance(override, dict) or not override:
            return self._without_operation_overrides()
        return ProposalReviewPolicy(
            min_provenance_count=int(override.get("min_provenance_count", self.min_provenance_count)),
            min_provenance_confidence=(
                self.min_provenance_confidence
                if override.get("min_provenance_confidence") is None
                else float(override["min_provenance_confidence"])
            ),
            allowed_source_types=(
                self.allowed_source_types
                if override.get("allowed_source_types") is None
                else tuple(str(item) for item in override.get("allowed_source_types", ()) if str(item).strip())
            ),
            require_locator=bool(override.get("require_locator", self.require_locator)),
            require_quoted_text=bool(override.get("require_quoted_text", self.require_quoted_text)),
            require_review_notes=bool(override.get("require_review_notes", self.require_review_notes)),
        )

    def _without_operation_overrides(self) -> ProposalReviewPolicy:
        return ProposalReviewPolicy(
            min_provenance_count=self.min_provenance_count,
            min_provenance_confidence=self.min_provenance_confidence,
            allowed_source_types=self.allowed_source_types,
            require_locator=self.require_locator,
            require_quoted_text=self.require_quoted_text,
            require_review_notes=self.require_review_notes,
        )

    def effective_operation_policies_payload(self) -> dict[str, Any]:
        operations = ("create", "revise", "merge", "deprecate", "contradict")
        return {
            operation: self.resolve_for_operation(operation).base_summary_payload()
            for operation in operations
        }


@dataclass
class SemanticMemoryService:
    semantic_store: SemanticMemoryStore
    provider: Any
    store_path: str | None = None
    provider_name: str = "unknown"
    provider_config: dict[str, Any] | None = None
    review_policy: ProposalReviewPolicy = field(default_factory=ProposalReviewPolicy)
    proposal_journal_path: str | None = None
    proposal_event_journal_path: str | None = None
    proposals: dict[str, dict[str, Any]] = field(default_factory=dict)
    proposal_events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.proposal_journal_path is None:
            local_store_path = _local_path_or_none(self.store_path)
            if local_store_path is not None:
                self.proposal_journal_path = str(local_store_path.with_suffix(local_store_path.suffix + ".proposals.json"))
        if self.proposal_event_journal_path is None:
            local_store_path = _local_path_or_none(self.store_path)
            if local_store_path is not None:
                self.proposal_event_journal_path = str(local_store_path.with_suffix(local_store_path.suffix + ".proposal-events.jsonl"))
        self._load_proposals()
        self._load_proposal_events()

    def summary_payload(self) -> dict[str, Any]:
        proposal_counts: dict[str, int] = {}
        for proposal in self.proposals.values():
            status = str(proposal.get("status", "unknown"))
            proposal_counts[status] = proposal_counts.get(status, 0) + 1
        return {
            "enabled": True,
            "store_path": self.store_path,
            "provider": {
                "name": self.provider_name,
                "config": dict(self.provider_config or {}),
            },
            "revision_count": self.semantic_store.revision_count,
            "active_memory_count": len(self.semantic_store.active_memories),
            "vector_length": self.semantic_store.vector_length,
            "metadata": dict(self.semantic_store.metadata),
            "proposal_journal_path": self.proposal_journal_path,
            "proposal_event_journal_path": self.proposal_event_journal_path,
            "proposal_counts": proposal_counts,
            "proposal_event_count": len(self.proposal_events),
            "review_policy": self.review_policy.summary_payload(),
        }

    def status_payload(self) -> dict[str, Any]:
        return {
            "object": "semantic-memory-status",
            **self.summary_payload(),
        }

    def get_memory_payload(self, memory_id: str, revision: int | None = None) -> dict[str, Any]:
        memory = self.semantic_store.get_object(memory_id, revision=revision)
        return {
            "object": "semantic-memory",
            "data": memory.to_payload(),
        }

    def query_payload(self, *, question: str, top_k: int = 5, include_inactive: bool = False) -> dict[str, Any]:
        results = self.semantic_store.query_text(
            self.provider,
            question,
            top_k=top_k,
            include_inactive=include_inactive,
        )
        return {
            "object": "list",
            "question": question,
            "top_k": top_k,
            "include_inactive": include_inactive,
            "data": [
                {
                    "score": item.score,
                    "memory": item.memory.to_payload(),
                }
                for item in results
            ],
            "service": self.summary_payload(),
        }

    def answer_payload(
        self,
        *,
        question: str,
        top_k: int = 5,
        support_limit: int = 3,
        include_inactive: bool = False,
    ) -> dict[str, Any]:
        result = self.semantic_store.answer_question(
            self.provider,
            question,
            top_k=top_k,
            support_limit=support_limit,
            include_inactive=include_inactive,
        )
        return {
            "object": "semantic-answer",
            "question": result.question,
            "answer_text": result.answer_text,
            "primary_memory": result.primary_memory.to_payload(),
            "supporting_memories": [
                {
                    "score": item.score,
                    "memory": item.memory.to_payload(),
                }
                for item in result.supporting_memories
            ],
            "conflicting_memories": [item.to_payload() for item in result.conflicting_memories],
            "relation_links": list(result.relation_links),
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
    ) -> dict[str, Any]:
        packet = self.semantic_store.route_evidence(
            self.provider,
            question,
            top_k=top_k,
            support_limit=support_limit,
            neutral_limit=neutral_limit,
            conflict_limit=conflict_limit,
            include_inactive=include_inactive,
            preferred_primary_memory_id=preferred_primary_memory_id,
        )
        return {
            "object": "semantic-reasoning-packet",
            "data": packet.to_payload(),
            "service": self.summary_payload(),
        }

    def list_proposals_payload(self, *, status: str | None = None) -> dict[str, Any]:
        proposals = list(self.proposals.values())
        if status is not None:
            proposals = [proposal for proposal in proposals if proposal.get("status") == status]
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
    ) -> dict[str, Any]:
        filtered = list(self.proposal_events)
        if proposal_id is not None:
            filtered = [event for event in filtered if event.get("proposal_id") == proposal_id]
        if event_type is not None:
            normalized_event_type = str(event_type).strip().lower()
            filtered = [event for event in filtered if str(event.get("event_type") or "").lower() == normalized_event_type]
        filtered = filtered[-max(1, min(int(limit), 1000)):]
        return {
            "object": "list",
            "proposal_id_filter": proposal_id,
            "event_type_filter": event_type,
            "data": filtered,
            "service": self.summary_payload(),
        }

    def propose_change(
        self,
        *,
        operation: str,
        proposer: str,
        proposer_role: str | None = None,
        proposer_credential_source: str | None = None,
        memory_id: str | None = None,
        title: str | None = None,
        canonical_text: str | None = None,
        kind: str | None = None,
        summary: str | None = None,
        facts: list[str] | tuple[str, ...] | None = None,
        tags: list[str] | tuple[str, ...] | None = None,
        relations: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        provenance: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
        conflicting_memory_id: str | None = None,
        replacement_memory_id: str | None = None,
        reason: str | None = None,
        weight: float | None = None,
        source_model: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        normalized_operation = operation.strip().lower()
        if normalized_operation not in {"create", "revise", "merge", "deprecate", "contradict"}:
            raise ValueError(f"Unsupported proposal operation: {operation}")
        payload = {
            "memory_id": memory_id,
            "title": title,
            "canonical_text": canonical_text,
            "kind": kind,
            "summary": summary,
            "facts": list(facts or []),
            "tags": list(tags or []),
            "relations": list(relations or []),
            "provenance": list(provenance or []),
            "status": status,
            "metadata": dict(metadata or {}),
            "conflicting_memory_id": conflicting_memory_id,
            "replacement_memory_id": replacement_memory_id,
            "reason": reason,
            "weight": weight,
            "source_model": source_model,
        }
        self._validate_proposal_payload(normalized_operation, payload)
        proposal_id = f"proposal-{uuid.uuid4().hex[:12]}"
        timestamp = _utc_now()
        proposal = {
            "proposal_id": proposal_id,
            "status": "pending",
            "operation": normalized_operation,
            "proposer": proposer,
            "proposer_role": None if proposer_role is None else str(proposer_role),
            "proposer_credential_source": None if proposer_credential_source is None else str(proposer_credential_source),
            "notes": notes,
            "created_at": timestamp,
            "updated_at": timestamp,
            "payload": payload,
            "reviews": [],
            "commit": None,
        }
        self.proposals[proposal_id] = proposal
        self._save_proposals()
        self._append_proposal_event(
            proposal,
            event_type="proposed",
            timestamp=timestamp,
            actor=proposer,
            actor_role=proposer_role,
            credential_source=proposer_credential_source,
            details={
                "notes": notes,
                "payload": dict(payload),
            },
        )
        return {
            "object": "semantic-memory-proposal",
            "data": proposal,
            "service": self.summary_payload(),
        }

    def review_proposal(
        self,
        *,
        proposal_id: str,
        reviewer: str,
        reviewer_role: str | None = None,
        reviewer_credential_source: str | None = None,
        decision: str,
        notes: str | None = None,
    ) -> dict[str, Any]:
        proposal = self._require_proposal(proposal_id)
        normalized_decision = decision.strip().lower()
        if normalized_decision not in {"approve", "reject"}:
            raise ValueError(f"Unsupported review decision: {decision}")
        if proposal["status"] == "committed":
            raise ValueError("Committed proposals cannot be reviewed")
        policy_check = None
        if normalized_decision == "approve":
            policy_check = self._evaluate_policy_check(proposal, review_notes=notes, require_review_notes=True)
            if not policy_check["passed"]:
                reasons = "; ".join(policy_check["reasons"])
                raise ValueError(f"Proposal does not satisfy review policy: {reasons}")
        review = {
            "reviewer": reviewer,
            "reviewer_role": None if reviewer_role is None else str(reviewer_role),
            "reviewer_credential_source": None if reviewer_credential_source is None else str(reviewer_credential_source),
            "decision": normalized_decision,
            "notes": notes,
            "reviewed_at": _utc_now(),
            "policy_check": policy_check,
        }
        proposal.setdefault("reviews", []).append(review)
        proposal["status"] = "approved" if normalized_decision == "approve" else "rejected"
        proposal["updated_at"] = review["reviewed_at"]
        self._save_proposals()
        self._append_proposal_event(
            proposal,
            event_type="reviewed",
            timestamp=review["reviewed_at"],
            actor=reviewer,
            actor_role=reviewer_role,
            credential_source=reviewer_credential_source,
            details={
                "decision": normalized_decision,
                "notes": notes,
                "policy_check": None if policy_check is None else dict(policy_check),
            },
        )
        return {
            "object": "semantic-memory-proposal",
            "data": proposal,
            "service": self.summary_payload(),
        }

    def commit_proposal(
        self,
        *,
        proposal_id: str,
        actor: str,
        actor_role: str | None = None,
        actor_credential_source: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        proposal = self._require_proposal(proposal_id)
        if proposal["status"] != "approved":
            raise ValueError("Only approved proposals can be committed")
        policy_check = self._evaluate_policy_check(proposal, review_notes=None, require_review_notes=False)
        if not policy_check["passed"]:
            reasons = "; ".join(policy_check["reasons"])
            raise ValueError(f"Proposal no longer satisfies commit policy: {reasons}")
        commit_payload = self._apply_proposal(proposal)
        committed_at = _utc_now()
        proposal["status"] = "committed"
        proposal["updated_at"] = committed_at
        proposal["commit"] = {
            "actor": actor,
            "actor_role": None if actor_role is None else str(actor_role),
            "actor_credential_source": None if actor_credential_source is None else str(actor_credential_source),
            "notes": notes,
            "committed_at": committed_at,
            "policy_check": policy_check,
            "result": commit_payload,
        }
        self._save_proposals()
        self._append_proposal_event(
            proposal,
            event_type="committed",
            timestamp=committed_at,
            actor=actor,
            actor_role=actor_role,
            credential_source=actor_credential_source,
            details={
                "notes": notes,
                "policy_check": dict(policy_check),
                "result": dict(commit_payload),
            },
        )
        return {
            "object": "semantic-memory-proposal",
            "data": proposal,
            "service": self.summary_payload(),
        }

    def _require_proposal(self, proposal_id: str) -> dict[str, Any]:
        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"Unknown proposal: {proposal_id}")
        return proposal

    def _validate_proposal_payload(self, operation: str, payload: dict[str, Any]) -> None:
        if operation == "create":
            if not str(payload.get("memory_id") or "").strip():
                raise ValueError("create proposals require memory_id")
            if not str(payload.get("canonical_text") or "").strip():
                raise ValueError("create proposals require canonical_text")
            if not str(payload.get("title") or "").strip():
                raise ValueError("create proposals require title")
            try:
                self.semantic_store.get_object(str(payload["memory_id"]))
            except KeyError:
                return
            raise ValueError(f"Memory already exists: {payload['memory_id']}")
        if operation in {"revise", "merge", "deprecate", "contradict"} and not str(payload.get("memory_id") or "").strip():
            raise ValueError(f"{operation} proposals require memory_id")
        if operation == "contradict" and not str(payload.get("conflicting_memory_id") or "").strip():
            raise ValueError("contradict proposals require conflicting_memory_id")
        if operation == "deprecate" and not str(payload.get("reason") or "").strip():
            raise ValueError("deprecate proposals require reason")

    def _evaluate_policy_check(
        self,
        proposal: dict[str, Any],
        *,
        review_notes: str | None,
        require_review_notes: bool,
    ) -> dict[str, Any]:
        operation = str(proposal.get("operation") or "")
        policy = self.review_policy.resolve_for_operation(operation)
        provenance_rows = self._proposal_provenance_rows(proposal)
        qualified_rows: list[dict[str, Any]] = []
        reasons: list[str] = []
        allowed_source_types = {item for item in policy.allowed_source_types if item}

        for row in provenance_rows:
            source_type = str(row.get("source_type") or "document")
            confidence = row.get("confidence")
            locator = row.get("locator")
            quoted_text = row.get("quoted_text")
            if allowed_source_types and source_type not in allowed_source_types:
                continue
            if policy.min_provenance_confidence is not None:
                if confidence is None or float(confidence) < policy.min_provenance_confidence:
                    continue
            if policy.require_locator and not str(locator or "").strip():
                continue
            if policy.require_quoted_text and not str(quoted_text or "").strip():
                continue
            qualified_rows.append(dict(row))

        if policy.min_provenance_count > 0 and len(qualified_rows) < policy.min_provenance_count:
            reasons.append(
                f"requires at least {policy.min_provenance_count} qualifying provenance references but found {len(qualified_rows)}"
            )
        if allowed_source_types and not qualified_rows:
            reasons.append(
                "requires provenance from allowed source types: " + ", ".join(sorted(allowed_source_types))
            )
        if policy.require_review_notes and require_review_notes and not str(review_notes or "").strip():
            reasons.append("approval review notes are required by policy")

        return {
            "passed": not reasons,
            "operation": operation,
            "policy": policy.summary_payload(),
            "provenance_count": len(provenance_rows),
            "qualified_provenance_count": len(qualified_rows),
            "qualified_provenance": qualified_rows,
            "reasons": reasons,
        }

    def _proposal_provenance_rows(self, proposal: dict[str, Any]) -> list[dict[str, Any]]:
        payload = dict(proposal.get("payload") or {})
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str | None, str | None]] = set()

        def add_row(row: dict[str, Any]) -> None:
            source_id = str(row.get("source_id") or "").strip()
            if not source_id:
                return
            source_type = str(row.get("source_type") or "document")
            locator = None if row.get("locator") is None else str(row.get("locator"))
            quoted_text = None if row.get("quoted_text") is None else str(row.get("quoted_text"))
            key = (source_id, source_type, locator, quoted_text)
            if key in seen:
                return
            seen.add(key)
            normalized_row: dict[str, Any] = {
                "source_id": source_id,
                "source_type": source_type,
            }
            if locator is not None:
                normalized_row["locator"] = locator
            if quoted_text is not None:
                normalized_row["quoted_text"] = quoted_text
            if row.get("confidence") is not None:
                normalized_row["confidence"] = float(row["confidence"])
            metadata = row.get("metadata")
            if isinstance(metadata, dict) and metadata:
                normalized_row["metadata"] = dict(metadata)
            rows.append(normalized_row)

        for memory_key in ("memory_id", "conflicting_memory_id"):
            memory_id = payload.get(memory_key)
            if memory_id is None:
                continue
            try:
                memory = self.semantic_store.get_object(str(memory_id))
            except KeyError:
                continue
            for reference in memory.provenance:
                add_row(reference.to_payload())

        for item in payload.get("provenance", []):
            if isinstance(item, dict):
                add_row(item)

        return rows

    def _apply_proposal(self, proposal: dict[str, Any]) -> dict[str, Any]:
        operation = str(proposal["operation"])
        payload = dict(proposal["payload"])
        source_model = payload.get("source_model")

        if operation == "create":
            memory = SemanticMemoryObject(
                memory_id=str(payload["memory_id"]),
                revision=1,
                title=str(payload["title"]),
                canonical_text=str(payload["canonical_text"]),
                kind=str(payload.get("kind") or "concept"),
                summary=None if payload.get("summary") is None else str(payload["summary"]),
                facts=tuple(str(item) for item in payload.get("facts", [])),
                tags=tuple(str(item) for item in payload.get("tags", [])),
                relations=tuple(SemanticRelation.from_payload(item) for item in payload.get("relations", []) if isinstance(item, dict)),
                provenance=tuple(ProvenanceRef.from_payload(item) for item in payload.get("provenance", []) if isinstance(item, dict)),
                status=str(payload.get("status") or "active"),
                source_model=None if source_model is None else str(source_model),
                metadata=dict(payload.get("metadata") or {}),
            )
            self.semantic_store.add_object_activation(memory, activation=self._encode_text(memory.canonical_text))
            self._persist_store()
            return {"operation": operation, "memory_id": memory.memory_id, "revision": memory.revision}

        if operation == "revise":
            revised = self.semantic_store.revise_object(
                str(payload["memory_id"]),
                activation=None if payload.get("canonical_text") is None else self._encode_text(str(payload["canonical_text"])),
                title=None if payload.get("title") is None else str(payload["title"]),
                canonical_text=None if payload.get("canonical_text") is None else str(payload["canonical_text"]),
                kind=None if payload.get("kind") is None else str(payload["kind"]),
                summary=None if payload.get("summary") is None else str(payload["summary"]),
                facts=None if not payload.get("facts") else tuple(str(item) for item in payload.get("facts", [])),
                tags=None if not payload.get("tags") else tuple(str(item) for item in payload.get("tags", [])),
                relations=None if not payload.get("relations") else tuple(SemanticRelation.from_payload(item) for item in payload.get("relations", []) if isinstance(item, dict)),
                provenance=None if not payload.get("provenance") else tuple(ProvenanceRef.from_payload(item) for item in payload.get("provenance", []) if isinstance(item, dict)),
                status=None if payload.get("status") is None else str(payload["status"]),
                source_model=None if source_model is None else str(source_model),
                metadata=None if payload.get("metadata") is None else dict(payload.get("metadata") or {}),
            )
            self._persist_store()
            return {"operation": operation, "memory_id": revised.memory_id, "revision": revised.revision}

        if operation == "merge":
            merged = self.semantic_store.merge_update(
                str(payload["memory_id"]),
                activation=None if payload.get("canonical_text") is None else self._encode_text(str(payload["canonical_text"])),
                title=None if payload.get("title") is None else str(payload["title"]),
                canonical_text=None if payload.get("canonical_text") is None else str(payload["canonical_text"]),
                kind=None if payload.get("kind") is None else str(payload["kind"]),
                summary=None if payload.get("summary") is None else str(payload["summary"]),
                facts=tuple(str(item) for item in payload.get("facts", [])),
                tags=tuple(str(item) for item in payload.get("tags", [])),
                relations=tuple(SemanticRelation.from_payload(item) for item in payload.get("relations", []) if isinstance(item, dict)),
                provenance=tuple(ProvenanceRef.from_payload(item) for item in payload.get("provenance", []) if isinstance(item, dict)),
                source_model=None if source_model is None else str(source_model),
                metadata_patch=None if payload.get("metadata") is None else dict(payload.get("metadata") or {}),
            )
            self._persist_store()
            return {"operation": operation, "memory_id": merged.memory_id, "revision": merged.revision}

        if operation == "deprecate":
            deprecated = self.semantic_store.deprecate_object(
                str(payload["memory_id"]),
                reason=str(payload["reason"]),
                replacement_memory_id=None if payload.get("replacement_memory_id") is None else str(payload["replacement_memory_id"]),
                source_model=None if source_model is None else str(source_model),
            )
            self._persist_store()
            return {"operation": operation, "memory_id": deprecated.memory_id, "revision": deprecated.revision, "status": deprecated.status}

        if operation == "contradict":
            left, right = self.semantic_store.register_contradiction(
                str(payload["memory_id"]),
                str(payload["conflicting_memory_id"]),
                reason=None if payload.get("reason") is None else str(payload["reason"]),
                weight=float(payload.get("weight") or 1.0),
                source_model=None if source_model is None else str(source_model),
            )
            self._persist_store()
            return {
                "operation": operation,
                "memory_id": left.memory_id,
                "revision": left.revision,
                "conflicting_memory_id": right.memory_id,
                "conflicting_revision": right.revision,
            }

        raise ValueError(f"Unsupported proposal operation: {operation}")

    def _persist_store(self) -> None:
        local_store_path = _local_path_or_none(self.store_path)
        if local_store_path is None:
            return
        local_store_path.parent.mkdir(parents=True, exist_ok=True)
        save_semantic_memory_store(local_store_path, self.semantic_store)

    def _encode_text(self, text: str):
        try:
            return self.provider.encode_text(text)
        except Exception as exc:
            raise ValueError(f"Provider could not encode semantic text: {text}") from exc

    def _load_proposals(self) -> None:
        journal_path = _local_path_or_none(self.proposal_journal_path)
        if journal_path is None or not journal_path.exists():
            return
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            self.proposals = {
                str(item["proposal_id"]): dict(item)
                for item in payload
                if isinstance(item, dict) and item.get("proposal_id")
            }

    def _save_proposals(self) -> None:
        journal_path = _local_path_or_none(self.proposal_journal_path)
        if journal_path is None:
            return
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        proposals = sorted(self.proposals.values(), key=lambda item: str(item.get("created_at", "")))
        journal_path.write_text(json.dumps(proposals, indent=2) + "\n", encoding="utf-8")

    def _load_proposal_events(self) -> None:
        journal_path = _local_path_or_none(self.proposal_event_journal_path)
        if journal_path is None or not journal_path.exists():
            return
        events: list[dict[str, Any]] = []
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict) and payload.get("event_id"):
                events.append(dict(payload))
        self.proposal_events = events

    def _append_proposal_event(
        self,
        proposal: dict[str, Any],
        *,
        event_type: str,
        timestamp: str,
        actor: str,
        actor_role: str | None,
        credential_source: str | None,
        details: dict[str, Any],
    ) -> None:
        event = {
            "event_id": f"proposal-event-{uuid.uuid4().hex[:12]}",
            "event_type": str(event_type),
            "timestamp": str(timestamp),
            "proposal_id": str(proposal["proposal_id"]),
            "proposal_status": str(proposal.get("status") or "unknown"),
            "operation": str(proposal.get("operation") or ""),
            "memory_id": (proposal.get("payload") or {}).get("memory_id"),
            "actor": str(actor),
            "actor_role": None if actor_role is None else str(actor_role),
            "credential_source": None if credential_source is None else str(credential_source),
            "details": dict(details),
        }
        conflicting_memory_id = (proposal.get("payload") or {}).get("conflicting_memory_id")
        if conflicting_memory_id is not None:
            event["conflicting_memory_id"] = str(conflicting_memory_id)
        self.proposal_events.append(event)

        journal_path = _local_path_or_none(self.proposal_event_journal_path)
        if journal_path is None:
            return
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        with journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")