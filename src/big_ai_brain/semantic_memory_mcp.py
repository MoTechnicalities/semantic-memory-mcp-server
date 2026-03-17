from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from .federated_memory import FederatedSemanticMemoryBroker
from .semantic_memory_service import ProposalReviewPolicy, SemanticMemoryService


_TOOLS = (
    {
        "name": "memory_list_stores",
        "description": "List configured semantic memory stores and their availability state.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "memory_set_active_stores",
        "description": "Replace the active semantic memory store set used for federated query, answer, and reason operations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "store_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["store_ids"],
            "additionalProperties": False,
        },
    },
    {
        "name": "memory_status",
        "description": "Return semantic memory service status and store metadata.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "memory_get",
        "description": "Return a semantic memory object by id and optional revision.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "revision": {"type": "integer"},
                "store_id": {"type": "string"},
            },
            "required": ["memory_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "memory_query",
        "description": "Query the semantic memory store and return ranked matches.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1},
                "include_inactive": {"type": "boolean"},
                "store_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
    {
        "name": "memory_answer",
        "description": "Answer a question from semantic memory with supporting and conflicting memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1},
                "support_limit": {"type": "integer", "minimum": 1},
                "include_inactive": {"type": "boolean"},
                "store_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
    {
        "name": "memory_reason",
        "description": "Return the full semantic reasoning packet for a question.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1},
                "support_limit": {"type": "integer", "minimum": 0},
                "neutral_limit": {"type": "integer", "minimum": 0},
                "conflict_limit": {"type": "integer", "minimum": 0},
                "include_inactive": {"type": "boolean"},
                "preferred_primary_memory_id": {"type": "string"},
                "store_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
    {
        "name": "memory_list_proposals",
        "description": "List staged semantic memory proposals.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "store_id": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "memory_list_proposal_events",
        "description": "List append-only semantic memory proposal audit events.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "proposal_id": {"type": "string"},
                "event_type": {"type": "string", "enum": ["proposed", "reviewed", "committed"]},
                "limit": {"type": "integer", "minimum": 1},
                "store_id": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "memory_propose",
        "description": "Stage a semantic memory create or update proposal for later review and commit.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["create", "revise", "merge", "deprecate", "contradict"]},
                "proposer": {"type": "string"},
                "memory_id": {"type": "string"},
                "title": {"type": "string"},
                "canonical_text": {"type": "string"},
                "kind": {"type": "string"},
                "summary": {"type": "string"},
                "facts": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}},
                "relations": {"type": "array", "items": {"type": "object"}},
                "provenance": {"type": "array", "items": {"type": "object"}},
                "status": {"type": "string"},
                "metadata": {"type": "object"},
                "conflicting_memory_id": {"type": "string"},
                "replacement_memory_id": {"type": "string"},
                "reason": {"type": "string"},
                "weight": {"type": "number"},
                "source_model": {"type": "string"},
                "notes": {"type": "string"},
                "store_id": {"type": "string"},
            },
            "required": ["operation", "proposer"],
            "additionalProperties": False,
        },
    },
    {
        "name": "memory_review_proposal",
        "description": "Approve or reject a staged semantic memory proposal.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "proposal_id": {"type": "string"},
                "reviewer": {"type": "string"},
                "decision": {"type": "string", "enum": ["approve", "reject"]},
                "notes": {"type": "string"},
                "store_id": {"type": "string"},
            },
            "required": ["proposal_id", "reviewer", "decision"],
            "additionalProperties": False,
        },
    },
    {
        "name": "memory_commit_proposal",
        "description": "Commit an approved semantic memory proposal into the RWIF semantic store.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "proposal_id": {"type": "string"},
                "actor": {"type": "string"},
                "notes": {"type": "string"},
                "store_id": {"type": "string"},
            },
            "required": ["proposal_id", "actor"],
            "additionalProperties": False,
        },
    },
)


@dataclass(frozen=True)
class MemoryMutationToolPrincipal:
    role: str
    credential_source: str


@dataclass(frozen=True)
class MemoryMutationToolAuth:
    token_roles: dict[str, frozenset[str]] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return bool(self.token_roles)

    def authorize(self, *, tool_name: str, auth_token: str | None) -> MemoryMutationToolPrincipal | None:
        if not self.enabled:
            return None
        required_roles = self.required_roles_for_tool(tool_name)
        if not required_roles:
            return None
        token = str(auth_token or "").strip()
        if not token:
            raise PermissionError(f"{tool_name} requires auth_token when MCP write-tool auth is enabled")
        roles = self.token_roles.get(token)
        if roles is None:
            raise PermissionError(f"{tool_name} received an invalid auth_token")
        matched_roles = roles.intersection(required_roles)
        if not matched_roles:
            raise PermissionError(f"{tool_name} requires one of these roles: {', '.join(sorted(required_roles))}")
        return MemoryMutationToolPrincipal(
            role=self._preferred_role(matched_roles),
            credential_source="mcp-auth-token",
        )

    def required_roles_for_tool(self, tool_name: str) -> set[str]:
        if tool_name == "memory_propose":
            return {"proposal", "review", "commit", "admin"}
        if tool_name == "memory_review_proposal":
            return {"review", "admin"}
        if tool_name == "memory_commit_proposal":
            return {"commit", "admin"}
        return set()

    def summary_payload(self) -> dict[str, Any]:
        tool_roles = {
            tool_name: sorted(self.required_roles_for_tool(tool_name))
            for tool_name in ("memory_propose", "memory_review_proposal", "memory_commit_proposal")
        }
        configured_roles: dict[str, int] = {}
        for roles in self.token_roles.values():
            for role in roles:
                configured_roles[role] = configured_roles.get(role, 0) + 1
        return {
            "enabled": self.enabled,
            "configured_roles": configured_roles,
            "tool_roles": tool_roles,
        }

    def _preferred_role(self, roles: set[str] | frozenset[str]) -> str:
        for role in ("admin", "commit", "review", "proposal"):
            if role in roles:
                return role
        return sorted(roles)[0]


def build_memory_mutation_tool_auth(
    *,
    proposal_api_keys: list[str] | tuple[str, ...] | None,
    review_api_keys: list[str] | tuple[str, ...] | None,
    commit_api_keys: list[str] | tuple[str, ...] | None,
    admin_api_keys: list[str] | tuple[str, ...] | None,
) -> MemoryMutationToolAuth:
    token_roles: dict[str, set[str]] = {}

    def register(keys: list[str] | tuple[str, ...] | None, role: str) -> None:
        for key in keys or ():
            token = str(key).strip()
            if not token:
                continue
            token_roles.setdefault(token, set()).add(role)

    register(proposal_api_keys, "proposal")
    register(review_api_keys, "review")
    register(commit_api_keys, "commit")
    register(admin_api_keys, "admin")
    return MemoryMutationToolAuth(token_roles={token: frozenset(roles) for token, roles in token_roles.items()})


def _tool_metadata(
    *,
    tool_name: str,
    tool_auth: MemoryMutationToolAuth | None,
    review_policy: ProposalReviewPolicy,
) -> dict[str, Any]:
    metadata = {
        "auth": {
            "enabled": False if tool_auth is None else tool_auth.enabled,
            "required_roles": sorted(tool_auth.required_roles_for_tool(tool_name)) if tool_auth is not None else [],
        }
    }
    if tool_name in {"memory_propose", "memory_review_proposal", "memory_commit_proposal"}:
        metadata["proposal_review_policy"] = {
            "base": review_policy.base_summary_payload(),
            "operations": review_policy.effective_operation_policies_payload(),
        }
    return metadata


def _tool_schema_with_auth_token(schema: dict[str, Any], *, description: str | None = None) -> dict[str, Any]:
    updated = dict(schema)
    properties = dict(updated.get("properties") or {})
    properties["auth_token"] = {
        "type": "string",
        "description": description or "Optional when local-only. Required when MCP write-tool auth is enabled.",
    }
    updated["properties"] = properties
    return updated


def tool_definitions(
    tool_auth: MemoryMutationToolAuth | None = None,
    review_policy: ProposalReviewPolicy | None = None,
) -> list[dict[str, Any]]:
    policy = review_policy or ProposalReviewPolicy()
    definitions: list[dict[str, Any]] = []
    for item in _TOOLS:
        tool = dict(item)
        if tool.get("name") in {"memory_propose", "memory_review_proposal", "memory_commit_proposal"}:
            tool["inputSchema"] = _tool_schema_with_auth_token(
                dict(tool["inputSchema"]),
                description="Credential token used for MCP write-tool authorization when enabled.",
            )
            if tool_auth is not None and tool_auth.enabled:
                tool["description"] = str(tool["description"]) + " Requires auth_token with a permitted role when MCP write-tool auth is enabled."
        tool["x-big-ai-brain"] = _tool_metadata(
            tool_name=str(tool.get("name")),
            tool_auth=tool_auth,
            review_policy=policy,
        )
        definitions.append(tool)
    return definitions


def call_tool(
    service: SemanticMemoryService | FederatedSemanticMemoryBroker,
    name: str,
    arguments: dict[str, Any] | None = None,
    tool_auth: MemoryMutationToolAuth | None = None,
) -> dict[str, Any]:
    args = dict(arguments or {})
    auth_token = None if args.get("auth_token") is None else str(args.pop("auth_token"))
    principal = None
    if tool_auth is not None:
        principal = tool_auth.authorize(tool_name=name, auth_token=auth_token)
    if name == "memory_list_stores":
        if isinstance(service, FederatedSemanticMemoryBroker):
            return service.list_stores_payload()
        return {
            "object": "list",
            "data": [
                {
                    "store_id": "default",
                    "label": service.store_path or "default",
                    "path": service.store_path,
                    "available": True,
                    "active": True,
                    "enabled": True,
                    "mounted": True,
                    "removable": False,
                    "service": service.summary_payload(),
                }
            ],
            "service": service.summary_payload(),
        }
    if name == "memory_set_active_stores":
        if not isinstance(service, FederatedSemanticMemoryBroker):
            raise ValueError("memory_set_active_stores requires a federated semantic-memory broker")
        return service.set_active_stores_payload(store_ids=[str(item) for item in args.get("store_ids", [])])
    if name == "memory_status":
        return service.status_payload()
    if name == "memory_get":
        if isinstance(service, FederatedSemanticMemoryBroker):
            return service.get_memory_payload(str(args["memory_id"]), revision=args.get("revision"), store_id=args.get("store_id"))
        return service.get_memory_payload(str(args["memory_id"]), revision=args.get("revision"))
    if name == "memory_query":
        payload_kwargs = {
            "question": str(args["question"]),
            "top_k": int(args.get("top_k", 5)),
            "include_inactive": bool(args.get("include_inactive", False)),
        }
        if isinstance(service, FederatedSemanticMemoryBroker):
            payload_kwargs["store_ids"] = [str(item) for item in args.get("store_ids", [])] if args.get("store_ids") else None
        return service.query_payload(**payload_kwargs)
    if name == "memory_answer":
        payload_kwargs = {
            "question": str(args["question"]),
            "top_k": int(args.get("top_k", 5)),
            "support_limit": int(args.get("support_limit", 3)),
            "include_inactive": bool(args.get("include_inactive", False)),
        }
        if isinstance(service, FederatedSemanticMemoryBroker):
            payload_kwargs["store_ids"] = [str(item) for item in args.get("store_ids", [])] if args.get("store_ids") else None
        return service.answer_payload(**payload_kwargs)
    if name == "memory_reason":
        payload_kwargs = {
            "question": str(args["question"]),
            "top_k": int(args.get("top_k", 8)),
            "support_limit": int(args.get("support_limit", 3)),
            "neutral_limit": int(args.get("neutral_limit", 3)),
            "conflict_limit": int(args.get("conflict_limit", 3)),
            "include_inactive": bool(args.get("include_inactive", False)),
            "preferred_primary_memory_id": args.get("preferred_primary_memory_id"),
        }
        if isinstance(service, FederatedSemanticMemoryBroker):
            payload_kwargs["store_ids"] = [str(item) for item in args.get("store_ids", [])] if args.get("store_ids") else None
        return service.reasoning_payload(**payload_kwargs)
    if name == "memory_list_proposals":
        if isinstance(service, FederatedSemanticMemoryBroker):
            return service.list_proposals_payload(
                status=None if args.get("status") is None else str(args["status"]),
                store_id=None if args.get("store_id") is None else str(args["store_id"]),
            )
        return service.list_proposals_payload(status=None if args.get("status") is None else str(args["status"]))
    if name == "memory_list_proposal_events":
        if isinstance(service, FederatedSemanticMemoryBroker):
            return service.list_proposal_events_payload(
                proposal_id=None if args.get("proposal_id") is None else str(args["proposal_id"]),
                event_type=None if args.get("event_type") is None else str(args["event_type"]),
                limit=int(args.get("limit", 100)),
                store_id=None if args.get("store_id") is None else str(args["store_id"]),
            )
        return service.list_proposal_events_payload(
            proposal_id=None if args.get("proposal_id") is None else str(args["proposal_id"]),
            event_type=None if args.get("event_type") is None else str(args["event_type"]),
            limit=int(args.get("limit", 100)),
        )
    if name == "memory_propose":
        proposal_args = dict(args)
        target_store_id = None if proposal_args.get("store_id") is None else str(proposal_args.pop("store_id"))
        if isinstance(service, FederatedSemanticMemoryBroker):
            return service.propose_change(
                store_id=target_store_id,
                **proposal_args,
                proposer_role=None if principal is None else principal.role,
                proposer_credential_source=None if principal is None else principal.credential_source,
            )
        return service.propose_change(
            **proposal_args,
            proposer_role=None if principal is None else principal.role,
            proposer_credential_source=None if principal is None else principal.credential_source,
        )
    if name == "memory_review_proposal":
        target_store_id = None if args.get("store_id") is None else str(args["store_id"])
        if isinstance(service, FederatedSemanticMemoryBroker):
            return service.review_proposal(
                store_id=target_store_id,
                proposal_id=str(args["proposal_id"]),
                reviewer=str(args["reviewer"]),
                reviewer_role=None if principal is None else principal.role,
                reviewer_credential_source=None if principal is None else principal.credential_source,
                decision=str(args["decision"]),
                notes=None if args.get("notes") is None else str(args["notes"]),
            )
        return service.review_proposal(
            proposal_id=str(args["proposal_id"]),
            reviewer=str(args["reviewer"]),
            reviewer_role=None if principal is None else principal.role,
            reviewer_credential_source=None if principal is None else principal.credential_source,
            decision=str(args["decision"]),
            notes=None if args.get("notes") is None else str(args["notes"]),
        )
    if name == "memory_commit_proposal":
        target_store_id = None if args.get("store_id") is None else str(args["store_id"])
        if isinstance(service, FederatedSemanticMemoryBroker):
            return service.commit_proposal(
                store_id=target_store_id,
                proposal_id=str(args["proposal_id"]),
                actor=str(args["actor"]),
                actor_role=None if principal is None else principal.role,
                actor_credential_source=None if principal is None else principal.credential_source,
                notes=None if args.get("notes") is None else str(args["notes"]),
            )
        return service.commit_proposal(
            proposal_id=str(args["proposal_id"]),
            actor=str(args["actor"]),
            actor_role=None if principal is None else principal.role,
            actor_credential_source=None if principal is None else principal.credential_source,
            notes=None if args.get("notes") is None else str(args["notes"]),
        )
    raise KeyError(f"Unknown MCP tool: {name}")


def handle_mcp_message(
    service: SemanticMemoryService | FederatedSemanticMemoryBroker,
    message: dict[str, Any],
    tool_auth: MemoryMutationToolAuth | None = None,
) -> dict[str, Any] | None:
    method = message.get("method")
    message_id = message.get("id")

    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "big-ai-brain-semantic-memory",
                    "version": "0.2.0",
                },
                "capabilities": {
                    "tools": {},
                },
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": message_id, "result": {}}
    if method == "tools/list":
        review_policy = getattr(service, "review_policy", ProposalReviewPolicy())
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": {"tools": tool_definitions(tool_auth, review_policy)},
        }
    if method == "tools/call":
        params = dict(message.get("params") or {})
        name = str(params["name"])
        arguments = dict(params.get("arguments") or {})
        try:
            payload = call_tool(service, name, arguments, tool_auth=tool_auth)
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(payload, indent=2),
                        }
                    ],
                    "isError": False,
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({"error": str(exc)}, indent=2),
                        }
                    ],
                    "isError": True,
                },
            }
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {
            "code": -32601,
            "message": f"Method not found: {method}",
        },
    }