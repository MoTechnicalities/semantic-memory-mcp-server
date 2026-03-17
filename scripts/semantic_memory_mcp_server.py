from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from big_ai_brain.federated_memory import FederatedSemanticMemoryBroker
from big_ai_brain.semantic_memory_mcp import build_memory_mutation_tool_auth, handle_mcp_message
from big_ai_brain.semantic_memory_service import ProposalReviewPolicy, SemanticMemoryService
from rwif_retriever.providers import TransformersActivationProvider
from rwif_semantic_memory import load_semantic_memory_store


def _policy_override_from_args(args: argparse.Namespace, prefix: str) -> dict[str, object]:
    override: dict[str, object] = {}
    min_count = getattr(args, f"memory_{prefix}_min_provenance_count")
    min_confidence = getattr(args, f"memory_{prefix}_min_provenance_confidence")
    allowed_types = getattr(args, f"memory_{prefix}_allowed_source_types")
    require_locator = getattr(args, f"memory_{prefix}_require_provenance_locator")
    require_quote = getattr(args, f"memory_{prefix}_require_provenance_quote")
    require_notes = getattr(args, f"memory_{prefix}_require_review_notes")
    if min_count is not None:
        override["min_provenance_count"] = max(0, int(min_count))
    if min_confidence is not None:
        override["min_provenance_confidence"] = float(min_confidence)
    if allowed_types:
        override["allowed_source_types"] = [str(item) for item in allowed_types if str(item).strip()]
    if require_locator:
        override["require_locator"] = True
    if require_quote:
        override["require_quoted_text"] = True
    if require_notes:
        override["require_review_notes"] = True
    return override


def _build_semantic_review_policy_from_args(args: argparse.Namespace) -> ProposalReviewPolicy:
    operation_overrides = {
        operation: override
        for operation, override in {
            "create": _policy_override_from_args(args, "create"),
            "revise": _policy_override_from_args(args, "revise"),
            "merge": _policy_override_from_args(args, "merge"),
            "deprecate": _policy_override_from_args(args, "deprecate"),
            "contradict": _policy_override_from_args(args, "contradict"),
        }.items()
        if override
    }
    return ProposalReviewPolicy(
        min_provenance_count=max(0, args.memory_min_provenance_count),
        min_provenance_confidence=args.memory_min_provenance_confidence,
        allowed_source_types=tuple(str(item) for item in args.memory_allowed_source_types if str(item).strip()),
        require_locator=args.memory_require_provenance_locator,
        require_quoted_text=args.memory_require_provenance_quote,
        require_review_notes=args.memory_require_review_notes,
        operation_overrides=operation_overrides,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expose the semantic memory service as a thin MCP stdio tool server.")
    parser.add_argument("--semantic-store", help="RWIF semantic memory store path.")
    parser.add_argument("--federated-config", help="JSON config describing federated semantic-memory roots, stores, and access policy.")
    parser.add_argument("--federated-registry", help="Legacy alias for --federated-config.")
    parser.add_argument("--semantic-model-id", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--semantic-layer-index", type=int, default=-1)
    parser.add_argument("--semantic-pooling", choices=["last_token", "mean"], default="mean")
    parser.add_argument("--semantic-device")
    parser.add_argument("--semantic-max-length", type=int, default=128)
    parser.add_argument("--semantic-provider-warmup", choices=["off", "startup"], default="off")
    parser.add_argument("--default-active-store", action="append", default=[])
    parser.add_argument("--memory-min-provenance-count", type=int, default=0)
    parser.add_argument("--memory-min-provenance-confidence", type=float)
    parser.add_argument("--memory-allowed-source-type", dest="memory_allowed_source_types", action="append", default=[])
    parser.add_argument("--memory-require-provenance-locator", action="store_true")
    parser.add_argument("--memory-require-provenance-quote", action="store_true")
    parser.add_argument("--memory-require-review-notes", action="store_true")
    parser.add_argument("--memory-proposal-api-key", action="append", default=[])
    parser.add_argument("--memory-review-api-key", action="append", default=[])
    parser.add_argument("--memory-commit-api-key", action="append", default=[])
    parser.add_argument("--memory-admin-api-key", action="append", default=[])
    parser.add_argument("--memory-create-min-provenance-count", type=int)
    parser.add_argument("--memory-create-min-provenance-confidence", type=float)
    parser.add_argument("--memory-create-allowed-source-type", dest="memory_create_allowed_source_types", action="append", default=[])
    parser.add_argument("--memory-create-require-provenance-locator", action="store_true")
    parser.add_argument("--memory-create-require-provenance-quote", action="store_true")
    parser.add_argument("--memory-create-require-review-notes", action="store_true")
    parser.add_argument("--memory-revise-min-provenance-count", type=int)
    parser.add_argument("--memory-revise-min-provenance-confidence", type=float)
    parser.add_argument("--memory-revise-allowed-source-type", dest="memory_revise_allowed_source_types", action="append", default=[])
    parser.add_argument("--memory-revise-require-provenance-locator", action="store_true")
    parser.add_argument("--memory-revise-require-provenance-quote", action="store_true")
    parser.add_argument("--memory-revise-require-review-notes", action="store_true")
    parser.add_argument("--memory-merge-min-provenance-count", type=int)
    parser.add_argument("--memory-merge-min-provenance-confidence", type=float)
    parser.add_argument("--memory-merge-allowed-source-type", dest="memory_merge_allowed_source_types", action="append", default=[])
    parser.add_argument("--memory-merge-require-provenance-locator", action="store_true")
    parser.add_argument("--memory-merge-require-provenance-quote", action="store_true")
    parser.add_argument("--memory-merge-require-review-notes", action="store_true")
    parser.add_argument("--memory-deprecate-min-provenance-count", type=int)
    parser.add_argument("--memory-deprecate-min-provenance-confidence", type=float)
    parser.add_argument("--memory-deprecate-allowed-source-type", dest="memory_deprecate_allowed_source_types", action="append", default=[])
    parser.add_argument("--memory-deprecate-require-provenance-locator", action="store_true")
    parser.add_argument("--memory-deprecate-require-provenance-quote", action="store_true")
    parser.add_argument("--memory-deprecate-require-review-notes", action="store_true")
    parser.add_argument("--memory-contradict-min-provenance-count", type=int)
    parser.add_argument("--memory-contradict-min-provenance-confidence", type=float)
    parser.add_argument("--memory-contradict-allowed-source-type", dest="memory_contradict_allowed_source_types", action="append", default=[])
    parser.add_argument("--memory-contradict-require-provenance-locator", action="store_true")
    parser.add_argument("--memory-contradict-require-provenance-quote", action="store_true")
    parser.add_argument("--memory-contradict-require-review-notes", action="store_true")
    args = parser.parse_args()
    selected_modes = [bool(args.semantic_store), bool(args.federated_config), bool(args.federated_registry)]
    if sum(1 for enabled in selected_modes if enabled) != 1:
        raise SystemExit("Provide exactly one of --semantic-store, --federated-config, or --federated-registry")
    return args


def build_service(args: argparse.Namespace) -> SemanticMemoryService:
    provider = TransformersActivationProvider(
        model_id=args.semantic_model_id,
        layer_index=args.semantic_layer_index,
        pooling=args.semantic_pooling,
        device=args.semantic_device,
        max_length=args.semantic_max_length,
    )
    return SemanticMemoryService(
        semantic_store=load_semantic_memory_store(args.semantic_store),
        provider=provider,
        store_path=args.semantic_store,
        provider_name="transformers",
        provider_config={
            "model_id": args.semantic_model_id,
            "layer_index": args.semantic_layer_index,
            "pooling": args.semantic_pooling,
            "device": args.semantic_device,
            "max_length": args.semantic_max_length,
        },
        review_policy=_build_semantic_review_policy_from_args(args),
    )


def build_backend(args: argparse.Namespace) -> SemanticMemoryService | FederatedSemanticMemoryBroker:
    review_policy = _build_semantic_review_policy_from_args(args)
    federated_path = args.federated_config or args.federated_registry
    if federated_path:
        broker = FederatedSemanticMemoryBroker.from_config(
            federated_path,
            default_provider_config={
                "model_id": args.semantic_model_id,
                "layer_index": args.semantic_layer_index,
                "pooling": args.semantic_pooling,
                "device": args.semantic_device,
                "max_length": args.semantic_max_length,
            },
            review_policy=review_policy,
            default_active_store_ids=[str(item) for item in args.default_active_store if str(item).strip()],
        )
        if args.semantic_provider_warmup == "startup":
            broker.warmup()
        return broker
    service = build_service(args)
    if args.semantic_provider_warmup == "startup":
        warmup = getattr(service.provider, "warmup", None)
        if callable(warmup):
            warmup(sample_text="semantic memory warmup")
        else:
            service._encode_text("semantic memory warmup")
    return service


def _read_message(framing_mode: str | None) -> tuple[dict | None, str | None]:
    first_line = sys.stdin.buffer.readline()
    if not first_line:
        return None, framing_mode

    if framing_mode == "newline" or (framing_mode is None and first_line[:1] == b"{"):
        body = first_line.decode("utf-8").strip()
        if not body:
            return None, "newline"
        return json.loads(body), "newline"

    headers: dict[str, str] = {}
    line = first_line
    while True:
        if line in {b"\r\n", b"\n"}:
            break
        decoded = line.decode("utf-8").strip()
        if not decoded:
            break
        key, _, value = decoded.partition(":")
        headers[key.strip().lower()] = value.strip()
        line = sys.stdin.buffer.readline()
        if not line:
            return None, "content-length"
    content_length = int(headers.get("content-length", "0"))
    if content_length <= 0:
        return None, "content-length"
    body = sys.stdin.buffer.read(content_length)
    if not body:
        return None, "content-length"
    return json.loads(body.decode("utf-8")), "content-length"


def _write_message(payload: dict, framing_mode: str | None) -> None:
    body = json.dumps(payload)
    if framing_mode == "newline":
        sys.stdout.write(body)
        sys.stdout.write("\n")
        sys.stdout.flush()
        return
    encoded = body.encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(encoded)}\r\n\r\n".encode("utf-8"))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def main() -> None:
    args = parse_args()
    service = build_backend(args)
    tool_auth = build_memory_mutation_tool_auth(
        proposal_api_keys=args.memory_proposal_api_key,
        review_api_keys=args.memory_review_api_key,
        commit_api_keys=args.memory_commit_api_key,
        admin_api_keys=args.memory_admin_api_key,
    )
    framing_mode: str | None = None
    while True:
        message, framing_mode = _read_message(framing_mode)
        if message is None:
            break
        response = handle_mcp_message(service, message, tool_auth=tool_auth)
        if response is not None:
            _write_message(response, framing_mode)


if __name__ == "__main__":
    main()