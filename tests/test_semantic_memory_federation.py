from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from big_ai_brain.federated_memory import FederatedSemanticMemoryBroker
from big_ai_brain.semantic_memory_mcp import handle_mcp_message
from rwif_retriever import ArrayActivationProvider
from rwif_semantic_memory import ProvenanceRef, SemanticMemoryObject, SemanticMemoryStore, save_semantic_memory_store


class FederatedSemanticMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.question = "What is the capital of France?"
        self.activations = {
            "Paris is the capital of France.": [3.0, 0.2, 0.0, 0.0],
            "France is a country in Western Europe.": [2.7, 0.15, 0.0, 0.0],
            "Lyon is a city in France.": [2.3, 0.17, 0.0, 0.0],
            "What is the capital of France?": [2.9, 0.22, 0.0, 0.0],
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_config_backed_broker_merges_duplicate_hits(self) -> None:
        broker = FederatedSemanticMemoryBroker.from_config(self._write_config())

        status_payload = broker.status_payload()
        self.assertEqual(status_payload["mode"], "federated")
        self.assertEqual(sorted(status_payload["active_store_ids"]), ["core", "personal"])

        query_payload = broker.query_payload(question=self.question, top_k=3)
        self.assertEqual(query_payload["data"][0]["memory"]["canonical_text"], "Paris is the capital of France.")
        self.assertEqual(query_payload["data"][0]["duplicate_store_ids"], ["core", "personal"])

        selection_payload = broker.set_active_stores_payload(store_ids=["personal"])
        self.assertEqual(selection_payload["active_store_ids"], ["personal"])

        narrowed_payload = broker.query_payload(question=self.question, top_k=3)
        self.assertEqual(narrowed_payload["data"][0]["duplicate_store_ids"], ["personal"])

    def test_config_roots_and_access_mode_control_mutations(self) -> None:
        broker = FederatedSemanticMemoryBroker.from_config(self._write_config())

        stores_payload = broker.list_stores_payload()
        core_store = next(item for item in stores_payload["data"] if item["store_id"] == "core")
        personal_store = next(item for item in stores_payload["data"] if item["store_id"] == "personal")
        self.assertEqual(core_store["access_mode"], "read-write")
        self.assertTrue(core_store["writable"])
        self.assertEqual(personal_store["access_mode"], "read-only")
        self.assertFalse(personal_store["writable"])

        proposal = broker.propose_change(
            store_id="core",
            operation="create",
            proposer="tester",
            memory_id="core-new-memory",
            title="New writable memory",
            canonical_text="Paris has many bridges.",
        )
        self.assertEqual(proposal["store"]["store_id"], "core")

        with self.assertRaises(PermissionError):
            broker.propose_change(
                store_id="personal",
                operation="create",
                proposer="tester",
                memory_id="personal-new-memory",
                title="Should fail",
                canonical_text="This store is read only.",
            )

    def test_mcp_handler_exposes_federated_store_tools(self) -> None:
        broker = FederatedSemanticMemoryBroker.from_config(self._write_config())

        list_response = handle_mcp_message(
            broker,
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        tool_names = {tool["name"] for tool in list_response["result"]["tools"]}
        self.assertIn("memory_list_stores", tool_names)
        self.assertIn("memory_set_active_stores", tool_names)

        stores_response = handle_mcp_message(
            broker,
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "memory_list_stores", "arguments": {}}},
        )
        stores_payload = json.loads(stores_response["result"]["content"][0]["text"])
        self.assertEqual(len(stores_payload["data"]), 2)

        activate_response = handle_mcp_message(
            broker,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "memory_set_active_stores",
                    "arguments": {"store_ids": ["core"]},
                },
            },
        )
        activate_payload = json.loads(activate_response["result"]["content"][0]["text"])
        self.assertEqual(activate_payload["active_store_ids"], ["core"])

        query_response = handle_mcp_message(
            broker,
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "memory_query",
                    "arguments": {"question": self.question, "top_k": 2},
                },
            },
        )
        query_payload = json.loads(query_response["result"]["content"][0]["text"])
        self.assertEqual(query_payload["data"][0]["store"]["store_id"], "core")

    def _write_config(self) -> Path:
        core_store_path = self.temp_path / "core_memory.rwif"
        personal_store_path = self.temp_path / "personal_memory.rwif"
        self._save_store(
            core_store_path,
            [
                SemanticMemoryObject(
                    memory_id="core-paris",
                    revision=1,
                    title="Paris capital fact",
                    canonical_text="Paris is the capital of France.",
                    kind="fact",
                    summary="Paris is France's capital city.",
                    facts=("Paris is the capital of France.",),
                    tags=("geography", "capital", "france"),
                    provenance=(
                        ProvenanceRef(
                            source_id="doc-core-paris",
                            source_type="document",
                            locator="paragraph:1",
                            quoted_text="Paris is the capital of France.",
                        ),
                    ),
                ),
                SemanticMemoryObject(
                    memory_id="core-context",
                    revision=1,
                    title="France context",
                    canonical_text="France is a country in Western Europe.",
                    kind="fact",
                    facts=("France is a country in Western Europe.",),
                    tags=("geography", "country", "france"),
                ),
            ],
        )
        self._save_store(
            personal_store_path,
            [
                SemanticMemoryObject(
                    memory_id="personal-paris",
                    revision=1,
                    title="Travel note about Paris",
                    canonical_text="Paris is the capital of France.",
                    kind="fact",
                    summary="Personal notes agree that Paris is France's capital.",
                    facts=("Paris is the capital of France.",),
                    tags=("geography", "travel", "france"),
                    provenance=(
                        ProvenanceRef(
                            source_id="doc-personal-paris",
                            source_type="note",
                            locator="entry:3",
                            quoted_text="Paris is the capital of France.",
                        ),
                    ),
                ),
                SemanticMemoryObject(
                    memory_id="personal-lyon",
                    revision=1,
                    title="Travel note about Lyon",
                    canonical_text="Lyon is a city in France.",
                    kind="fact",
                    facts=("Lyon is a city in France.",),
                    tags=("geography", "travel", "france"),
                ),
            ],
        )
        config_path = self.temp_path / "federated_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "metadata": {"name": "test federation"},
                    "default_provider": {"name": "array", "activations": self.activations},
                    "store_roots": {
                        "local": str(self.temp_path),
                    },
                    "default_active_store_ids": ["core", "personal"],
                    "stores": [
                        {
                            "store_id": "core",
                            "label": "Core knowledge",
                            "root": "local",
                            "relative_path": core_store_path.name,
                            "access_mode": "read-write",
                            "domain_tags": ["core", "general"],
                            "trust_weight": 1.0,
                        },
                        {
                            "store_id": "personal",
                            "label": "Personal records",
                            "root": "local",
                            "relative_path": personal_store_path.name,
                            "access_mode": "read-only",
                            "domain_tags": ["personal", "travel"],
                            "trust_weight": 1.2,
                            "removable": True,
                            "required_mount_path": str(self.temp_path),
                        },
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return config_path

    def _save_store(self, path: Path, objects: list[SemanticMemoryObject]) -> None:
        provider = ArrayActivationProvider(activations=self.activations)
        store = SemanticMemoryStore.from_objects(
            provider=provider,
            objects=objects,
            calibration_texts=[item.canonical_text for item in objects],
            top_k_waves=4,
        )
        save_semantic_memory_store(path, store)


if __name__ == "__main__":
    unittest.main()