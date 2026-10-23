"""Run the CT-200 v1 -> v2 workflow against isolated temporary storage.

This script is intentionally standalone and does not modify the project's
application data, output, PDFs, databases, or environment files.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import app as api
import generation
from selection_store import get_generation


PROJECT_ROOT = Path(__file__).resolve().parent
V1_PDF = PROJECT_ROOT / "data" / "ct200_manual.pdf"
V2_PDF = PROJECT_ROOT / "data" / "ct200_manual_v2.pdf"


class WorkflowFailure(RuntimeError):
    """A workflow assertion failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkflowFailure(message)


def required_field(record: dict[str, Any], field: str, label: str) -> Any:
    """Read a required field while reporting the actual record shape."""
    if field not in record:
        raise WorkflowFailure(
            f"{label} missing required field {field!r}; "
            f"actual keys: {sorted(record.keys())}"
        )
    return record[field]


def response_json(response: Any, step: str) -> dict[str, Any]:
    if response.status_code >= 400:
        raise WorkflowFailure(
            f"{step}: HTTP {response.status_code}: {response.text[:500]}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise WorkflowFailure(f"{step}: response was not JSON") from exc
    require(isinstance(payload, dict), f"{step}: expected a JSON object")
    return payload


def print_pass(step: str, evidence: str) -> None:
    print(f"PASS - {step}: {evidence}")


def print_fail(step: str, error: BaseException) -> None:
    print(f"FAIL - {step}: {error}")


def changed_safe_node(
    v1_nodes: list[dict[str, Any]], v2_nodes: list[dict[str, Any]]
) -> dict[str, Any]:
    """Choose a changed, non-root node whose text is safe for mock validation."""
    v2_by_id = {node["id"]: node for node in v2_nodes}
    banned = re.compile(
        r"\b(?:do not|does not|did not|must not|should not|never|avoid|prohibited|cannot|can't)\b",
        re.IGNORECASE,
    )
    candidates = []
    for node in v1_nodes:
        body = " ".join(str(node.get("body", "")).split())
        target = v2_by_id.get(node.get("id"))
        target_body = " ".join(str(target.get("body", "")).split()) if target else ""
        if (
            node.get("level", 0) > 0
            and body
            and target is not None
            and body != target_body
            and len(body) <= 1100
            and not banned.search(f"{node.get('heading', '')} {body}")
        ):
            candidates.append(node)
    if not candidates:
        raise WorkflowFailure(
            "could not find a changed, non-root v1 node with validator-safe text"
        )
    return max(candidates, key=lambda node: len(str(node.get("body", ""))))


def mocked_cases(node: dict[str, Any]) -> dict[str, Any]:
    node_id = node["id"]
    # Each phrase is copied from the selected 4.2 Error Codes table row.
    # Keeping the claims short preserves the validator's ordered phrase rule.
    return {
        "cases": [
            {
                "title": "E1 Cuff not connected or leak detected",
                "requirement": "Cuff not connected or leak detected",
                "preconditions": [],
                "steps": ["Cuff not connected or leak detected"],
                "expected_result": "Cuff not connected or leak detected",
                "source_node_ids": [node_id],
            },
            {
                "title": "E2 Motion artifact detected during measurement",
                "requirement": "Motion artifact detected during measurement",
                "preconditions": [],
                "steps": ["Motion artifact detected during measurement"],
                "expected_result": "Motion artifact detected during measurement",
                "source_node_ids": [node_id],
            },
            {
                "title": "E3 Overpressure condition",
                "requirement": "Overpressure condition",
                "preconditions": [],
                "steps": ["Overpressure condition"],
                "expected_result": "Overpressure condition",
                "source_node_ids": [node_id],
            },
        ]
    }


def main() -> int:
    original_storage_dir = api.app.state.storage_dir
    original_provider = generation.request_groq_completion
    workflow_step = "initialization"

    try:
        require(V1_PDF.is_file(), f"missing supplied PDF: {V1_PDF}")
        require(V2_PDF.is_file(), f"missing supplied PDF: {V2_PDF}")

        with tempfile.TemporaryDirectory(prefix="ct200-e2e-") as temp_dir:
            isolated_storage = Path(temp_dir) / "storage"
            api.app.state.storage_dir = isolated_storage

            def mocked_provider(messages: list[dict[str, str]], model: str) -> tuple[str, str]:
                del messages
                return mocked_raw_response, "mocked/local-provider"

            with TestClient(api.app) as client:
                workflow_step = "upload v1"
                v1_response = client.post(
                    "/versions/upload",
                    params={"filename": V1_PDF.name},
                    content=V1_PDF.read_bytes(),
                    headers={"content-type": "application/pdf"},
                )
                v1 = response_json(v1_response, workflow_step)
                require(v1.get("version_number") == 1, f"{workflow_step}: expected version 1")
                print_pass(workflow_step, f"version_id={v1['id']}, version_number=1")

                workflow_step = "upload v2 without overwriting v1"
                v2_response = client.post(
                    "/versions/upload",
                    params={"filename": V2_PDF.name},
                    content=V2_PDF.read_bytes(),
                    headers={"content-type": "application/pdf"},
                )
                v2 = response_json(v2_response, workflow_step)
                require(v2.get("version_number") == 2, f"{workflow_step}: expected version 2")
                v1_retrieved = response_json(
                    client.get(f"/versions/{v1['id']}"), "retrieve v1 after v2"
                )
                require(v1_retrieved["id"] == v1["id"], f"{workflow_step}: v1 was not retained")
                print_pass(
                    workflow_step,
                    f"v1={v1['id']} retained, v2={v2['id']} created",
                )

                workflow_step = "compare versions"
                comparison = response_json(
                    client.get(
                        "/versions/compare",
                        params={
                            "old_version_id": v1["id"],
                            "new_version_id": v2["id"],
                        },
                    ),
                    workflow_step,
                )
                summary = comparison.get("summary", {})
                print_pass(workflow_step, f"summary={summary}")

                workflow_step = "select changed non-root v1 node"
                v1_tree = response_json(
                    client.get(f"/versions/{v1['id']}/tree"), "retrieve v1 tree"
                )
                v2_tree = response_json(
                    client.get(f"/versions/{v2['id']}/tree"), "retrieve v2 tree"
                )
                node = changed_safe_node(v1_tree["nodes"], v2_tree["nodes"])
                selection_response = client.post(
                    "/selections",
                    json={
                        "name": "CT-200 E2E verification selection",
                        "version_id": v1["id"],
                        "node_ids": [node["id"]],
                    },
                )
                selection = response_json(selection_response, workflow_step)
                require(selection["version_id"] == v1["id"], "selection is not pinned to v1")
                require(selection["node_ids"] == [node["id"]], "selection node mismatch")
                print_pass(
                    workflow_step,
                    f"selection_id={selection['id']}, heading={node['heading']!r}, node_id={node['id']}",
                )

                mocked_raw_response = json.dumps(mocked_cases(node), ensure_ascii=False)
                generation.request_groq_completion = mocked_provider

                workflow_step = "generate mocked test cases"
                generated_response = client.post(
                    f"/selections/{selection['id']}/generate"
                )
                generated = response_json(generated_response, workflow_step)
                require(3 <= len(generated["cases"]) <= 5, "expected 3-5 generated cases")
                print_pass(
                    workflow_step,
                    f"generation_id={generated['id']}, case_count={len(generated['cases'])}",
                )

                workflow_step = "verify exact source provenance"
                saved_generation = response_json(
                    client.get(f"/generations/{generated['id']}"), workflow_step
                )
                print(
                    "INFO - generation API keys: "
                    f"{sorted(saved_generation.keys())}"
                )
                api_cases = saved_generation.get("cases")
                if isinstance(api_cases, list) and api_cases:
                    print(
                        "INFO - generation API case keys: "
                        f"{sorted(api_cases[0].keys())}"
                    )
                persisted_generation = get_generation(
                    api.app.state.storage_dir, generated["id"]
                )
                require(
                    persisted_generation is not None,
                    "generation was not found in isolated storage",
                )
                print(
                    "INFO - persisted generation keys: "
                    f"{sorted(persisted_generation.keys())}"
                )
                persisted_cases = persisted_generation.get("cases")
                require(
                    isinstance(persisted_cases, list),
                    "persisted generation cases is not a list: "
                    f"{type(persisted_cases).__name__}",
                )
                if persisted_cases:
                    print(
                        "INFO - persisted case keys: "
                        f"{sorted(persisted_cases[0].keys())}"
                    )
                persisted_selection_id = required_field(
                    persisted_generation, "selection_id", "persisted generation"
                )
                persisted_source_version_id = required_field(
                    persisted_generation, "source_version_id", "persisted generation"
                )
                persisted_source_node_ids = required_field(
                    persisted_generation, "source_node_ids", "persisted generation"
                )
                require(
                    persisted_selection_id == selection["id"],
                    "generation is not linked to the selected selection",
                )
                require(
                    persisted_source_version_id == v1["id"],
                    "persisted generation is not pinned to v1",
                )
                require(
                    persisted_source_version_id == selection["version_id"],
                    "persisted generation source version differs from the selection version",
                )
                require(
                    persisted_source_node_ids == selection["node_ids"],
                    "persisted generation source node IDs differ from the selection node IDs",
                )
                require(
                    node["id"] in persisted_source_node_ids,
                    "persisted generation source node IDs omit the selected node",
                )
                for field in ("source_snapshot", "source_hashes", "snapshot_hash"):
                    require(field in persisted_generation, f"storage provenance field missing: {field}")
                    require(
                        persisted_generation[field] == selection[field],
                        f"persisted provenance field mismatch: {field}",
                    )
                api_selection_id = required_field(
                    saved_generation, "selection_id", "generation API response"
                )
                api_source_version_id = required_field(
                    saved_generation, "source_version_id", "generation API response"
                )
                api_source_node_ids = required_field(
                    saved_generation, "source_node_ids", "generation API response"
                )
                require(
                    api_selection_id == selection["id"],
                    "API generation is not linked to the selected selection",
                )
                require(
                    api_source_version_id == v1["id"],
                    "API generation is not pinned to v1",
                )
                require(
                    api_source_node_ids == selection["node_ids"],
                    "API generation source node IDs differ from the selection node IDs",
                )
                require(
                    node["id"] in api_source_node_ids,
                    "API generation source node IDs omit the selected node",
                )
                for field in ("source_snapshot", "source_hashes", "snapshot_hash"):
                    require(
                        saved_generation[field] == selection[field],
                        f"API provenance field mismatch: {field}",
                    )
                require(
                    isinstance(api_cases, list),
                    "API generation cases is not a list",
                )
                require(
                    api_cases == persisted_cases,
                    "API and persisted generated cases differ",
                )
                for case_source, cases in (("API", api_cases), ("persisted", persisted_cases)):
                    for case in cases:
                        case_source_node_ids = case.get("source_node_ids")
                        require(
                            isinstance(case_source_node_ids, list),
                            f"{case_source} case lacks source_node_ids; actual case keys: "
                            f"{sorted(case.keys())}",
                        )
                        require(
                            case_source_node_ids == [node["id"]],
                            f"{case_source} case references an unexpected source node",
                        )
                persisted_snapshot = required_field(
                    persisted_generation, "source_snapshot", "persisted generation"
                )
                persisted_hashes = required_field(
                    persisted_generation, "source_hashes", "persisted generation"
                )
                for snapshot in persisted_snapshot:
                    node_id = snapshot["node_id"]
                    require(
                        persisted_hashes[node_id]
                        == hashlib.sha256(snapshot["body"].encode("utf-8")).hexdigest(),
                        f"source hash mismatch for node {node_id}",
                    )
                expected_response_hash = hashlib.sha256(
                    mocked_raw_response.encode("utf-8")
                ).hexdigest()
                require(
                    required_field(
                        persisted_generation, "response_hash", "persisted generation"
                    )
                    == expected_response_hash,
                    "response hash mismatch",
                )
                persisted_snapshot_hash = required_field(
                    persisted_generation, "snapshot_hash", "persisted generation"
                )
                print_pass(
                    workflow_step,
                    f"source_version_id={persisted_source_version_id}, "
                    f"snapshot_hash={persisted_snapshot_hash}",
                )

                workflow_step = "check staleness against v2"
                staleness = response_json(
                    client.get(
                        f"/generations/{generated['id']}/staleness",
                        params={"against_version_id": v2["id"]},
                    ),
                    workflow_step,
                )
                require(staleness["is_stale"] is True, "expected generation to be stale against v2")
                print_pass(
                    workflow_step,
                    f"checked_against_version={v2['id']}, is_stale={staleness['is_stale']}",
                )

                workflow_step = "retrieve generated cases with staleness"
                retrieved = response_json(
                    client.get(f"/generations/{saved_generation['id']}"), workflow_step
                )
                require(retrieved["cases"] == saved_generation["cases"], "retrieved cases differ")
                require(retrieved["source_snapshot"] == selection["source_snapshot"], "retrieved snapshot differs")
                require("staleness" in retrieved, "retrieval omitted staleness metadata")
                require(retrieved["staleness"]["is_stale"] is True, "retrieval staleness is not true")
                print_pass(
                    workflow_step,
                    f"generation_id={retrieved['id']}, case_count={len(retrieved['cases'])}, "
                    f"is_stale={retrieved['staleness']['is_stale']}",
                )

            print("PASS - isolated workflow completed; temporary storage will be removed")
        return 0
    except Exception as exc:
        print_fail(workflow_step, exc)
        traceback.print_exc()
        return 1
    finally:
        api.app.state.storage_dir = original_storage_dir
        generation.request_groq_completion = original_provider


if __name__ == "__main__":
    sys.exit(main())
