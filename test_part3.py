"""Deterministic tests for saved selections, provenance, and staleness."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import app as api
import generation
from parser import TreeNode, save_tree
from selection_store import create_generation, get_generation
from version_store import create_version, version_file_paths


@pytest.fixture
def storage(tmp_path):
    previous_storage_dir = api.app.state.storage_dir
    storage_dir = tmp_path / "part3_storage"
    api.app.state.storage_dir = storage_dir
    try:
        yield storage_dir
    finally:
        api.app.state.storage_dir = previous_storage_dir


def add_version(storage_dir, version_id, sections, uploaded_at):
    nodes = [TreeNode(id="root", heading="ROOT", level=0)]
    for node_id, heading, body in sections:
        node = TreeNode(
            id=node_id,
            heading=heading,
            level=1,
            body=body,
            parent_id="root",
        )
        node.compute_hash()
        nodes[0].children.append(node_id)
        nodes.append(node)
    nodes[0].compute_hash()
    _, tree_path = version_file_paths(storage_dir, version_id)
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    save_tree(nodes, tree_path)
    return create_version(
        storage_dir=storage_dir,
        version_id=version_id,
        filename=f"{version_id}.pdf",
        uploaded_at=uploaded_at,
        content_hash=hashlib.sha256(version_id.encode()).hexdigest(),
        node_count=len(nodes),
    )


def create_selection(version_id="v1", node_ids=None):
    return api.post_selection(
        api.SelectionCreate(
            name="Safety checks",
            version_id=version_id,
            node_ids=node_ids or ["node-a", "node-b"],
        )
    )


def generated_cases(node_id="node-a"):
    return {
        "cases": [
            {
                "title": "Systolic reading is displayed",
                "requirement": "The monitor displays a systolic reading.",
                "preconditions": [],
                "steps": ["View the monitor display for systolic readings."],
                "expected_result": "The monitor displays the systolic reading.",
                "source_node_ids": [node_id],
            },
            {
                "title": "Diastolic reading is displayed",
                "requirement": "The monitor displays a diastolic reading.",
                "preconditions": [],
                "steps": ["View the monitor display for diastolic readings."],
                "expected_result": "The monitor displays the diastolic reading.",
                "source_node_ids": [node_id],
            },
            {
                "title": "Pressure readings are displayed",
                "requirement": "The monitor displays pressure readings.",
                "preconditions": [],
                "steps": ["View the monitor display for pressure readings."],
                "expected_result": "The monitor displays pressure readings.",
                "source_node_ids": [node_id],
            },
        ]
    }


def make_selection_with_source(storage, body="The monitor displays systolic and diastolic pressure readings."):
    add_version(storage, "v1", [("node-a", "Pressure readings", body)], "2026-10-23T10:00:00+00:00")
    return create_selection(node_ids=["node-a"])


def cleaning_source_snapshot():
    return [
        {
            "node_id": "cleaning-node",
            "heading": "6.1 Cleaning Instructions",
            "body": (
                "Wipe the device body and cuff exterior with a soft, dry cloth or one "
                "lightly dampened with water. Do not submerge the device or cuff, and do "
                "not use alcohol, solvents, or abrasive cleaners."
            ),
        }
    ]


def safe_cleaning_cases():
    node_id = "cleaning-node"
    return {
        "cases": [
            {
                "title": "Wipe Device Body with Soft Dry Cloth",
                "requirement": "Wipe the device body and cuff exterior with a soft, dry cloth.",
                "preconditions": [],
                "steps": ["Wipe the device body and cuff exterior with a soft, dry cloth."],
                "expected_result": "Wipe the device body and cuff exterior with a soft, dry cloth.",
                "source_node_ids": [node_id],
            },
            {
                "title": "Wipe with One Lightly Dampened with Water",
                "requirement": "Wipe the device body and cuff exterior with one lightly dampened with water.",
                "preconditions": [],
                "steps": ["Wipe the device body and cuff exterior with one lightly dampened with water."],
                "expected_result": "Wipe the device body and cuff exterior with one lightly dampened with water.",
                "source_node_ids": [node_id],
            },
            {
                "title": "Do Not Use Alcohol or Abrasive Cleaners",
                "requirement": "Do not use alcohol, solvents, or abrasive cleaners.",
                "preconditions": [],
                "steps": ["Do not use alcohol, solvents, or abrasive cleaners."],
                "expected_result": "Do not use alcohol, solvents, or abrasive cleaners.",
                "source_node_ids": [node_id],
            },
        ]
    }


def groq_submersion_case():
    return {
        "title": "Verify Submersion Not Allowed",
        "requirement": "Confirm that submerging the device or cuff is not permitted",
        "preconditions": [],
        "steps": [
            "Ensure the device is powered off.",
            "Submerge the device in water.",
            "Remove the device from water and dry it.",
            "Observe the device for operational status and any signs of damage.",
        ],
        "expected_result": (
            "Device should not be operational or is considered unsuitable for submersion; "
            "submerging is not recommended per the instructions."
        ),
        "source_node_ids": ["cleaning-node"],
    }


def groq_alcohol_case():
    return {
        "title": "Avoid Use of Alcohol on Display",
        "requirement": "Ensure alcohol, solvents, or abrasive cleaners are not used on the display",
        "preconditions": [],
        "steps": [
            "Ensure the device is powered off.",
            "Apply a small amount of alcohol to a clean cloth.",
            "Wipe the display surface with the cloth.",
            "Observe the display for any changes in appearance or functionality.",
        ],
        "expected_result": (
            "Device remains operational; however, cleaning with alcohol is not recommended "
            "as per the instructions."
        ),
        "source_node_ids": ["cleaning-node"],
    }


def test_selection_creation_retrieval_and_hash_snapshot(storage):
    add_version(
        storage,
        "v1",
        [("node-a", "Safety", "Keep this exact source text.\nSecond line.")],
        "2026-10-23T10:00:00+00:00",
    )

    selection = create_selection(node_ids=["node-a"])
    assert selection["name"] == "Safety checks"
    assert selection["version_id"] == "v1"
    assert selection["node_ids"] == ["node-a"]
    assert selection["source_snapshot"][0]["body"] == "Keep this exact source text.\nSecond line."
    assert selection["source_hashes"]["node-a"] == hashlib.sha256(
        b"Keep this exact source text.\nSecond line."
    ).hexdigest()
    assert len(selection["snapshot_hash"]) == 64

    retrieved = api.get_saved_selection(selection["id"])
    assert retrieved == selection


def test_invalid_version_and_node_references_return_clear_errors(storage):
    add_version(storage, "v1", [("node-a", "Safety", "Body")], "2026-10-23T10:00:00+00:00")

    with pytest.raises(HTTPException) as missing_version:
        create_selection(version_id="missing", node_ids=["node-a"])
    with pytest.raises(HTTPException) as missing_node:
        create_selection(node_ids=["not-in-v1"])
    assert missing_version.value.status_code == 404
    assert "missing" in missing_version.value.detail
    assert missing_node.value.status_code == 422
    assert missing_node.value.detail["invalid_node_ids"] == ["not-in-v1"]


def test_duplicate_or_blank_selection_inputs_are_rejected():
    with pytest.raises(ValidationError):
        api.SelectionCreate(name="Safety", version_id="v1", node_ids=["node-a", "node-a"])
    with pytest.raises(ValidationError):
        api.SelectionCreate(name="   ", version_id="v1", node_ids=["node-a"])


def test_unverified_version_cannot_be_selected(storage):
    add_version(storage, "legacy", [("node-a", "Safety", "Body")], "2026-10-23T10:00:00+00:00")
    with sqlite3.connect(storage / "versions.sqlite3") as connection:
        connection.execute(
            "UPDATE versions SET is_ct200_manual = 0 WHERE id = ?", ("legacy",)
        )

    with pytest.raises(HTTPException) as unverified:
        create_selection(version_id="legacy", node_ids=["node-a"])
    assert unverified.value.status_code == 422


def test_generation_provenance_survives_staleness_and_repeated_writes(storage):
    add_version(
        storage,
        "v1",
        [
            ("node-a", "Safety", "Original safety instructions."),
            ("node-b", "Setup", "Original setup instructions."),
            ("node-c", "Cleaning", "Clean with a dry cloth."),
        ],
        "2026-10-23T10:00:00+00:00",
    )
    selection = create_selection(node_ids=["node-a", "node-b", "node-c"])
    mock_cases = [{"case": "mocked stored result", "source_node_id": "node-a"}]
    first = create_generation(
        storage_dir=storage,
        selection=selection,
        cases=mock_cases,
        created_at="2026-10-23T10:01:00+00:00",
        provider="mock",
        model="deterministic-test",
    )
    second = create_generation(
        storage_dir=storage,
        selection=selection,
        cases=mock_cases,
        created_at="2026-10-23T10:02:00+00:00",
        provider="mock",
        model="deterministic-test",
    )
    assert first["id"] != second["id"]
    assert first["source_version_id"] == "v1"
    assert first["source_node_ids"] == ["node-a", "node-b", "node-c"]
    assert first["source_snapshot"] == selection["source_snapshot"]
    assert get_generation(storage, first["id"]) == first

    add_version(
        storage,
        "v2",
        [
            ("node-a", "Safety", "Updated safety instructions."),
            ("node-c", "Cleaning", "Clean with a dry cloth."),
        ],
        "2026-10-23T10:03:00+00:00",
    )
    payload = api.get_selection_staleness(selection["id"])
    assert payload["is_stale"] is True
    statuses = {item["source_node_id"]: item["status"] for item in payload["source_nodes"]}
    assert statuses == {"node-a": "changed", "node-b": "unmatched", "node-c": "unchanged"}
    assert payload["affected_generation_ids"] == [first["id"], second["id"]]

    version_impact = api.get_version_selection_impact("v2")
    assert version_impact["scanned_selection_count"] == 1
    assert version_impact["affected_selection_count"] == 1
    assert version_impact["selections"][0]["selection_id"] == selection["id"]
    assert version_impact["selections"][0]["affected_generation_ids"] == [
        first["id"],
        second["id"],
    ]

    fetched = api.get_saved_generation(first["id"])
    generation_stale = api.get_generation_staleness(first["id"])
    listed = api.get_selection_generations(selection["id"])
    assert fetched["cases"] == mock_cases
    assert fetched["source_snapshot"] == selection["source_snapshot"]
    assert generation_stale["is_stale"] is True
    assert listed["count"] == 2


def test_latest_version_staleness_is_current_for_source_version(storage):
    add_version(
        storage,
        "v1",
        [("node-a", "Safety", "Body")],
        "2026-10-23T10:00:00+00:00",
    )
    selection = create_selection(node_ids=["node-a"])
    response = api.get_selection_staleness(selection["id"])
    assert response["is_stale"] is False
    assert response["summary"] == {"unchanged": 1, "changed": 0, "unmatched": 0}


def test_staleness_cannot_be_checked_against_an_older_version(storage):
    add_version(storage, "v1", [("node-a", "Safety", "Body")], "2026-10-23T10:00:00+00:00")
    add_version(storage, "v2", [("node-a", "Safety", "Body v2")], "2026-10-23T10:01:00+00:00")
    selection = api.post_selection(
        api.SelectionCreate(name="Later selection", version_id="v2", node_ids=["node-a"])
    )

    with pytest.raises(HTTPException) as older_target:
        api.get_selection_staleness(selection["id"], against_version_id="v1")
    assert older_target.value.status_code == 400


def test_part3_routes_are_registered():
    paths = {route.path for route in api.app.routes}
    assert "/selections" in paths
    assert "/selections/{selection_id}" in paths
    assert "/selections/{selection_id}/staleness" in paths
    assert "/selections/{selection_id}/generations" in paths
    assert "/versions/{version_id}/selection-impact" in paths
    assert "/generations/{generation_id}" in paths
    assert "/generations/{generation_id}/staleness" in paths
    assert "/selections/{selection_id}/generate" in paths


def test_generation_uses_mocked_provider_and_persists_exact_provenance(storage, monkeypatch):
    selection = make_selection_with_source(storage)
    raw = json.dumps(generated_cases())
    monkeypatch.setenv("GROQ_MODEL", "test/custom-model")
    monkeypatch.setattr(
        generation,
        "request_groq_completion",
        lambda messages, model: (raw, model),
    )

    saved = api.post_selection_generation(selection["id"])

    assert saved["selection_id"] == selection["id"]
    assert saved["provider"] == "groq"
    assert saved["model"] == "test/custom-model"
    assert saved["cases"] == generated_cases()["cases"]
    assert saved["source_version_id"] == selection["version_id"]
    assert saved["source_node_ids"] == selection["node_ids"]
    assert saved["source_snapshot"] == selection["source_snapshot"]
    assert saved["source_hashes"] == selection["source_hashes"]
    assert saved["snapshot_hash"] == selection["snapshot_hash"]
    assert saved["response_hash"] == hashlib.sha256(raw.encode()).hexdigest()
    assert saved["completion_attempts"] == 1
    assert api.get_generation_staleness(saved["id"])["is_stale"] is False


def test_project_dotenv_loads_values_without_overriding_process_environment(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "GROQ_API_KEY=dotenv-test-value\nGROQ_MODEL=dotenv-test-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    assert generation.load_environment(dotenv_path) is True
    assert os.environ["GROQ_API_KEY"] == "dotenv-test-value"
    assert os.environ["GROQ_MODEL"] == "dotenv-test-model"

    monkeypatch.setenv("GROQ_API_KEY", "process-test-value")
    assert generation.load_environment(dotenv_path) is True
    assert os.environ["GROQ_API_KEY"] == "process-test-value"


def test_groq_request_uses_environment_key_in_authorization_header(monkeypatch):
    expected_key = "mock-only-groq-key"
    captured = {}

    class MockResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "model": "test/model",
                "choices": [{"message": {"content": '{"ok": true}'}}],
            }

    def mocked_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return MockResponse()

    monkeypatch.setenv("GROQ_API_KEY", expected_key)
    monkeypatch.setattr(generation.httpx, "post", mocked_post)

    content, model = generation.request_groq_completion(
        [{"role": "user", "content": "test"}], "test/model"
    )

    assert captured["headers"]["Authorization"] == f"Bearer {expected_key}"
    assert captured["json"]["model"] == "test/model"
    assert content == '{"ok": true}'
    assert model == "test/model"


def test_generation_schema_and_grounding_validation_reject_bad_output():
    cases = generated_cases()["cases"]
    with pytest.raises(ValidationError):
        generation.GenerationOutput.model_validate({"cases": cases[:2]})
    with pytest.raises(ValidationError):
        generation.GenerationOutput.model_validate(
            {"cases": [{**cases[0], "unexpected": True}, *cases[1:]]}
        )
    with pytest.raises(generation.ModelOutputInvalid, match="unselected nodes"):
        generation._parse_and_validate(
            json.dumps(
                {"cases": [{**case, "source_node_ids": ["not-selected"]} for case in cases]}
            ),
            [{"node_id": "node-a", "heading": "Pressure readings", "body": "systolic readings"}],
            ["node-a"],
        )
    with pytest.raises(generation.ModelOutputInvalid, match="numeric details"):
        generation._parse_and_validate(
            json.dumps(
                {"cases": [{**case, "steps": ["Wait 30 seconds"]} for case in cases]}
            ),
            [{"node_id": "node-a", "heading": "Pressure readings", "body": "systolic readings"}],
            ["node-a"],
        )


@pytest.mark.parametrize("unsafe_case_factory", [groq_submersion_case, groq_alcohol_case])
def test_explicitly_prohibited_groq_actions_are_rejected(unsafe_case_factory):
    safe_cases = safe_cleaning_cases()["cases"]
    cases = [unsafe_case_factory(), *safe_cases[:2]]

    with pytest.raises(generation.ModelOutputInvalid, match="explicitly prohibited"):
        generation._parse_and_validate(
            json.dumps({"cases": cases}),
            cleaning_source_snapshot(),
            ["cleaning-node"],
        )


@pytest.mark.parametrize(
    ("field", "value", "unsupported_term"),
    [
        ("steps", ["Ensure the device is powered off."], "powered"),
        ("expected_result", "The device remains operational.", "operational"),
    ],
)
def test_unsupported_conditions_and_outcomes_are_rejected(field, value, unsupported_term):
    cases = safe_cleaning_cases()["cases"]
    cases[0] = {**cases[0], field: value}

    with pytest.raises(generation.ModelOutputInvalid, match="not supported") as failure:
        generation._parse_and_validate(
            json.dumps({"cases": cases}),
            cleaning_source_snapshot(),
            ["cleaning-node"],
        )

    assert unsupported_term in str(failure.value)


def test_vague_steps_without_source_verifiable_content_are_rejected():
    cases = safe_cleaning_cases()["cases"]
    cases[0] = {**cases[0], "steps": ["Check."]}

    with pytest.raises(generation.ModelOutputInvalid, match="no verifiable source content"):
        generation._parse_and_validate(
            json.dumps({"cases": cases}),
            cleaning_source_snapshot(),
            ["cleaning-node"],
        )


def test_conservative_source_validation_accepts_directly_supported_cases():
    output = generation._parse_and_validate(
        json.dumps(safe_cleaning_cases()),
        cleaning_source_snapshot(),
        ["cleaning-node"],
    )

    assert len(output.cases) == 3


def test_case_cannot_negate_an_affirmative_source_instruction():
    cases = safe_cleaning_cases()["cases"]
    cases[0] = {
        **cases[0],
        "requirement": "Do not wipe the device body and cuff exterior with a soft, dry cloth.",
    }

    with pytest.raises(generation.ModelOutputInvalid, match="negates a required source action"):
        generation._parse_and_validate(
            json.dumps({"cases": cases}),
            cleaning_source_snapshot(),
            ["cleaning-node"],
        )


def test_case_cannot_negate_an_affirmative_source_result():
    cases = generated_cases()["cases"]
    cases[0] = {
        **cases[0],
        "requirement": "The monitor does not display a systolic reading.",
    }

    with pytest.raises(generation.ModelOutputInvalid, match="negates a required source action"):
        generation._parse_and_validate(
            json.dumps({"cases": cases}),
            [
                {
                    "node_id": "node-a",
                    "heading": "Pressure readings",
                    "body": "The monitor displays systolic and diastolic pressure readings.",
                }
            ],
            ["node-a"],
        )


def test_unsafe_mocked_completion_retries_once_and_reports_attempt_count():
    replies = [
        (json.dumps({"cases": [groq_submersion_case(), *safe_cleaning_cases()["cases"][:2]]}), "mock"),
        (json.dumps(safe_cleaning_cases()), "mock"),
    ]
    calls = []

    def mocked_provider(messages, model):
        calls.append(messages)
        return replies[len(calls) - 1]

    generated = generation.generate_qa_cases(
        cleaning_source_snapshot(),
        ["cleaning-node"],
        provider_call=mocked_provider,
    )

    assert len(calls) == 2
    assert "explicitly prohibited" in calls[1][-1]["content"]
    assert generated["completion_attempts"] == 2
    assert generated["cases"] == safe_cleaning_cases()["cases"]


def test_two_semantically_invalid_mocked_completions_fail_with_visible_attempt_count(
    storage, monkeypatch
):
    source = cleaning_source_snapshot()[0]
    add_version(
        storage,
        "v1",
        [("cleaning-node", source["heading"], source["body"])],
        "2026-10-23T10:00:00+00:00",
    )
    selection = create_selection(node_ids=["cleaning-node"])
    replies = [
        json.dumps({"cases": [groq_submersion_case(), *safe_cleaning_cases()["cases"][:2]]}),
        json.dumps({"cases": [groq_alcohol_case(), *safe_cleaning_cases()["cases"][:2]]}),
    ]
    calls = []

    def mocked_provider(messages, model):
        calls.append(messages)
        return replies[len(calls) - 1], model

    monkeypatch.setattr(generation, "request_groq_completion", mocked_provider)
    with pytest.raises(HTTPException) as failure:
        api.post_selection_generation(selection["id"])

    assert len(calls) == 2
    assert failure.value.status_code == 502
    assert "No cases were saved" in failure.value.detail
    assert "Completion attempts: 2." in failure.value.detail
    assert api.get_selection_generations(selection["id"])["count"] == 0


def test_invalid_first_response_is_corrected_once_and_then_saved(storage, monkeypatch):
    selection = make_selection_with_source(storage)
    replies = iter([("not JSON", "openai/gpt-oss-20b"), (json.dumps(generated_cases()), "openai/gpt-oss-20b")])
    calls = []

    def mocked_provider(messages, model):
        calls.append(messages)
        return next(replies)

    monkeypatch.setattr(generation, "request_groq_completion", mocked_provider)
    saved = api.post_selection_generation(selection["id"])

    assert len(calls) == 2
    assert "Correct your prior response" in calls[1][-1]["content"]
    assert saved["cases"] == generated_cases()["cases"]
    assert saved["completion_attempts"] == 2


def test_two_invalid_provider_responses_fail_without_saving(storage, monkeypatch):
    selection = make_selection_with_source(storage)
    replies = iter([("{}", "model"), ("still not JSON", "model")])
    monkeypatch.setattr(generation, "request_groq_completion", lambda messages, model: next(replies))

    with pytest.raises(HTTPException) as failure:
        api.post_selection_generation(selection["id"])

    assert failure.value.status_code == 502
    assert "No cases were saved" in failure.value.detail
    assert api.get_selection_generations(selection["id"])["count"] == 0


def test_missing_groq_credentials_and_provider_failure_are_clear(storage, monkeypatch):
    selection = make_selection_with_source(storage)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(HTTPException) as missing_credentials:
        api.post_selection_generation(selection["id"])
    assert missing_credentials.value.status_code == 503
    assert "GROQ_API_KEY" in missing_credentials.value.detail
    assert "Completion attempts: 0." in missing_credentials.value.detail
    assert api.get_selection_generations(selection["id"])["count"] == 0

    def timed_out_provider(messages, model):
        raise generation.GenerationFailure(504, "Groq generation timed out. Retry the request.")

    monkeypatch.setattr(generation, "request_groq_completion", timed_out_provider)
    with pytest.raises(HTTPException) as provider_error:
        api.post_selection_generation(selection["id"])
    assert provider_error.value.status_code == 504
    assert api.get_selection_generations(selection["id"])["count"] == 0


def test_repeated_generation_appends_and_staleness_keeps_original_cases(storage, monkeypatch):
    selection = make_selection_with_source(storage)
    raw = json.dumps(generated_cases())
    monkeypatch.setattr(generation, "request_groq_completion", lambda messages, model: (raw, model))

    first = api.post_selection_generation(selection["id"])
    second = api.post_selection_generation(selection["id"])
    assert first["id"] != second["id"]
    assert api.get_selection_generations(selection["id"])["count"] == 2

    add_version(
        storage,
        "v2",
        [("node-a", "Pressure readings", "The monitor displays updated systolic pressure readings.")],
        "2026-10-23T10:03:00+00:00",
    )
    stale = api.get_generation_staleness(first["id"])
    retrieved = api.get_saved_generation(first["id"])
    assert stale["is_stale"] is True
    assert stale["source_nodes"][0]["status"] == "changed"
    assert retrieved["staleness"]["is_stale"] is True
    assert retrieved["staleness"]["source_nodes"][0]["status"] == "changed"
    assert retrieved["cases"] == generated_cases()["cases"]
    assert retrieved["source_snapshot"] == selection["source_snapshot"]
