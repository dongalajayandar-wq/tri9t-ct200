"""API and comparison tests for Part 2 document versioning."""

from __future__ import annotations

import hashlib
import sqlite3

import fitz
import pytest
from fastapi.testclient import TestClient

import app as api
import parser as parser_module
from parser import OCRProcessingError, TreeNode, _TextLine
from version_store import compare_trees


@pytest.fixture
def client(tmp_path):
    previous_storage_dir = api.app.state.storage_dir
    api.app.state.storage_dir = tmp_path / "version_storage"
    with TestClient(api.app) as test_client:
        yield test_client
    api.app.state.storage_dir = previous_storage_dir


def make_pdf(sections: list[tuple[str, str]]) -> bytes:
    """Create a small text PDF that uses the parser's heading/font rules."""
    document = fitz.open()
    page = document.new_page()
    page.insert_text((50, 20), "CardioTrack CT-200 Home Blood Pressure Monitor", fontsize=8)
    page.insert_text((50, 34), "Device Overview and Intended Use", fontsize=8)
    y = 55
    for index, (heading, body) in enumerate(sections):
        # Keep each synthetic detail line on one visual line. Embedded newlines
        # in test input are whitespace variations, not intended page layout.
        body = " ".join(body.split())
        if y > 700:
            page = document.new_page()
            y = 55
        page.insert_text((50, y), heading, fontsize=20)
        for line_number in range(3):
            page.insert_text(
                (50, y + 24 + line_number * 14),
                f"{body} detail {line_number + 1}",
                fontsize=11,
            )
        y += 80
    content = document.tobytes()
    document.close()
    return content


def make_assignment_pdf() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((50, 55), "AI Engineering Internship Assignment - Tri9T AI", fontsize=18)
    page.insert_text((50, 100), "CardioTrack CT-200 Home Blood Pressure Monitor", fontsize=11)
    page.insert_text((50, 125), "Build a document tree API. Key Terms and Expected Tech Stack.", fontsize=11)
    content = document.tobytes()
    document.close()
    return content


def upload(client: TestClient, name: str, sections: list[tuple[str, str]]) -> dict:
    response = client.post(
        f"/versions/upload?filename={name}",
        content=make_pdf(sections),
        headers={"content-type": "application/pdf"},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_upload_stores_pdf_tree_and_metadata(client, tmp_path):
    content = make_pdf([("1 Overview", "The monitor measures pressure.")])
    response = client.post(
        "/versions/upload?filename=manual.pdf",
        content=content,
        headers={"content-type": "application/pdf"},
    )

    assert response.status_code == 201
    version = response.json()
    storage = tmp_path / "version_storage"
    assert version["version_number"] == 1
    assert version["content_hash"] == hashlib.sha256(content).hexdigest()
    assert (storage / "versions.sqlite3").exists()
    assert (storage / "uploads" / f"{version['id']}.pdf").read_bytes() == content
    assert (storage / "trees" / f"{version['id']}.json").exists()


def test_upload_reports_ocr_failure_without_saving_the_pdf(client, tmp_path, monkeypatch):
    def fail_ocr(pdf_path):
        raise OCRProcessingError("Tesseract OCR executable was not found; set TESSERACT_CMD.")

    monkeypatch.setattr(api, "parse_pdf", fail_ocr)
    response = client.post(
        "/versions/upload?filename=scanned.pdf",
        content=make_pdf([("1 Overview", "This is a CT-200 manual section.")]),
        headers={"content-type": "application/pdf"},
    )

    assert response.status_code == 503
    assert "TESSERACT_CMD" in response.json()["detail"]
    assert not list((tmp_path / "version_storage" / "uploads").glob("*.pdf"))


def test_scanned_manual_identity_is_checked_after_mocked_ocr(client, monkeypatch):
    document = fitz.open()
    page = document.new_page()
    page.draw_rect(fitz.Rect(30, 30, 100, 100))
    content = document.tobytes()
    document.close()
    ocr_lines = [
        _TextLine("CardioTrack CT-200 Home Blood Pressure Monitor", 18, 0, (40, 40, 300, 60), 0),
        _TextLine("1 Device Overview", 14, 0, (40, 90, 220, 108), 0),
        _TextLine("This is the overview section for blood pressure.", 12, 0, (40, 120, 270, 135), 0),
        _TextLine("1.1 Intended Use", 14, 0, (40, 155, 220, 173), 0),
        _TextLine("This manual describes intended use of the monitor.", 12, 0, (40, 185, 280, 200), 0),
    ]
    monkeypatch.setattr(parser_module, "_ocr_page_lines", lambda page, page_number: ocr_lines)

    response = client.post(
        "/versions/upload?filename=scanned-manual.pdf",
        content=content,
        headers={"content-type": "application/pdf"},
    )

    assert response.status_code == 201, response.text
    assert response.json()["is_ct200_manual"] is True
    assert response.json()["node_count"] >= 3


def test_list_versions_returns_all_versions(client):
    first = upload(client, "first.pdf", [("1 Overview", "First manual.")])
    second = upload(client, "second.pdf", [("1 Overview", "Second manual.")])

    response = client.get("/versions")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert [item["id"] for item in payload["versions"]] == [first["id"], second["id"]]
    assert payload["latest_version_id"] == second["id"]
    assert client.get("/versions/latest").json()["id"] == second["id"]


def test_get_version_metadata_and_tree_uses_requested_version(client):
    first = upload(client, "first.pdf", [("1 First Section", "Original details.")])
    second = upload(client, "second.pdf", [("1 Second Section", "Updated details.")])

    metadata = client.get(f"/versions/{first['id']}")
    tree = client.get(f"/versions/{first['id']}/tree")
    assert metadata.status_code == 200
    assert metadata.json()["id"] == first["id"]
    assert tree.status_code == 200
    assert tree.json()["version"]["id"] == first["id"]
    headings = [node["heading"] for node in tree.json()["nodes"]]
    assert "1 First Section" in headings
    assert "1 Second Section" not in headings
    assert second["id"] != first["id"]


def test_compare_detects_added_section(client):
    old = upload(client, "old.pdf", [("1 Overview", "Overview text.")])
    new = upload(
        client,
        "new.pdf",
        [("1 Overview", "Overview text."), ("2 Safety", "Keep the cuff dry.")],
    )

    response = client.get(
        "/versions/compare",
        params={"old_version_id": old["id"], "new_version_id": new["id"]},
    )
    assert response.status_code == 200
    assert response.json()["summary"] == {
        "added": 1,
        "removed": 0,
        "modified": 0,
        "unchanged": 1,
    }
    assert next(c for c in response.json()["changes"] if c["change_type"] == "added")["heading"] == "2 Safety"


def test_compare_detects_removed_section(client):
    old = upload(
        client,
        "old.pdf",
        [("1 Overview", "Overview text."), ("2 Legacy", "Old instructions.")],
    )
    new = upload(client, "new.pdf", [("1 Overview", "Overview text.")])

    response = client.get(
        "/versions/compare",
        params={"old_version_id": old["id"], "new_version_id": new["id"]},
    )
    assert response.status_code == 200
    assert response.json()["summary"]["removed"] == 1
    assert next(c for c in response.json()["changes"] if c["change_type"] == "removed")["heading"] == "2 Legacy"


def test_compare_detects_modified_section(client):
    old = upload(client, "old.pdf", [("1 Overview", "Original instructions.")])
    new = upload(client, "new.pdf", [("1 Overview", "Revised instructions.")])

    response = client.get(
        "/versions/compare",
        params={"old_version_id": old["id"], "new_version_id": new["id"]},
    )
    assert response.status_code == 200
    changed = next(c for c in response.json()["changes"] if c["change_type"] == "modified")
    assert changed["old_content_hash"] != changed["new_content_hash"]
    assert "Original instructions" in changed["old_preview"]
    assert "Revised instructions" in changed["new_preview"]


def get_node_id(client, version_id, heading):
    response = client.get(f"/versions/{version_id}/tree")
    assert response.status_code == 200
    return next(node["id"] for node in response.json()["nodes"] if node["heading"] == heading)


def test_node_compare_reports_unchanged_node(client):
    old = upload(client, "old.pdf", [("1 Overview", "Same instructions.")])
    new = upload(client, "new.pdf", [("1 Overview", "Same instructions.")])
    node_id = get_node_id(client, old["id"], "1 Overview")

    response = client.get(
        "/versions/compare/node",
        params={
            "old_version_id": old["id"],
            "new_version_id": new["id"],
            "node_id": node_id,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["node_id"] == node_id
    assert payload["old_version"]["id"] == old["id"]
    assert payload["new_version"]["id"] == new["id"]
    assert payload["change"]["change_type"] == "unchanged"
    assert payload["change"]["old_content_hash"] == payload["change"]["new_content_hash"]
    assert "old_preview" not in payload["change"]
    assert "new_preview" not in payload["change"]


def test_node_compare_reports_modified_node_with_previews(client):
    old = upload(client, "old.pdf", [("1 Overview", "Original instructions.")])
    new = upload(client, "new.pdf", [("1 Overview", "Revised instructions.")])
    node_id = get_node_id(client, old["id"], "1 Overview")

    response = client.get(
        "/versions/compare/node",
        params={
            "old_version_id": old["id"],
            "new_version_id": new["id"],
            "node_id": node_id,
        },
    )

    assert response.status_code == 200
    change = response.json()["change"]
    assert change["change_type"] == "modified"
    assert change["old_content_hash"] != change["new_content_hash"]
    assert "Original instructions" in change["old_preview"]
    assert "Revised instructions" in change["new_preview"]


def test_node_compare_reports_added_and_removed_nodes(client):
    old = upload(
        client,
        "old.pdf",
        [("1 Overview", "Same instructions."), ("2 Legacy", "Old instructions.")],
    )
    new = upload(
        client,
        "new.pdf",
        [("1 Overview", "Same instructions."), ("3 Safety", "New instructions.")],
    )

    for version, heading, expected_change in (
        (new, "3 Safety", "added"),
        (old, "2 Legacy", "removed"),
    ):
        node_id = get_node_id(client, version["id"], heading)
        response = client.get(
            "/versions/compare/node",
            params={
                "old_version_id": old["id"],
                "new_version_id": new["id"],
                "node_id": node_id,
            },
        )
        assert response.status_code == 200
        assert response.json()["change"]["change_type"] == expected_change


def test_node_compare_rejects_invalid_identifiers(client):
    old = upload(client, "old.pdf", [("1 Overview", "Manual instructions.")])
    new = upload(client, "new.pdf", [("1 Overview", "Manual instructions.")])
    params = {"old_version_id": old["id"], "new_version_id": new["id"]}

    missing_node = client.get("/versions/compare/node", params={**params, "node_id": "missing"})
    assert missing_node.status_code == 404
    assert "not found in either selected version" in missing_node.json()["detail"]

    missing_version = client.get(
        "/versions/compare/node",
        params={**params, "old_version_id": "missing-version", "node_id": "node"},
    )
    assert missing_version.status_code == 404
    assert "missing-version" in missing_version.json()["detail"]

    same_version = client.get(
        "/versions/compare/node",
        params={**params, "new_version_id": old["id"], "node_id": "node"},
    )
    assert same_version.status_code == 400

    blank_node = client.get("/versions/compare/node", params={**params, "node_id": " "})
    assert blank_node.status_code == 400


def test_compare_ignores_whitespace_only_changes(client):
    old = upload(client, "old.pdf", [("1 Overview", "Keep this text.")])
    new = upload(client, "new.pdf", [("1 Overview", "Keep   this   text.\n")])

    response = client.get(
        "/versions/compare",
        params={"old_version_id": old["id"], "new_version_id": new["id"]},
    )
    assert response.status_code == 200
    assert response.json()["summary"]["modified"] == 0
    assert response.json()["summary"]["unchanged"] == 1


def test_compare_returns_added_removed_modified_and_unchanged(client):
    old = upload(
        client,
        "old.pdf",
        [
            ("1 Intro", "Same text."),
            ("2 Setup", "Old setup."),
            ("3 Legacy", "Legacy text."),
        ],
    )
    new = upload(
        client,
        "new.pdf",
        [
            ("1 Intro", "Same text."),
            ("2 Setup", "New setup."),
            ("4 New Section", "New section text."),
        ],
    )

    response = client.get(
        "/versions/compare",
        params={"old_version_id": old["id"], "new_version_id": new["id"]},
    )
    assert response.status_code == 200
    assert response.json()["summary"] == {
        "added": 1,
        "removed": 1,
        "modified": 1,
        "unchanged": 1,
    }


def test_unknown_version_returns_404(client):
    assert client.get("/versions/missing-version").status_code == 404
    assert client.get("/versions/missing-version/tree").status_code == 404


def test_invalid_pdf_returns_readable_400_error(client):
    response = client.post(
        "/versions/upload?filename=bad.pdf",
        content=b"%PDF-1.7\nThis is not a complete PDF document.",
        headers={"content-type": "application/pdf"},
    )
    assert response.status_code == 400
    assert "valid PDF" in response.json()["detail"]


def test_assignment_pdf_is_rejected_as_a_manual(client, tmp_path):
    response = client.post(
        "/versions/upload?filename=assignment.pdf",
        content=make_assignment_pdf(),
        headers={"content-type": "application/pdf"},
    )

    assert response.status_code == 422
    assert "does not appear to be the CardioTrack CT-200 manual" in response.json()["detail"]
    assert client.get("/versions").json()["count"] == 0
    assert not list((tmp_path / "version_storage" / "uploads").glob("*.pdf"))


def test_legacy_unverified_version_is_not_listed_or_retrieved_as_a_tree(client, tmp_path):
    storage = tmp_path / "version_storage"
    storage.mkdir()
    with sqlite3.connect(storage / "versions.sqlite3") as connection:
        connection.execute(
            """CREATE TABLE versions (
                id TEXT PRIMARY KEY,
                version_number INTEGER NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                node_count INTEGER NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO versions VALUES (?, ?, ?, ?, ?, ?)",
            ("legacy-assignment", 1, "assignment.pdf", "2026-10-23T00:00:00+00:00", "hash", 45),
        )

    assert client.get("/versions").json()["count"] == 0
    metadata = client.get("/versions/legacy-assignment")
    assert metadata.status_code == 200
    assert metadata.json()["is_ct200_manual"] is False
    assert client.get("/versions/legacy-assignment/tree").status_code == 422


def test_same_version_cannot_be_compared(client):
    version = upload(client, "manual.pdf", [("1 Overview", "Manual text.")])
    response = client.get(
        "/versions/compare",
        params={"old_version_id": version["id"], "new_version_id": version["id"]},
    )
    assert response.status_code == 400


def test_compare_falls_back_to_normalized_heading_path_when_ids_change():
    old_tree = [
        TreeNode(id="old-root", heading="ROOT", level=0),
        TreeNode(
            id="old-section", heading="1 Overview", level=1,
            body="Original body", parent_id="old-root",
        ),
    ]
    new_tree = [
        TreeNode(id="new-root", heading="ROOT", level=0),
        TreeNode(
            id="new-section", heading="1   Overview", level=1,
            body="Updated body", parent_id="new-root",
        ),
    ]

    result = compare_trees(old_tree, new_tree)
    assert result["summary"]["modified"] == 1
    changed = result["changes"][0]
    assert changed["old_section_id"] == "old-section"
    assert changed["new_section_id"] == "new-section"
    assert changed["change_type"] == "modified"
