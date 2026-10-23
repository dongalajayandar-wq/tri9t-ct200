"""SQLite persistence for immutable source selections and QA generations."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from version_store import _connect as _connect_versions


def _connect(storage_dir: str | Path) -> sqlite3.Connection:
    """Open the existing version database and add Part 3 tables if needed.

    The migration is additive: it never rebuilds or edits existing version rows.
    """
    connection = _connect_versions(storage_dir)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS selections (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            version_id TEXT NOT NULL REFERENCES versions(id) ON DELETE RESTRICT,
            created_at TEXT NOT NULL,
            node_ids_json TEXT NOT NULL,
            source_snapshot_json TEXT NOT NULL,
            source_hashes_json TEXT NOT NULL,
            snapshot_hash TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS selections_by_version
            ON selections(version_id);

        CREATE TABLE IF NOT EXISTS generations (
            id TEXT PRIMARY KEY,
            selection_id TEXT NOT NULL REFERENCES selections(id) ON DELETE RESTRICT,
            created_at TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            source_version_id TEXT NOT NULL REFERENCES versions(id) ON DELETE RESTRICT,
            source_node_ids_json TEXT NOT NULL,
            source_snapshot_json TEXT NOT NULL,
            source_hashes_json TEXT NOT NULL,
            snapshot_hash TEXT NOT NULL,
            cases_json TEXT NOT NULL,
            response_hash TEXT
        );
        CREATE INDEX IF NOT EXISTS generations_by_selection
            ON generations(selection_id, created_at, id);
        """
    )
    connection.commit()
    return connection


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _selection(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "version_id": row["version_id"],
        "created_at": row["created_at"],
        "node_ids": json.loads(row["node_ids_json"]),
        "source_snapshot": json.loads(row["source_snapshot_json"]),
        "source_hashes": json.loads(row["source_hashes_json"]),
        "snapshot_hash": row["snapshot_hash"],
    }


def create_selection(
    *,
    storage_dir: str | Path,
    name: str,
    version_id: str,
    node_ids: list[str],
    source_snapshot: list[dict[str, Any]],
    source_hashes: dict[str, str],
    created_at: str,
) -> dict[str, Any]:
    """Insert a new immutable selection. Names are descriptive, not unique."""
    selection_id = uuid.uuid4().hex
    node_ids_json = _json(node_ids)
    snapshot_json = _json(source_snapshot)
    hashes_json = _json(source_hashes)
    snapshot_hash = hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest()
    connection = _connect(storage_dir)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """INSERT INTO selections
               (id, name, version_id, created_at, node_ids_json,
                source_snapshot_json, source_hashes_json, snapshot_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                selection_id,
                name,
                version_id,
                created_at,
                node_ids_json,
                snapshot_json,
                hashes_json,
                snapshot_hash,
            ),
        )
        connection.commit()
        row = connection.execute(
            "SELECT * FROM selections WHERE id = ?", (selection_id,)
        ).fetchone()
        return _selection(row)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def get_selection(storage_dir: str | Path, selection_id: str) -> dict[str, Any] | None:
    connection = _connect(storage_dir)
    try:
        row = connection.execute(
            "SELECT * FROM selections WHERE id = ?", (selection_id,)
        ).fetchone()
        return _selection(row) if row else None
    finally:
        connection.close()


def list_selections(storage_dir: str | Path) -> list[dict[str, Any]]:
    connection = _connect(storage_dir)
    try:
        rows = connection.execute(
            "SELECT * FROM selections ORDER BY created_at, id"
        ).fetchall()
        return [_selection(row) for row in rows]
    finally:
        connection.close()


def _generation(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "selection_id": row["selection_id"],
        "created_at": row["created_at"],
        "provider": row["provider"],
        "model": row["model"],
        "source_version_id": row["source_version_id"],
        "source_node_ids": json.loads(row["source_node_ids_json"]),
        "source_snapshot": json.loads(row["source_snapshot_json"]),
        "source_hashes": json.loads(row["source_hashes_json"]),
        "snapshot_hash": row["snapshot_hash"],
        "cases": json.loads(row["cases_json"]),
        "response_hash": row["response_hash"],
    }


def create_generation(
    *,
    storage_dir: str | Path,
    selection: dict[str, Any],
    cases: list[dict[str, Any]],
    created_at: str,
    provider: str | None = None,
    model: str | None = None,
    response_hash: str | None = None,
) -> dict[str, Any]:
    """Append one immutable result; repeated calls create distinct records."""
    generation_id = uuid.uuid4().hex
    source_snapshot_json = _json(selection["source_snapshot"])
    source_hashes_json = _json(selection["source_hashes"])
    cases_json = _json(cases)
    connection = _connect(storage_dir)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """INSERT INTO generations
               (id, selection_id, created_at, provider, model, source_version_id,
                source_node_ids_json, source_snapshot_json, source_hashes_json,
                snapshot_hash, cases_json, response_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                generation_id,
                selection["id"],
                created_at,
                provider,
                model,
                selection["version_id"],
                _json(selection["node_ids"]),
                source_snapshot_json,
                source_hashes_json,
                selection["snapshot_hash"],
                cases_json,
                response_hash,
            ),
        )
        connection.commit()
        row = connection.execute(
            "SELECT * FROM generations WHERE id = ?", (generation_id,)
        ).fetchone()
        return _generation(row)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def get_generation(storage_dir: str | Path, generation_id: str) -> dict[str, Any] | None:
    connection = _connect(storage_dir)
    try:
        row = connection.execute(
            "SELECT * FROM generations WHERE id = ?", (generation_id,)
        ).fetchone()
        return _generation(row) if row else None
    finally:
        connection.close()


def list_generations(storage_dir: str | Path, selection_id: str) -> list[dict[str, Any]]:
    connection = _connect(storage_dir)
    try:
        rows = connection.execute(
            "SELECT * FROM generations WHERE selection_id = ? ORDER BY created_at, id",
            (selection_id,),
        ).fetchall()
        return [_generation(row) for row in rows]
    finally:
        connection.close()
