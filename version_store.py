"""Small SQLite-backed metadata store for uploaded manual versions."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any

from parser import TreeNode


def version_file_paths(storage_dir: str | Path, version_id: str) -> tuple[Path, Path]:
    """Return the PDF and parsed-tree paths for a generated version ID."""
    root = Path(storage_dir)
    return root / "uploads" / f"{version_id}.pdf", root / "trees" / f"{version_id}.json"


def _connect(storage_dir: str | Path) -> sqlite3.Connection:
    root = Path(storage_dir)
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root / "versions.sqlite3", timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS versions (
            id TEXT PRIMARY KEY,
            version_number INTEGER NOT NULL UNIQUE,
            filename TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            node_count INTEGER NOT NULL,
            is_ct200_manual INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(versions)").fetchall()
    }
    if "is_ct200_manual" not in columns:
        # Existing records are unverified until they are checked; in particular,
        # this prevents a previously uploaded assignment PDF from appearing as a
        # valid CT-200 manual version after the migration.
        connection.execute(
            "ALTER TABLE versions ADD COLUMN is_ct200_manual INTEGER NOT NULL DEFAULT 0"
        )
    connection.commit()
    return connection


def _metadata(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "version_number": row["version_number"],
        "filename": row["filename"],
        "uploaded_at": row["uploaded_at"],
        "content_hash": row["content_hash"],
        "node_count": row["node_count"],
        "is_ct200_manual": bool(row["is_ct200_manual"]),
    }


def create_version(
    *,
    storage_dir: str | Path,
    version_id: str,
    filename: str,
    uploaded_at: str,
    content_hash: str,
    node_count: int,
) -> dict[str, Any]:
    """Insert version metadata and assign the next sequential version number."""
    connection = _connect(storage_dir)
    try:
        connection.execute("BEGIN IMMEDIATE")
        next_number = connection.execute(
            "SELECT COALESCE(MAX(version_number), 0) + 1 FROM versions"
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO versions
                (id, version_number, filename, uploaded_at, content_hash, node_count, is_ct200_manual)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            """,
            (version_id, next_number, filename, uploaded_at, content_hash, node_count),
        )
        connection.commit()
        row = connection.execute("SELECT * FROM versions WHERE id = ?", (version_id,)).fetchone()
        return _metadata(row)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def list_versions(storage_dir: str | Path) -> list[dict[str, Any]]:
    connection = _connect(storage_dir)
    try:
        rows = connection.execute(
            "SELECT * FROM versions WHERE is_ct200_manual = 1 ORDER BY version_number ASC"
        ).fetchall()
        return [_metadata(row) for row in rows]
    finally:
        connection.close()


def get_version(storage_dir: str | Path, version_id: str) -> dict[str, Any] | None:
    connection = _connect(storage_dir)
    try:
        row = connection.execute(
            "SELECT * FROM versions WHERE id = ?", (version_id,)
        ).fetchone()
        return _metadata(row) if row else None
    finally:
        connection.close()


def _normalize_body(text: str) -> str:
    """Normalize Unicode and whitespace while preserving meaningful case."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _normalize_heading(text: str) -> str:
    return _normalize_body(text).casefold()


def _section_paths(nodes: list[TreeNode]) -> dict[str, str]:
    """Build normalized heading paths, adding sibling ordinals for duplicates."""
    by_id = {node.id: node for node in nodes}
    occurrences: dict[tuple[str | None, str], int] = {}
    components: dict[str, str] = {}
    for node in nodes:
        if node.level == 0:
            continue
        heading = _normalize_heading(node.heading)
        occurrence_key = (node.parent_id, heading)
        ordinal = occurrences.get(occurrence_key, 0) + 1
        occurrences[occurrence_key] = ordinal
        components[node.id] = f"{heading} [{ordinal}]"

    paths: dict[str, str] = {}

    def path_for(node_id: str, visiting: set[str] | None = None) -> str:
        if node_id in paths:
            return paths[node_id]
        node = by_id[node_id]
        if node.level == 0:
            paths[node_id] = ""
            return ""
        visiting = visiting or set()
        if node_id in visiting:
            return components.get(node_id, _normalize_heading(node.heading))
        visiting.add(node_id)
        parent_path = ""
        if node.parent_id in by_id and by_id[node.parent_id].level != 0:
            parent_path = path_for(node.parent_id, visiting)
        current = components.get(node_id, _normalize_heading(node.heading))
        paths[node_id] = f"{parent_path} > {current}" if parent_path else current
        visiting.remove(node_id)
        return paths[node_id]

    for node in nodes:
        if node.level != 0:
            path_for(node.id)
    return paths


def _comparison_hash(text: str) -> str:
    return hashlib.sha256(_normalize_body(text).encode("utf-8")).hexdigest()


def _preview(text: str, limit: int = 300) -> str:
    normalized = _normalize_body(text)
    return normalized if len(normalized) <= limit else normalized[:limit].rstrip() + "…"


def compare_trees(old_nodes: list[TreeNode], new_nodes: list[TreeNode]) -> dict[str, Any]:
    """Compare sections using parser IDs first, then normalized heading paths."""
    old_by_id = {node.id: node for node in old_nodes if node.level != 0}
    new_by_id = {node.id: node for node in new_nodes if node.level != 0}
    old_paths = _section_paths(old_nodes)
    new_paths = _section_paths(new_nodes)
    new_id_by_path = {
        new_paths[node_id]: node_id
        for node_id in new_by_id
        if node_id in new_paths
    }

    matched_old: set[str] = set()
    matched_new: set[str] = set()
    pairs: list[tuple[TreeNode | None, TreeNode | None, str]] = []

    # Parser-generated IDs are deterministic when a section's heading path stays put.
    for old_node in old_nodes:
        if old_node.level == 0:
            continue
        new_node = new_by_id.get(old_node.id)
        if new_node is not None:
            matched_old.add(old_node.id)
            matched_new.add(new_node.id)
            pairs.append((old_node, new_node, new_paths.get(new_node.id, old_paths.get(old_node.id, ""))))

    # If IDs changed, match remaining sections by normalized heading path.
    for old_node in old_nodes:
        if old_node.level == 0 or old_node.id in matched_old:
            continue
        key = old_paths.get(old_node.id, "")
        candidate_id = new_id_by_path.get(key)
        if candidate_id and candidate_id not in matched_new:
            matched_old.add(old_node.id)
            matched_new.add(candidate_id)
            pairs.append((old_node, new_by_id[candidate_id], key))
        else:
            pairs.append((old_node, None, key))

    for new_node in new_nodes:
        if new_node.level != 0 and new_node.id not in matched_new:
            pairs.append((None, new_node, new_paths.get(new_node.id, "")))

    changes: list[dict[str, Any]] = []
    counts = {"added": 0, "removed": 0, "modified": 0, "unchanged": 0}
    for old_node, new_node, matching_key in pairs:
        if old_node is None:
            change_type = "added"
        elif new_node is None:
            change_type = "removed"
        elif _normalize_body(old_node.body) != _normalize_body(new_node.body):
            change_type = "modified"
        else:
            change_type = "unchanged"
        counts[change_type] += 1

        item: dict[str, Any] = {
            "section_id": new_node.id if new_node else old_node.id,
            "old_section_id": old_node.id if old_node else None,
            "new_section_id": new_node.id if new_node else None,
            "matching_key": matching_key,
            "heading": new_node.heading if new_node else old_node.heading,
            "old_heading": old_node.heading if old_node else None,
            "new_heading": new_node.heading if new_node else None,
            "change_type": change_type,
            "old_content_hash": _comparison_hash(old_node.body) if old_node else None,
            "new_content_hash": _comparison_hash(new_node.body) if new_node else None,
        }
        if change_type != "unchanged":
            item["old_preview"] = _preview(old_node.body) if old_node else None
            item["new_preview"] = _preview(new_node.body) if new_node else None
        changes.append(item)

    return {"summary": counts, "changes": changes}
