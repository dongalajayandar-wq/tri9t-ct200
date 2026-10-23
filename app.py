"""FastAPI API for browsing and versioning the CT-200 manual."""

import hashlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import fitz
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from parser import OCRProcessingError, TreeNode, is_ct200_manual, load_tree, parse_pdf, save_tree, tree_to_dict
from version_store import (
    _section_paths,
    compare_trees,
    create_version,
    get_version,
    list_versions,
    version_file_paths,
)
from selection_store import (
    create_generation,
    create_selection,
    get_generation,
    get_selection,
    list_selections,
    list_generations,
)
from generation import GenerationFailure, generate_qa_cases

app = FastAPI(
    title="CT-200 Document Tree API",
    version="0.2.0",
    description="Browse CT-200 manual versions, save source selections, and generate provenance-aware QA cases.",
)

PROJECT_DIR = Path(__file__).resolve().parent
TREE_PATH = PROJECT_DIR / "output" / "ct200_tree.json"
DEFAULT_STORAGE_DIR = Path(
    os.environ.get("CT200_STORAGE_DIR", PROJECT_DIR / "storage")
).expanduser()
app.state.storage_dir = DEFAULT_STORAGE_DIR


class SelectionCreate(BaseModel):
    """Validated request body for a version-pinned named selection."""

    name: str = Field(min_length=1, max_length=120)
    version_id: str = Field(min_length=1, max_length=128)
    node_ids: list[str] = Field(min_length=1, max_length=200)

    @field_validator("name", "version_id")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("node_ids")
    @classmethod
    def validate_node_ids(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("node IDs must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("node IDs must be unique")
        return normalized


def _storage_dir() -> Path:
    """Read the configurable storage root from app state (also useful in tests)."""
    return Path(app.state.storage_dir)


def _get_tree() -> list[TreeNode]:
    if not TREE_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Tree not built yet. Run: python parser.py data/ct200_manual.pdf"
        )
    return load_tree(TREE_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tree")
def get_tree():
    """Return the full document tree as a list of nodes."""
    nodes = _get_tree()
    return {
        "total_nodes": len(nodes),
        "nodes": [
            {
                "id": n.id,
                "heading": n.heading,
                "level": n.level,
                "parent_id": n.parent_id,
                "children_count": len(n.children),
                "content_hash": n.content_hash,
                "body_preview": n.body[:200] + "…" if len(n.body) > 200 else n.body,
            }
            for n in nodes
        ],
    }


@app.get("/tree/{node_id}")
def get_node(node_id: str):
    """Return a single node by ID, including its full body text and children."""
    nodes = _get_tree()
    idx = {n.id: n for n in nodes}

    if node_id not in idx:
        raise HTTPException(status_code=404, detail=f"Node {node_id!r} not found")

    node = idx[node_id]
    children_detail = []
    for cid in node.children:
        if cid in idx:
            c = idx[cid]
            children_detail.append({
                "id": c.id,
                "heading": c.heading,
                "level": c.level,
            })

    return {
        "id": node.id,
        "heading": node.heading,
        "level": node.level,
        "body": node.body,
        "content_hash": node.content_hash,
        "parent_id": node.parent_id,
        "children": children_detail,
    }


@app.get("/search")
def search_nodes(q: str = ""):
    """Search node headings and body text (case-insensitive substring match)."""
    if not q:
        raise HTTPException(status_code=400, detail="Provide ?q=<search term>")

    nodes = _get_tree()
    q_lower = q.lower()
    results = []
    for n in nodes:
        if q_lower in n.heading.lower() or q_lower in n.body.lower():
            results.append({
                "id": n.id,
                "heading": n.heading,
                "level": n.level,
                "parent_id": n.parent_id,
                "match_in": "heading" if q_lower in n.heading.lower() else "body",
            })

    return {"query": q, "count": len(results), "results": results}


@app.post(
    "/versions/upload",
    status_code=201,
    tags=["versions"],
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/pdf": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        }
    },
)
async def upload_version(
    file: bytes = Body(..., media_type="application/pdf"),
    filename: str = "manual.pdf",
):
    """Parse and permanently store one raw application/pdf request body."""
    filename = Path(filename).name
    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Upload a file with a .pdf extension.")

    content = file
    if not content or b"%PDF-" not in content[:1024]:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            if document.page_count == 0:
                raise HTTPException(status_code=400, detail="The PDF contains no pages.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF. Check that it is readable, unencrypted, and undamaged.",
        ) from exc

    storage_dir = _storage_dir()
    version_id = uuid.uuid4().hex
    pdf_path, tree_path = version_file_paths(storage_dir, version_id)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(content)

    try:
        nodes = parse_pdf(pdf_path)
    except OCRProcessingError as exc:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF could not be converted into document text.",
        ) from exc

    parsed_text = "\n".join(
        f"{node.heading}\n{node.body}" for node in nodes
    )
    if not is_ct200_manual(parsed_text):
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=(
                "This PDF does not appear to be the CardioTrack CT-200 manual. "
                "Expected product details plus the Device Overview and Intended Use sections."
            ),
        )

    try:
        save_tree(nodes, tree_path)
        return create_version(
            storage_dir=storage_dir,
            version_id=version_id,
            filename=filename,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            content_hash=hashlib.sha256(content).hexdigest(),
            node_count=len(nodes),
        )
    except Exception:
        pdf_path.unlink(missing_ok=True)
        tree_path.unlink(missing_ok=True)
        raise


@app.get("/versions", tags=["versions"])
def get_versions():
    """List stored manual versions in ascending version-number order."""
    versions = list_versions(_storage_dir())
    return {
        "count": len(versions),
        "latest_version_id": versions[-1]["id"] if versions else None,
        "versions": versions,
    }


@app.get("/versions/latest", tags=["versions"])
def get_latest_version():
    versions = list_versions(_storage_dir())
    if not versions:
        raise HTTPException(status_code=404, detail="No manual versions have been uploaded.")
    return versions[-1]


@app.get("/versions/compare", tags=["versions"])
def compare_versions(old_version_id: str, new_version_id: str):
    """Compare two versions by section IDs, with normalized heading paths as fallback."""
    if old_version_id == new_version_id:
        raise HTTPException(status_code=400, detail="Choose two different version IDs to compare.")

    old_version = get_version(_storage_dir(), old_version_id)
    new_version = get_version(_storage_dir(), new_version_id)
    if old_version is None:
        raise HTTPException(status_code=404, detail=f"Version {old_version_id!r} not found.")
    if new_version is None:
        raise HTTPException(status_code=404, detail=f"Version {new_version_id!r} not found.")
    if not old_version["is_ct200_manual"] or not new_version["is_ct200_manual"]:
        raise HTTPException(
            status_code=422,
            detail="Both selected versions must be verified CT-200 manuals before comparison.",
        )

    _, old_tree_path = version_file_paths(_storage_dir(), old_version_id)
    _, new_tree_path = version_file_paths(_storage_dir(), new_version_id)
    try:
        old_tree = load_tree(old_tree_path)
        new_tree = load_tree(new_tree_path)
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=500, detail="A stored version tree could not be read.") from exc

    return {
        "old_version": old_version,
        "new_version": new_version,
        **compare_trees(old_tree, new_tree),
    }


@app.get("/versions/compare/node", tags=["versions"])
def compare_version_node(old_version_id: str, new_version_id: str, node_id: str):
    """Compare one node, identified in either version, using the section matching rules."""
    if not node_id.strip():
        raise HTTPException(status_code=400, detail="Choose a non-empty node ID to compare.")

    comparison = compare_versions(old_version_id, new_version_id)
    change = next(
        (
            item
            for item in comparison["changes"]
            if node_id in (item["old_section_id"], item["new_section_id"])
        ),
        None,
    )
    if change is None:
        raise HTTPException(
            status_code=404,
            detail=f"Node {node_id!r} not found in either selected version.",
        )

    return {
        "old_version": comparison["old_version"],
        "new_version": comparison["new_version"],
        "node_id": node_id,
        "change": change,
    }


@app.get("/versions/{version_id}", tags=["versions"])
def get_version_metadata(version_id: str):
    version = get_version(_storage_dir(), version_id)
    if version is None:
        raise HTTPException(status_code=404, detail=f"Version {version_id!r} not found.")
    return version


@app.get("/versions/{version_id}/tree", tags=["versions"])
def get_version_tree(version_id: str):
    version = get_version(_storage_dir(), version_id)
    if version is None:
        raise HTTPException(status_code=404, detail=f"Version {version_id!r} not found.")
    if not version["is_ct200_manual"]:
        raise HTTPException(
            status_code=422,
            detail="This stored PDF is not verified as a CT-200 manual and has no accepted document tree.",
        )
    _, tree_path = version_file_paths(_storage_dir(), version_id)
    try:
        nodes = load_tree(tree_path)
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=500, detail="The stored version tree could not be read.") from exc
    return {"version": version, "total_nodes": len(nodes), "nodes": tree_to_dict(nodes)}


def _load_verified_version_tree(version_id: str) -> tuple[dict, list[TreeNode]]:
    version = get_version(_storage_dir(), version_id)
    if version is None:
        raise HTTPException(status_code=404, detail=f"Version {version_id!r} not found.")
    if not version["is_ct200_manual"]:
        raise HTTPException(
            status_code=422,
            detail="Selections and staleness checks require a verified CT-200 manual version.",
        )
    _, tree_path = version_file_paths(_storage_dir(), version_id)
    try:
        nodes = load_tree(tree_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=500, detail="The stored version tree could not be read.") from exc
    return version, nodes


def _snapshot_impact(
    source_snapshot: list[dict], target_nodes: list[TreeNode]
) -> dict:
    """Match saved nodes by stable ID, then exact normalized heading path."""
    by_id = {node.id: node for node in target_nodes}
    paths = _section_paths(target_nodes)
    ids_by_path: dict[str, list[str]] = {}
    for node_id, path in paths.items():
        ids_by_path.setdefault(path, []).append(node_id)

    used_target_ids: set[str] = set()
    results: list[dict] = []
    for source in source_snapshot:
        source_id = source["node_id"]
        target = by_id.get(source_id)
        match_method = "id" if target is not None and target.id not in used_target_ids else None
        if match_method is None:
            target = None
            path = source.get("matching_key", "")
            candidates = [
                candidate_id
                for candidate_id in ids_by_path.get(path, [])
                if candidate_id not in used_target_ids
            ]
            if len(candidates) == 1:
                target = by_id[candidates[0]]
                match_method = "heading_path"

        if target is None:
            results.append(
                {
                    "source_node_id": source_id,
                    "source_heading": source["heading"],
                    "source_hash": source["content_hash"],
                    "target_node_id": None,
                    "target_heading": None,
                    "target_hash": None,
                    "status": "unmatched",
                    "match_method": None,
                    "detail": "No unique ID or normalized heading-path match exists in the checked version.",
                }
            )
            continue

        used_target_ids.add(target.id)
        target_hash = hashlib.sha256(target.body.encode("utf-8")).hexdigest()
        status = "unchanged" if target_hash == source["content_hash"] else "changed"
        results.append(
            {
                "source_node_id": source_id,
                "source_heading": source["heading"],
                "source_hash": source["content_hash"],
                "target_node_id": target.id,
                "target_heading": target.heading,
                "target_hash": target_hash,
                "status": status,
                "match_method": match_method,
                "detail": None,
            }
        )

    counts = {
        "unchanged": sum(item["status"] == "unchanged" for item in results),
        "changed": sum(item["status"] == "changed" for item in results),
        "unmatched": sum(item["status"] == "unmatched" for item in results),
    }
    return {
        "is_stale": counts["changed"] > 0 or counts["unmatched"] > 0,
        "summary": counts,
        "source_nodes": results,
    }


def _staleness_target(source_version_id: str, against_version_id: str | None) -> tuple[dict, list[TreeNode]]:
    source_version = get_version(_storage_dir(), source_version_id)
    if source_version is None:
        raise HTTPException(status_code=404, detail=f"Source version {source_version_id!r} not found.")
    if against_version_id is None:
        versions = list_versions(_storage_dir())
        if not versions:
            raise HTTPException(status_code=404, detail="No verified manual versions are available.")
        target_version_id = versions[-1]["id"]
    else:
        target_version_id = against_version_id

    target_version, target_nodes = _load_verified_version_tree(target_version_id)
    if target_version["version_number"] < source_version["version_number"]:
        raise HTTPException(
            status_code=400,
            detail="Staleness can only be checked against the source version or a newer version.",
        )
    return target_version, target_nodes


@app.get("/versions/{version_id}/selection-impact", tags=["selections"])
def get_version_selection_impact(version_id: str):
    """List saved selections and generations affected by this manual version."""
    target_version, target_nodes = _load_verified_version_tree(version_id)
    affected_selections = []
    scanned_selection_count = 0
    for selection in list_selections(_storage_dir()):
        source_version = get_version(_storage_dir(), selection["version_id"])
        if source_version is None or source_version["version_number"] > target_version["version_number"]:
            continue
        scanned_selection_count += 1
        impact = _snapshot_impact(selection["source_snapshot"], target_nodes)
        if not impact["is_stale"]:
            continue
        affected_generation_ids = []
        for generation in list_generations(_storage_dir(), selection["id"]):
            generation_impact = _snapshot_impact(generation["source_snapshot"], target_nodes)
            if generation_impact["is_stale"]:
                affected_generation_ids.append(generation["id"])
        affected_selections.append(
            {
                "selection_id": selection["id"],
                "name": selection["name"],
                "source_version_id": selection["version_id"],
                "is_stale": True,
                "summary": impact["summary"],
                "source_nodes": impact["source_nodes"],
                "affected_generation_ids": affected_generation_ids,
            }
        )
    return {
        "checked_against_version": target_version,
        "scanned_selection_count": scanned_selection_count,
        "affected_selection_count": len(affected_selections),
        "selections": affected_selections,
    }


@app.post("/selections", status_code=201, tags=["selections"])
def post_selection(payload: SelectionCreate):
    """Save a named selection and exact source snapshot from one manual version."""
    _, nodes = _load_verified_version_tree(payload.version_id)
    by_id = {node.id: node for node in nodes}
    missing = [node_id for node_id in payload.node_ids if node_id not in by_id]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Every selected node must belong to the specified version.",
                "invalid_node_ids": missing,
            },
        )

    paths = _section_paths(nodes)
    snapshots: list[dict] = []
    hashes: dict[str, str] = {}
    for node_id in payload.node_ids:
        node = by_id[node_id]
        content_hash = hashlib.sha256(node.body.encode("utf-8")).hexdigest()
        hashes[node_id] = content_hash
        snapshots.append(
            {
                "node_id": node.id,
                "heading": node.heading,
                "level": node.level,
                "parent_id": node.parent_id,
                "body": node.body,
                "content_hash": content_hash,
                "matching_key": paths.get(node.id, ""),
            }
        )
    try:
        return create_selection(
            storage_dir=_storage_dir(),
            name=payload.name,
            version_id=payload.version_id,
            node_ids=payload.node_ids,
            source_snapshot=snapshots,
            source_hashes=hashes,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The selection could not be saved.") from exc


@app.get("/selections/{selection_id}", tags=["selections"])
def get_saved_selection(selection_id: str):
    selection = get_selection(_storage_dir(), selection_id)
    if selection is None:
        raise HTTPException(status_code=404, detail=f"Selection {selection_id!r} not found.")
    return selection


@app.post("/selections/{selection_id}/generate", status_code=201, tags=["generations"])
def post_selection_generation(selection_id: str):
    """Generate and persist validated QA cases from a saved source snapshot."""
    selection = get_selection(_storage_dir(), selection_id)
    if selection is None:
        raise HTTPException(status_code=404, detail=f"Selection {selection_id!r} not found.")

    try:
        generated = generate_qa_cases(
            source_snapshot=selection["source_snapshot"],
            selected_node_ids=selection["node_ids"],
        )
    except GenerationFailure as exc:
        detail = exc.detail
        if exc.completion_attempts is not None:
            detail += f" Completion attempts: {exc.completion_attempts}."
        raise HTTPException(status_code=exc.status_code, detail=detail) from exc

    try:
        saved = create_generation(
            storage_dir=_storage_dir(),
            selection=selection,
            cases=generated["cases"],
            created_at=datetime.now(timezone.utc).isoformat(),
            provider=generated["provider"],
            model=generated["model"],
            response_hash=generated["response_hash"],
        )
        # Expose the exact provider-call count in the create response without
        # changing the existing SQLite schema or historical records.
        saved["completion_attempts"] = generated["completion_attempts"]
        return saved
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "The validated generation could not be saved. "
                f"Completion attempts: {generated['completion_attempts']}."
            ),
        ) from exc


@app.get("/selections/{selection_id}/generations", tags=["generations"])
def get_selection_generations(selection_id: str):
    selection = get_selection(_storage_dir(), selection_id)
    if selection is None:
        raise HTTPException(status_code=404, detail=f"Selection {selection_id!r} not found.")
    generations = list_generations(_storage_dir(), selection_id)
    return {"selection_id": selection_id, "count": len(generations), "generations": generations}


@app.get("/selections/{selection_id}/staleness", tags=["selections"])
def get_selection_staleness(selection_id: str, against_version_id: str | None = None):
    selection = get_selection(_storage_dir(), selection_id)
    if selection is None:
        raise HTTPException(status_code=404, detail=f"Selection {selection_id!r} not found.")
    target_version, target_nodes = _staleness_target(
        selection["version_id"], against_version_id
    )
    impact = _snapshot_impact(selection["source_snapshot"], target_nodes)
    generation_impacts = []
    for generation in list_generations(_storage_dir(), selection_id):
        generation_impact = _snapshot_impact(generation["source_snapshot"], target_nodes)
        generation_impacts.append(
            {
                "generation_id": generation["id"],
                "created_at": generation["created_at"],
                "is_stale": generation_impact["is_stale"],
                "summary": generation_impact["summary"],
            }
        )
    return {
        "selection_id": selection_id,
        "source_version_id": selection["version_id"],
        "checked_against_version": target_version,
        **impact,
        "affected_generation_ids": [
            item["generation_id"] for item in generation_impacts if item["is_stale"]
        ],
        "generations": generation_impacts,
    }


@app.get("/generations/{generation_id}", tags=["generations"])
def get_saved_generation(generation_id: str):
    generation = get_generation(_storage_dir(), generation_id)
    if generation is None:
        raise HTTPException(status_code=404, detail=f"Generation {generation_id!r} not found.")
    target_version, target_nodes = _staleness_target(
        generation["source_version_id"], None
    )
    return {
        **generation,
        "staleness": {
            "checked_against_version": target_version,
            **_snapshot_impact(generation["source_snapshot"], target_nodes),
        },
    }


@app.get("/generations/{generation_id}/staleness", tags=["generations"])
def get_generation_staleness(generation_id: str, against_version_id: str | None = None):
    generation = get_generation(_storage_dir(), generation_id)
    if generation is None:
        raise HTTPException(status_code=404, detail=f"Generation {generation_id!r} not found.")
    target_version, target_nodes = _staleness_target(
        generation["source_version_id"], against_version_id
    )
    return {
        "generation_id": generation_id,
        "selection_id": generation["selection_id"],
        "source_version_id": generation["source_version_id"],
        "checked_against_version": target_version,
        **_snapshot_impact(generation["source_snapshot"], target_nodes),
    }
