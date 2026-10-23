# CT-200 Manual API

## Overview

This FastAPI prototype parses CardioTrack CT-200 blood pressure monitor manuals into section trees. Part 1 provides health, tree, and search endpoints. Part 2 stores each uploaded PDF and parsed tree as a separate version, then compares sections between versions. Part 3 adds named selections pinned to a manual version, immutable source snapshots, generation provenance storage, and selection/generation staleness checks.

The parser combines spans into visual lines, joins wrapped headings, preserves paragraph and list boundaries, writes detected tables as row-oriented text, and uses section numbering with typography as a fallback for hierarchy. Uploads are checked for CT-200 product details and the distinctive `Device Overview` and `Intended Use` sections. PDFs that appear to be assignment prompts or unrelated documents are rejected with `422` and are not stored as manual versions.

Version metadata is stored in SQLite. Uploaded PDFs and parsed JSON trees are stored under `storage/uploads/` and `storage/trees/`. By default, the storage directory is `storage/` in the project folder. Set `CT200_STORAGE_DIR` before starting the server to choose another location. Existing Part 1 data remains in `output/ct200_tree.json`.

## Install and run

From the project folder in PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the API:

```powershell
python -m uvicorn app:app --reload
```

Open Swagger UI at <http://127.0.0.1:8000/docs>.

Run the automated tests:

```powershell
python -m pytest -q
```

## API endpoints

### Part 1: current parsed tree

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/health` | API health check. |
| `GET` | `/tree` | List the nodes from `output/ct200_tree.json` with previews and tree metadata. |
| `GET` | `/tree/{node_id}` | Return one node's full body and immediate children. |
| `GET` | `/search?q=pressure` | Search node headings and body text. |

### Part 2: stored versions

| Method | Route | Description |
| --- | --- | --- |
| `POST` | `/versions/upload?filename=manual.pdf` | Upload and parse one PDF as the raw `application/pdf` request body. Returns version metadata. |
| `GET` | `/versions` | List all versions in version-number order, with the latest version ID. |
| `GET` | `/versions/latest` | Return metadata for the latest uploaded version. |
| `GET` | `/versions/{version_id}` | Return metadata for one stored version. |
| `GET` | `/versions/{version_id}/tree` | Return the selected version's full parsed tree. |
| `GET` | `/versions/compare?old_version_id=…&new_version_id=…` | Compare two distinct versions. |
| `GET` | `/versions/compare/node?old_version_id=…&new_version_id=…&node_id=…` | Compare one node from either version, including a lightweight diff for changes. |

### Part 3: saved selections and provenance

| Method | Route | Description |
| --- | --- | --- |
| `POST` | `/selections` | Save a named set of node IDs from one verified manual version. |
| `GET` | `/selections/{selection_id}` | Return selection metadata, exact source text, and hashes. |
| `POST` | `/selections/{selection_id}/generate` | Generate and persist validated QA cases from the pinned source snapshot. |
| `GET` | `/selections/{selection_id}/generations` | List saved generations linked to a selection, including their stored cases and provenance. |
| `GET` | `/selections/{selection_id}/staleness` | Compare the selection and its generations to the latest manual version, or `?against_version_id=…`. |
| `GET` | `/versions/{version_id}/selection-impact` | List older selections and generations made stale by a specified version. |
| `GET` | `/generations/{generation_id}` | Return stored cases, exact source provenance, and a latest-version staleness summary. |
| `GET` | `/generations/{generation_id}/staleness` | Compare one generation's source snapshot to the latest manual version, or `?against_version_id=…`. |

Create a selection with node IDs copied from `GET /versions/{version_id}/tree`:

```json
{
  "name": "Pressure measurement checks",
  "version_id": "a8d1ddea633e4350bf51dcbb3cdb52e8",
  "node_ids": ["node-id-from-the-version-tree"]
}
```

The version must exist and be marked as a verified CT-200 manual. Every node ID must occur in that version's stored tree. Blank names, blank or repeated IDs, missing versions, unverified versions, and missing node references return validation errors instead of creating a partial selection.

Each selection stores the exact selected node text, headings, levels, parent IDs, matching paths, per-node SHA-256 hashes, and a hash of the full snapshot. The selection is pinned to its original version and has no update operation. Creating another selection with the same name creates a separate record.

The existing `versions.sqlite3` database receives additive `selections` and `generations` tables when a Part 3 persistence endpoint is first used. Existing version rows and files are not rewritten. A generation record copies the selection's version ID, node IDs, source snapshot, and hashes so its provenance remains available if newer versions are uploaded. Validated cases are stored as JSON in the existing `cases_json` column; this keeps each generation's structured output intact without another storage system or a schema change. Each successful request appends a distinct generation, including repeated requests for the same selection; earlier cases are never overwritten. Failed or invalid responses are not saved.

#### LLM generation

Groq is the initial provider adapter. The assignment allows a provider choice; Groq was selected as a small, replaceable implementation using the existing `httpx` dependency. The default model is `openai/gpt-oss-20b`; set `GROQ_MODEL` to select another compatible model. The application loads these settings from a project-root `.env` file when present, using `python-dotenv`; explicit process environment variables take precedence. `.env` is ignored by Git. The key is never stored with generations or included in error responses. If it is missing, the endpoint returns an actionable configuration error and does not create a generation. Groq's OpenAI-compatible Chat Completions endpoint and JSON response mode are used.

The endpoint reconstructs its prompt from the saved source snapshot, not the latest tree. The prompt treats excerpts as source data and forbids turning explicit prohibitions into test actions. The response must be one JSON object with a cases array containing 3–5 items. Each item has exactly these fields: title, requirement, preconditions (which may be empty), ordered steps, expected_result, and source_node_ids. Pydantic rejects missing, extra, mistyped, blank, or out-of-range fields. Additional checks reject positive actions matching explicit source prohibitions, substantive terms absent from the cited text, claims that cannot be matched as short ordered source phrases, unselected node IDs, and numeric details absent from cited source text.

If JSON parsing, schema validation, source-reference validation, prohibition checks, or source-phrase checks fail, the API makes at most one correction request that includes concise validation feedback. A second invalid response returns a failure and is not persisted. Provider authentication failures, rate limits, timeouts, and outages return safe errors without exposing credentials or provider response bodies; those transport failures do not trigger a content-correction retry. The creation response exposes completion_attempts (and failures include the count in their detail); counts are not added to the existing SQLite schema. The generation record stores provider/model, timestamp, validated cases, a hash of the successful raw completion, and the exact source provenance copied from the selection.

The deterministic source checks are deliberately conservative, but they are not a general natural-language entailment or safety proof. They may reject a valid paraphrase, and wording combinations that use only source terms can still require human semantic review. Treat a case that fails these checks as unusable; review accepted cases against their cited excerpt before relying on them. The API does not call a provider during tests; tests mock provider responses. These decisions are documented in the separate `Approach_Document.md`.

Staleness checks compare node IDs first, then the same normalized heading paths and sibling ordinals used by version comparison. A matched node is changed when its exact body hash differs. If neither strategy finds one unique match, the API reports it as `unmatched` and marks the selection/generation stale. This safely covers removed or structurally renamed sections; it does not claim to distinguish those cases without a reliable match. `GET /generations/{generation_id}` includes a staleness summary; the dedicated staleness endpoints include per-node source and target IDs/hashes and the checked version. `GET /versions/{version_id}/selection-impact` scans selections pinned to that version or an earlier version and lists the affected selections and generations. Historical snapshots and cases remain unchanged, and the API never regenerates output automatically.

The upload response includes an ID, sequential version number, original filename, UTC upload time, SHA-256 hash of the PDF bytes, parsed node count, and `is_ct200_manual: true`. Missing versions return `404`; malformed PDFs and attempts to compare a version with itself return `400`; valid PDFs that do not look like the CT-200 manual return `422`.

Previously stored records from before manual validation was added are marked unverified during the SQLite migration. They remain on disk, but are excluded from the manual-version list and cannot be used for tree retrieval or comparison. Their metadata reports `is_ct200_manual: false`.

## Example requests

Upload a PDF using curl:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/versions/upload?filename=manual-v1.pdf" `
  -H "Content-Type: application/pdf" --data-binary "@manual-v1.pdf"
```

Example upload response:

```json
{
  "id": "9e72…",
  "version_number": 1,
  "filename": "manual-v1.pdf",
  "uploaded_at": "2026-10-23T10:20:30+00:00",
  "content_hash": "…",
  "node_count": 28
}
```

List versions:

```http
GET /versions
```

Retrieve a specific tree:

```http
GET /versions/9e72…/tree
```

Compare two versions:

```http
GET /versions/compare?old_version_id=9e72…&new_version_id=32ac…
```

The comparison response includes counts for `added`, `removed`, `modified`, and `unchanged`, plus one result per section. Each result includes the old and new section IDs when available, a matching key, heading, change type, normalized content hashes, and short previews for changed sections.

## Try two versions in Swagger UI

1. Open `/docs` and expand `POST /versions/upload`.
2. Enter a filename ending in `.pdf` in the `filename` query field, choose the first PDF as the request body, then select **Execute**. Copy the returned `id`.
3. Upload the second PDF the same way and copy its `id`.
4. Expand `GET /versions/compare`, paste the first ID into `old_version_id` and the second into `new_version_id`, then select **Execute**.
5. Review the summary counts and the per-section changes. Use `GET /versions/{version_id}/tree` to inspect either stored tree.

## Reproducible end-to-end workflow

The following PowerShell walkthrough uses only response fields returned by the API. Start the server first:

```powershell
python -m uvicorn app:app --reload
```

In another PowerShell window, from the project folder, upload the two supplied manuals. The returned `id` values are the version IDs used by later requests:

```powershell
$v1 = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/versions/upload?filename=manual-v1.pdf" `
  -ContentType "application/pdf" `
  -InFile "data/ct200_manual.pdf"

$v2 = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/versions/upload?filename=manual-v2.pdf" `
  -ContentType "application/pdf" `
  -InFile "data/ct200_manual_v2.pdf"

$v1.id
$v2.id
```

Retrieve v1's tree and choose an actual non-root node ID from its response. The selection request uses the `name`, `version_id`, and `node_ids` fields defined by `SelectionCreate`:

```powershell
$v1Tree = Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/versions/$($v1.id)/tree"

$nodeId = ($v1Tree.nodes | Where-Object { $_.level -gt 0 } | Select-Object -First 1).id

$selection = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/selections" `
  -ContentType "application/json" `
  -Body (@{
    name = "Manual v1 review"
    version_id = $v1.id
    node_ids = @($nodeId)
  } | ConvertTo-Json)

$selection.id
```

Generate and persist validated cases from the pinned v1 snapshot. This is the only step that requires the configured provider credentials and may contact Groq; tests use mocked provider responses instead:

```powershell
$generation = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/selections/$($selection.id)/generate"

$generation.id
$generation.cases
```

Check whether the selection and its saved generation are stale against v2. The response includes `checked_against_version`, `summary`, `source_nodes`, and `affected_generation_ids`:

```powershell
$staleness = Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/selections/$($selection.id)/staleness?against_version_id=$($v2.id)"

$staleness.is_stale
$staleness.summary
$staleness.source_nodes
$staleness.affected_generation_ids
```

Finally, retrieve the immutable saved generation and its staleness summary:

```powershell
$savedGeneration = Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/generations/$($generation.id)"

$savedGeneration.source_version_id
$savedGeneration.source_snapshot
$savedGeneration.cases
$savedGeneration.staleness
```

The generation record remains tied to v1 even after v2 is uploaded. Staleness detection reports changes or unmatched nodes; it does not rewrite or automatically regenerate the saved cases. The workflow above is a live-provider walkthrough and was not executed as part of offline verification.

## Regenerating local artifacts

The parser CLI rebuilds the Part 1 JSON tree from a PDF. The optional second argument controls the output path; if omitted, it defaults to `output/ct200_tree.json`:

```powershell
python parser.py data/ct200_manual.pdf output/ct200_tree.json
```

To produce a separate local tree for the second supplied PDF without replacing the Part 1 tree:

```powershell
python parser.py data/ct200_manual_v2.pdf output/ct200_tree_v2.json
```

The versioned PDF, parsed tree, and SQLite metadata are created by `POST /versions/upload`. Start the API with an optional storage root when a separate local version store is desired:

```powershell
$env:CT200_STORAGE_DIR = (Join-Path (Get-Location) "storage-walkthrough")
python -m uvicorn app:app --reload
```

With no `CT200_STORAGE_DIR`, version artifacts are written under `storage/`: uploaded PDFs go to `storage/uploads/`, parsed version trees go to `storage/trees/`, and metadata goes to `storage/versions.sqlite3`. These generated artifacts are ignored by Git. The Part 3 `selections` and `generations` tables are created additively when those persistence endpoints are first used. The API does not provide a destructive reset command; preserve an existing local store and use a separate `CT200_STORAGE_DIR` when reproducing a clean run.

## How comparison works

The comparison first matches sections with the parser's deterministic IDs. For sections whose IDs changed, it falls back to normalized heading paths (case-folded headings, collapsed whitespace, and sibling ordinals for duplicate headings). Section body hashes are computed after Unicode normalization and whitespace collapsing, so line breaks and repeated spaces alone do not count as edits. Text case and punctuation remain meaningful.

Added and removed sections include the available old or new ID, content hash, and preview. Modified sections include both versions. Unchanged sections are included in `changes` and counted in the summary, without previews.

## Current limitations

- Section recognition still uses numbering and font-size heuristics; unusual or scanned PDFs may need OCR or document-specific rules.
- Tables are preserved as text rows, but complex multi-column layouts may still be read in the wrong order. Images and diagrams are not extracted as visual content.
- Some embedded PDF fonts have broken character maps and PyMuPDF may return replacement characters for glyphs that are visually present. OCR is not bundled in this prototype.
- Interface cleanup removes known browser warnings, menu labels, tab labels, and repeated edge text. It does not try to guess whether every arbitrary line is interface chrome.
- A changed heading or changed ancestor heading can change parser IDs and the normalized heading path. Such a section may appear as a removal and an addition instead of a rename or modification.
- Duplicate sections with the same heading are matched by sibling order, so inserting a duplicate earlier in the tree can shift matches.
- Staleness matching uses only stable node IDs and normalized heading paths. Sections with changed heading ancestry can be reported as `unmatched`; this is intentionally safer than attaching a selection to a guessed section.
- This is a local prototype without authentication, upload-size limits, or production deployment hardening.
