# Tri9T AI Engineering Internship Assignment — Approach Document

## Scope and evidence

This document describes the implementation currently in the project. It is based on [parser.py](./parser.py), [version_store.py](./version_store.py), [selection_store.py](./selection_store.py), [generation.py](./generation.py), [app.py](./app.py), [README.md](./README.md), the tests in [test_parser.py](./test_parser.py), [test_versioning.py](./test_versioning.py), and [test_part3.py](./test_part3.py). The assignment update available in [data/IMPORTANT_ASSIGNMENT_UPDATE (Read to check MD).txt](./data/IMPORTANT_ASSIGNMENT_UPDATE%20(Read%20to%20check%20MD).txt) says to work from the provided PDF rather than a Markdown file and leaves the parsing tools and approach open.

After adding the OCR fallback, I ran the full suite with the project Python 3.12 environment: 79 passed with 2 dependency warnings. I also ran `verify_e2e_workflow.py`; its v1/v2 upload, comparison, pinned generation, provenance, staleness, and retrieval checks passed using a mocked provider. No external LLM request was made.

## 1. Architecture and data model

The system is a local FastAPI application with three layers:

1. **Parsing:** [parser.py](./parser.py) uses PyMuPDF (`fitz`) to convert PDF pages into visual text/table lines, falls back to Tesseract for pages with insufficient or damaged embedded text, then reconstructs a flat list representing a tree.
2. **Versioning and comparison:** [version_store.py](./version_store.py) stores version metadata in SQLite and stores the original PDF and parsed tree as files. It compares nodes using stable IDs first and normalized heading paths second.
3. **Selections, generation, and API orchestration:** [app.py](./app.py) exposes the endpoints. [selection_store.py](./selection_store.py) persists selections and generations. [generation.py](./generation.py) builds the prompt, calls the Groq-compatible endpoint, validates output, and returns provenance.

The core parsed object is `TreeNode`:

- `id`: deterministic 16-character SHA-256 prefix derived from parent ID, heading, and same-parent occurrence number.
- `heading`, `level`, and `body`.
- `parent_id` and a list of child IDs.
- `content_hash`: SHA-256 of the exact stored body text.

The parser returns a flat node list with a synthetic `ROOT` node at level 0. Parent/child IDs define the hierarchy. JSON serialization is used for parsed trees.

Version metadata contains an ID, sequential version number, original filename, UTC upload time, PDF SHA-256, node count, and `is_ct200_manual`. Uploaded PDFs are saved under `storage/uploads/`; trees are saved under `storage/trees/`; metadata is in `versions.sqlite3`. Existing records are migrated additively with an `is_ct200_manual` flag and legacy unverified records are excluded from manual-version operations.

Selections are immutable records containing a name, source version ID, selected node IDs, exact node snapshots, per-node hashes, and a full snapshot hash. A generation appends a record containing provider/model, timestamp, source provenance copied from the selection, validated cases as JSON, and a hash of the successful raw completion.

## 2. PDF parsing and hierarchy reconstruction

### Extraction pipeline

For every page, `_extract_lines` reads PyMuPDF text dictionaries with whitespace preservation. Spans on a visual line are joined using their text, x coordinates, leading-space information, and a gap heuristic. The dominant character-weighted font size and combined flags are retained. Lines are sorted by page y position and x position.

Before hierarchy reconstruction, the parser checks each page's extracted lines. It uses OCR if the page has fewer than 40 alphanumeric embedded-text characters or contains a Unicode replacement character, and the page has text, an image, or drawing content. Blank pages are skipped. `_ocr_page_lines` renders a page at 2× scale and calls Tesseract through `pytesseract`; Tesseract words are grouped by block, paragraph, and line into `_TextLine` objects. Their boxes are scaled back to page coordinates, and line height is rounded to a coarse 3-point size bucket to reduce variation caused by different letter shapes. OCR lines replace the weak embedded extraction for that page, after which the existing furniture cleanup, heading detection, hierarchy reconstruction, and content hashing run unchanged. `TESSERACT_CMD` selects the executable; otherwise the parser searches `PATH` for `tesseract`.

Detected tables are extracted with `page.find_tables()`. Their cells are normalized and rendered as pipe-separated rows, with a Markdown-like separator row when there is more than one column. Text lines whose centers fall inside a detected table region are removed, and the table is reinserted as one `table` line. This prevents bold table labels such as `Parameter` and `Value` from becoming false headings.

### Known irregularities and handling

- **Wrapped headings:** adjacent same-page lines with similar font size, close vertical position, similar x alignment, and both heading-like are joined. Hyphen or soft-hyphen endings are joined without an extra space.
- **Numbered headings:** dotted numeric forms such as `2.1` and `3.2.1`, and appendix forms such as `A.1` or `Appendix A`, provide explicit levels. A plain one-level numbered line is treated as a heading only when it is visibly larger than body text, reducing confusion with numbered lists.
- **Typography-only headings:** when numbering is unavailable, font-size thresholds are learned from the document. The most frequent visual-line size is treated as body text; larger sizes are mapped to heading levels.
- **Missing numbering ancestors:** a numbered heading searches previously seen prefixes for a parent. If an intermediate prefix was not found, the level is clamped to the next actual tree level rather than creating a missing parent.
- **Unnumbered hierarchy:** a stack of recent heading levels is used to find the parent; deeper levels are also clamped to one level below the selected parent.
- **Out-of-order numbered sections:** numbered siblings are naturally sorted for traversal after parsing. Node IDs, parent IDs, bodies, and content are not rewritten by this ordering step.
- **Document title:** a large first-page line near the top, before the first section and without section numbering, is appended to the root body rather than treated as a section.
- **Paragraphs and lists:** body lines are joined with spaces within a paragraph, blank-line separators when geometry indicates a paragraph break, and newline separators for adjacent list items. Bullets include common bullet characters, hyphens/en dashes, numeric markers, and letter markers.
- **Page furniture and interface text:** explicit browser warnings, tab labels, common menu-only labels, and repeated top/bottom edge text are filtered. The filter is intentionally conservative and does not identify arbitrary interface chrome.
- **Duplicate headings:** occurrence counts scoped to a parent are included in the ID seed. Same text under different parents also differs because the parent ID is included.
- **OCR and empty input:** a missing path raises `FileNotFoundError`; invalid or zero-byte PDFs propagate a PyMuPDF error. A page with little or replacement-character text invokes OCR if it contains page content. Missing Tesseract, OCR execution errors, malformed OCR data, and an OCR result with no recognized text raise explicit errors rather than being treated as a successful empty extraction. The upload API returns OCR failures as HTTP 503 and removes the incomplete upload. A genuinely blank page is skipped. A PDF with no usable text after processing may still contain only a root node if no page required OCR.
- **Manual identity:** the parser has a check requiring CardioTrack, CT-200, blood-pressure text, and both `Device Overview` and `Intended Use`. A prompt mentioning CT-200 alone is therefore rejected as an unverified manual with HTTP 422 and is not stored as a version.

## 3. Why this approach, validation, and limitations

PyMuPDF remains the primary extractor because it provides text and layout metadata for ordinary PDFs. Tesseract is a page-level fallback for scanned pages and pages whose embedded text is sparse or damaged; this avoids OCR on pages with sufficient embedded text. Both extraction paths produce `_TextLine` records, so the existing deterministic hierarchy and hashing pipeline remains shared. Numbering is preferred because it encodes structure directly; typography is a useful fallback when numbering is absent.

Validation is both automated and fixture-based. [test_parser.py](./test_parser.py) checks deterministic IDs, duplicate headings, parent/child consistency, malformed inputs, content hashes, numbering inference, serialization, title/hierarchy cleanup, table text, wrapped headings, paragraphs, bullets, and OCR fallback behavior. OCR tests mock Tesseract's structured response and cover text-page bypass, sparse-page fallback, coordinate conversion, a missing executable, an engine error, and an empty OCR result; they do not require Tesseract to be installed. [test_versioning.py](./test_versioning.py) exercises real and synthetic PDF uploads, manual validation, storage, added/removed/modified/unchanged comparisons, whitespace normalization, node-specific comparison, and clear API reporting when OCR is unavailable.

OCR fallback requires the Python packages `pytesseract` and Pillow plus an external Tesseract executable. On Windows, set `TESSERACT_CMD` to the installed executable when it is not available on `PATH`; a common location is `C:\Program Files\Tesseract-OCR\tesseract.exe`:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
python -m uvicorn app:app --reload
```

OCR words and boxes are approximate, and grouping by Tesseract line IDs does not restore complete layout semantics. Recognition may vary with Tesseract versions and can distort reading order. Scanned tables are treated as approximate text lines; column alignment, merged cells, and complex multi-column reading order are not guaranteed. Figures and diagrams are not interpreted as visual content, and captions may attach imperfectly. The parser is document-heuristic, not a general PDF layout engine.

## 4. Version matching and change detection

`compare_trees` ignores the synthetic root and matches each old node in two stages:

1. Exact parser-generated node ID.
2. Normalized heading path, where Unicode is NFKC-normalized, whitespace is collapsed, headings are case-folded, and duplicate sibling headings receive an ordinal.

For a matched pair, body text is normalized for Unicode and whitespace before comparison. Case and punctuation remain meaningful. Equal normalized bodies are `unchanged`; different bodies are `modified`. Unmatched old nodes are `removed`, and unmatched new nodes are `added`. Changed results include short previews and both hashes; unchanged results omit previews. The node-specific endpoint accepts a node ID from either selected version and reports added, removed, modified, or unchanged behavior using the same comparison logic.

Failure modes include malformed PDFs (400), unknown versions (404), comparing a version with itself (400), and unverified legacy records (not usable for trees/comparison/selections). A changed heading or ancestor can change IDs and paths, so a rename can appear as removal plus addition. Duplicate headings are matched by sibling ordinal; inserting a duplicate earlier can shift the match. Matching is therefore deterministic, but not a semantic section-identity guarantee.

## 5. Selection pinning and exact source snapshots

Creating a selection first loads a verified manual tree and rejects blank or duplicate input IDs through Pydantic validation. It then rejects any ID not present in the selected version. For every selected node it stores the exact heading, level, parent ID, body, content hash, and normalized matching key. It also stores the selected IDs and a hash of the serialized full snapshot.

There is no selection update operation. The generation prompt is rebuilt from this saved snapshot, never from the latest tree. A generation copies the source version, source IDs, source snapshot, hashes, and snapshot hash into its own row. This preserves the source used for the result even after later PDFs are uploaded. Repeated successful generation requests append distinct generations; previous cases are not overwritten. Failed or invalid provider responses are not saved.

## 6. Groq prompt, validation, retries, persistence, and duplicate policy

The provider adapter is Groq over its OpenAI-compatible Chat Completions endpoint using `httpx`. Configuration comes from process environment or the project `.env`; the API key is not persisted or returned. The default model is `openai/gpt-oss-20b`, configurable with `GROQ_MODEL`.

The system prompt requires a single JSON object with a `cases` array of 3–5 cases. Each case has exactly `title`, `requirement`, `preconditions`, `steps`, `expected_result`, and `source_node_ids`. The source prompt embeds the saved excerpts and tells the model to treat them as source data and not turn explicit prohibitions into test actions.

Validation has two layers:

- Pydantic strict models reject extra fields, missing fields, wrong types, blank text, empty steps, repeated source IDs, and out-of-range case counts.
- Deterministic checks reject unselected source IDs, unsupported numeric details, positive actions that conflict with parsed prohibitions, substantive terms not present in the selected text, and claims that cannot be aligned to short ordered source-term sequences.

If JSON parsing, schema validation, grounding, prohibition, or source-phrase validation fails, one correction request is made with concise feedback. The maximum is two completion attempts total. A second invalid response fails with no saved generation. Transport/configuration failures such as missing credentials, authentication failure, rate limits, timeout, or outage are returned as safe errors and do not use a content-correction retry. Completion attempts are exposed in the create response or failure detail but are not added to the SQLite schema.

The duplicate-selection policy is intentionally permissive: the same name may be saved more than once, because selection identity is the generated record ID and each request is a separate immutable snapshot. Repeated generation for one selection also appends a new generation rather than deduplicating or overwriting. There is no case-level deduplication across generations.

The semantic checks are heuristic, not a guarantee of correctness. They can reject valid paraphrases, and a generated case composed of source vocabulary can still be semantically wrong. Accepted cases require human review against their cited source excerpt before being treated as reliable QA requirements.

## 7. Staleness detection and retrieval behavior

Selection and generation staleness checks compare the saved snapshot with the latest verified manual version by default, or with an explicitly supplied same-or-newer version. Matching uses the saved node ID first, then a unique normalized heading-path match. A matched node is `unchanged` when its exact body hash matches, `changed` when it differs, and `unmatched` when neither method produces a unique match. Changed or unmatched nodes make the selection/generation stale.

Retrieval endpoints return the immutable stored selection/generation and add a staleness summary. The version `selection-impact` endpoint scans selections from that version or older and lists affected selections and generations. The system never rewrites historical snapshots and never regenerates cases automatically. Checking against an older version than the source is rejected. The matching strategy is conservative: it can report `unmatched` after an ancestor rename, and it cannot reliably distinguish a deletion from a structural rename without a match.

## 8. SQLite JSON storage and assignment alignment

The current implementation stores version metadata relationally and stores each selection/generation's structured arrays in JSON columns. This is a reasonable fit for immutable, read-mostly provenance: a generation's cases, source node IDs, source snapshot, and hashes are retrieved together, and the JSON preserves the exact structured output without a second document store or schema migration for every case-field change. PDF bytes and parsed trees remain separate files so the SQLite database does not contain large binary documents.

The assignment materials available in this repository specify PDF hierarchy reconstruction but do not specify a required database schema or a separate normalized storage model for generated cases. Consequently, there is no verified mismatch with an explicit storage requirement. If the expected evaluation requires one relational row per generated case, the current `generations.cases_json` design would be the mismatch: it would need a child `generation_cases` table (and likely child tables for steps/source references) or a documented migration. That requirement cannot be inferred from the available assignment update.

## 9. Improvements with more time

- Add OCR confidence thresholds, page-level extraction provenance, and a review path for low-confidence OCR results.
- Add stronger layout handling for multi-column text, merged/irregular tables, figures, captions, and reading order.
- Create golden extraction fixtures from both supplied manual PDFs and compare expected headings, paths, tables, and body fragments—not only structural invariants.
- Add parser confidence signals and a human review workflow for low-confidence sections.
- Replace heuristic version matching with stable source anchors or a reviewable diff for renames and duplicate-heading shifts.
- Add a semantic evaluator or human approval gate for generated cases; the current lexical checks cannot establish entailment.
- Normalize generated cases into queryable relational tables if downstream reporting requires case-level filtering.
- Add upload-size limits, authentication/authorization, rate limiting, concurrency/load testing, stronger path/file integrity checks, and production deployment controls.
- Add provider abstraction and deterministic offline fixtures for every generation validation failure mode while keeping live provider calls out of tests.

## 10. Decision log

### What could silently produce wrong results, and how would I detect it?

The largest silent risks are incorrect reading order in multi-column PDFs, missed or false headings from font-size heuristics, table extraction that loses column semantics, broken font maps, and a semantic error in a case that passes lexical grounding. Duplicate-heading ordinal shifts and heading/ancestor renames can also produce an apparently valid but incorrect version match. Detection should combine golden PDF fixtures, extracted-text spot checks, node-count/path/hash drift checks, comparison review for additions/removals, and human review of every accepted generated case against its cited excerpt. The API's semantic validation is heuristic, not a correctness guarantee.

### Where did I choose simplicity over correctness, and what would break first in production?

I chose a deterministic PyMuPDF-plus-heuristics parser with a page-level Tesseract fallback, pipe-text tables, flat JSON trees, and lexical source checks instead of a full layout model, semantic entailment system, or normalized case database. Production would first struggle with OCR errors, complex layouts, false confidence in generated cases, and ambiguous version matches. Operational concerns such as authentication, upload limits, rate limiting, and concurrent workload handling are also intentionally outside this prototype.

### What input is unsupported, and how does the system respond?

Unsupported or unreliable inputs include non-PDF/invalid PDFs, empty PDFs, scanned PDFs when Tesseract is unavailable or fails, PDFs with badly broken embedded font maps that OCR cannot recover, unrelated documents, assignment prompts that merely mention CT-200, and complex layouts whose reading order cannot be recovered by the implemented heuristics. Invalid PDFs produce a 400-style upload error; valid but non-CT-200 documents produce HTTP 422 and are not stored as manual versions; OCR failures produce HTTP 503 with a readable reason, while imperfect OCR may still produce a nonempty tree requiring review. Provider output that is malformed, ungrounded, prohibited, or outside the schema is retried once and then rejected without persistence.
