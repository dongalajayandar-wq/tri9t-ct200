"""
parser.py — Extract a structured section tree from the CT-200 PDF.

Strategy
--------
1. Extract full text lines, geometry, and font metadata from every page.
2. Preserve detected tables as row-oriented text, and join wrapped heading lines.
3. Prefer section numbering for hierarchy, with typography as a fallback.
4. Keep paragraph and list boundaries in node bodies and hash the final text.

Known limitations (will be documented in README and approach doc):
- Font-size heuristics break if the PDF uses unusual sizes for emphasis vs. headings.
- Tables are extracted as plain text lines — no column alignment is preserved.
- Figures/images are noted by placeholder text only; captions may or may not attach
  to the correct node.
- If two sibling sections share the exact same heading text, they still get unique
  IDs because we include an occurrence counter in the ID seed.
- PDFs with broken embedded font maps can still contain replacement characters;
  OCR is not currently bundled with this prototype.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """One section / paragraph in the document tree."""
    id: str
    heading: str
    level: int                          # 0 = root, 1 = top-level section, …
    body: str = ""
    content_hash: str = ""
    parent_id: Optional[str] = None
    children: list[str] = field(default_factory=list)   # child IDs

    def compute_hash(self) -> str:
        """SHA-256 of the body text (used for staleness detection later)."""
        self.content_hash = hashlib.sha256(self.body.encode("utf-8")).hexdigest()
        return self.content_hash


@dataclass
class _TextLine:
    """One visually grouped line (or detected table) extracted from a PDF page."""
    text: str
    size: float
    flags: int
    bbox: tuple[float, float, float, float]
    page_number: int
    kind: str = "text"


def _make_id(heading: str, parent_id: Optional[str], occurrence: int) -> str:
    """Deterministic node ID from heading path + occurrence index."""
    seed = f"{parent_id or 'ROOT'}|{heading}|{occurrence}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Page text extraction and cleanup helpers
# ---------------------------------------------------------------------------

def _join_spans(spans: list[dict]) -> tuple[str, float, int]:
    """Join font spans on one visual line without dropping word spacing."""
    pieces: list[str] = []
    previous_x1: float | None = None
    previous_text = ""
    sizes: list[float] = []
    flags = 0

    for span in spans:
        raw_text = span.get("text", "")
        text = raw_text.strip()
        if not text:
            continue
        x0, _, x1, _ = span.get("bbox", (0, 0, 0, 0))
        size = float(span.get("size", 0))
        sizes.extend([size] * max(1, len(text)))
        flags |= int(span.get("flags", 0))

        if pieces:
            gap = x0 - (previous_x1 or x0)
            has_leading_space = raw_text[:len(raw_text) - len(raw_text.lstrip())] != ""
            needs_space = has_leading_space or gap > max(1.2, size * 0.12)
            if previous_text.endswith(("-", "\u00ad")):
                needs_space = False
            if text[:1] in ",.;:!?)]}»":
                needs_space = False
            if needs_space:
                pieces.append(" ")
        pieces.append(text)
        previous_x1 = x1
        previous_text = text

    if not pieces:
        return "", 0.0, flags
    # Character-weighting prevents a one-character superscript span from
    # changing the line's dominant font size.
    from collections import Counter
    size_counts = Counter(round(size, 1) for size in sizes)
    size = size_counts.most_common(1)[0][0]
    return "".join(pieces), size, flags


def _format_table(rows: list[list[str | None]]) -> str:
    """Render a detected PDF table as readable pipe-separated rows."""
    cleaned = [
        [
            re.sub(r"(?<=\w)\s*-\s*(?=\w)", "-", (cell or "").replace("\n", " ").strip())
            for cell in row
        ]
        for row in rows
    ]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""

    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    used_columns = [
        index for index in range(width)
        if any(row[index] for row in cleaned)
    ]
    cleaned = [[row[index] for index in used_columns] for row in cleaned]
    width = len(used_columns)
    rendered = ["| " + " | ".join(row) + " |" for row in cleaned]
    if len(cleaned[0]) > 1:
        separator = "| " + " | ".join("---" for _ in range(width)) + " |"
        rendered.insert(1, separator)
    return "\n".join(rendered)


def _extract_lines(page: fitz.Page, page_number: int) -> list[_TextLine]:
    """Extract complete visual lines and replace detected table cells with rows."""
    result: list[_TextLine] = []
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text, size, flags = _join_spans(line.get("spans", []))
            if text:
                bbox = tuple(float(value) for value in line.get("bbox", (0, 0, 0, 0)))
                result.append(_TextLine(text, size, flags, bbox, page_number))

    # Tables should remain content inside their section. Treating individual
    # bold cells as headings was the source of the old "Parameter"/"Value"
    # level-4 nodes.
    try:
        tables = page.find_tables().tables
    except (AttributeError, RuntimeError, ValueError):
        tables = []
    table_regions: list[tuple[float, float, float, float]] = []
    for table in tables:
        bbox = tuple(float(value) for value in table.bbox)
        table_text = _format_table(table.extract())
        if table_text:
            table_regions.append(bbox)
            result.append(_TextLine(table_text, 0.0, 0, bbox, page_number, "table"))

    if table_regions:
        result = [
            line for line in result
            if line.kind == "table" or not any(
                bbox[0] - 1 <= (line.bbox[0] + line.bbox[2]) / 2 <= bbox[2] + 1
                and bbox[1] - 1 <= (line.bbox[1] + line.bbox[3]) / 2 <= bbox[3] + 1
                for bbox in table_regions
            )
        ]
    return sorted(result, key=lambda line: (round(line.bbox[1], 1), round(line.bbox[0], 1)))


_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_BROWSER_WARNING_RE = re.compile(
    r"cette version du navigateur n'est plus compatible|this version of your browser is no longer supported",
    re.IGNORECASE,
)
_TAB_LABEL_RE = re.compile(r"^(?:onglet|tab)\s*\d*$", re.IGNORECASE)
_MENU_WORDS = {
    "fichier", "édition", "edition", "affichage", "outils", "aide",
    "file", "edit", "view", "insert", "format", "tools", "help",
}


def _normalized_text(text: str) -> str:
    text = _ZERO_WIDTH_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _is_interface_artifact(text: str) -> bool:
    """Return true only for common, confidently identifiable browser chrome."""
    normalized = _normalized_text(text)
    if not normalized:
        return True
    if _BROWSER_WARNING_RE.search(normalized) or _TAB_LABEL_RE.fullmatch(normalized):
        return True
    words = normalized.split()
    return 4 <= len(words) <= 8 and all(word in _MENU_WORDS for word in words)


def _filter_page_furniture(lines: list[_TextLine], page_heights: dict[int, float]) -> list[_TextLine]:
    """Remove explicit interface labels and repeated top/bottom page furniture."""
    from collections import defaultdict
    pages_by_text: dict[str, set[int]] = defaultdict(set)
    for line in lines:
        if line.kind != "text":
            continue
        page_height = page_heights.get(line.page_number, float("inf"))
        near_edge = line.bbox[1] <= page_height * 0.08 or line.bbox[3] >= page_height * 0.94
        if near_edge:
            pages_by_text[_normalized_text(line.text)].add(line.page_number)

    repeated = {
        key for key, pages in pages_by_text.items()
        if len(pages) >= 2 and len(pages) >= max(2, (len(page_heights) + 1) // 2)
    }
    kept = []
    for line in lines:
        if _is_interface_artifact(line.text):
            continue
        page_height = page_heights.get(line.page_number, float("inf"))
        near_edge = line.bbox[1] <= page_height * 0.08 or line.bbox[3] >= page_height * 0.94
        if near_edge and _normalized_text(line.text) in repeated:
            continue
        kept.append(line)
    return kept


def _classify_font_sizes(all_sizes: list[float]) -> dict[str, float]:
    """Given every font size seen in the document, pick thresholds for
    heading levels.  Returns a dict mapping level labels to minimum sizes."""
    if not all_sizes:
        return {}
    unique = sorted(set(all_sizes), reverse=True)

    # The body size is the most frequent line size rather than the most frequent
    # span size, so documents with many mixed-style words remain stable.
    from collections import Counter
    freq = Counter(all_sizes)
    body_size = freq.most_common(1)[0][0]

    # Anything larger than body text is a heading.
    heading_sizes = [s for s in unique if s > body_size]

    thresholds: dict[str, float] = {}
    if len(heading_sizes) >= 3:
        thresholds["h1"] = heading_sizes[0]
        thresholds["h2"] = heading_sizes[1]
        thresholds["h3"] = heading_sizes[2]
    elif len(heading_sizes) == 2:
        thresholds["h1"] = heading_sizes[0]
        thresholds["h2"] = heading_sizes[1]
    elif len(heading_sizes) == 1:
        thresholds["h1"] = heading_sizes[0]

    thresholds["body"] = body_size
    return thresholds


def _heading_level(size: float, flags: int, thresholds: dict[str, float], text: str = "") -> int:
    """Return a numbered or typography-based heading level, or 0 for body text."""
    numbered = _numbered_heading_level(text, size, thresholds.get("body", 0))
    if numbered is not None:
        return numbered
    if "h1" in thresholds and size >= thresholds["h1"]:
        return 1
    if "h2" in thresholds and size >= thresholds["h2"]:
        return 2
    if "h3" in thresholds and size >= thresholds["h3"]:
        return 3
    # Bold body-sized cells and labels are not headings by themselves.
    return 0


# ---------------------------------------------------------------------------
# Numbered-section regex (e.g. "1.", "2.3", "10.2.1", "A.1")
# ---------------------------------------------------------------------------

SECTION_NUM_RE = re.compile(
    r"^((?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)+)\.?)\s+(.+)$",
    re.IGNORECASE,
)
APPENDIX_RE = re.compile(
    r"^(Appendix\s+[A-Z])\b[:\s]*(.*)$",
    re.IGNORECASE,
)


def _section_number(text: str) -> tuple[str, ...] | None:
    """Return a section's dotted number parts, including appendix subnumbers."""
    m = SECTION_NUM_RE.match(text)
    if m:
        return tuple(m.group(1).rstrip(".").split("."))
    m = APPENDIX_RE.match(text)
    if m:
        return (m.group(1).casefold(),)
    return None


def _numbered_heading_level(text: str, size: float, body_size: float) -> int | None:
    """Avoid interpreting ordinary one-level numbered lists as headings."""
    parts = _section_number(text)
    if parts is None:
        return None
    if len(parts) > 1 or APPENDIX_RE.match(text):
        return len(parts)
    # A single "1." line is common in lists. Top-level sections in this manual
    # use a visibly larger font; a plain body-sized list item stays in the body.
    if size > body_size * 1.05:
        return 1
    return None


def _infer_level_from_numbering(text: str) -> tuple[int, str] | None:
    """Infer a heading level from numbering such as "3.2.1" or "A.1"."""
    parts = _section_number(text)
    if parts is not None:
        return len(parts), text
    return None


def is_ct200_manual(text: str) -> bool:
    """Check for product identity and distinctive CT-200 manual sections.

    Mentioning the product in an assignment prompt is not enough: the document
    must also contain sections characteristic of the actual manual.
    """
    normalized = _normalized_text(text)
    has_model = bool(re.search(r"\bct\W*200\b", normalized))
    required_sections = ("device overview", "intended use")
    return (
        "cardiotrack" in normalized
        and has_model
        and "blood pressure" in normalized
        and all(section in normalized for section in required_sections)
    )


def _is_heading_candidate(line: _TextLine, body_size: float) -> bool:
    return line.kind == "text" and (
        _numbered_heading_level(line.text, line.size, body_size) is not None
        or line.size >= body_size * 1.12
    )


def _merge_heading_fragments(lines: list[_TextLine], body_size: float) -> list[_TextLine]:
    """Join adjacent wrapped heading lines while leaving separate sections intact."""
    merged: list[_TextLine] = []
    for line in lines:
        if merged:
            previous = merged[-1]
            same_page = previous.page_number == line.page_number
            similar_style = abs(previous.size - line.size) <= 1.5
            close_vertical = (
                0 < line.bbox[1] - previous.bbox[1]
                <= max(previous.size, line.size) * 1.4
            )
            aligned = abs(previous.bbox[0] - line.bbox[0]) <= 18
            if (
                same_page and similar_style and close_vertical and aligned
                and _is_heading_candidate(previous, body_size)
                and _is_heading_candidate(line, body_size)
            ):
                separator = "" if previous.text.rstrip().endswith(("-", "\u00ad")) else " "
                merged[-1] = _TextLine(
                    text=previous.text.rstrip() + separator + line.text.lstrip(),
                    size=previous.size,
                    flags=previous.flags | line.flags,
                    bbox=(
                        min(previous.bbox[0], line.bbox[0]),
                        min(previous.bbox[1], line.bbox[1]),
                        max(previous.bbox[2], line.bbox[2]),
                        max(previous.bbox[3], line.bbox[3]),
                    ),
                    page_number=previous.page_number,
                )
                continue
        merged.append(line)
    return merged


def _is_bullet(text: str) -> bool:
    return bool(re.match(r"^\s*(?:[\u2022\u00b7\u25aa\u25cf*]|[-\u2013]\s+|\d+[.)]|[A-Za-z][.)])\s*", text))


def _append_body_line(node: TreeNode, line: _TextLine, previous: _TextLine | None) -> None:
    """Append text without losing paragraph and list-item boundaries."""
    text = line.text.strip()
    if not text:
        return
    if not node.body:
        node.body = text
        return
    if previous is None or previous.page_number != line.page_number:
        separator = "\n\n"
    elif previous.kind == "table" or line.kind == "table":
        separator = "\n\n"
    elif _is_bullet(previous.text) or _is_bullet(text):
        vertical_gap = line.bbox[1] - previous.bbox[3]
        paragraph_gap = max(4.0, max(line.size, previous.size) * 0.4)
        if not _is_bullet(previous.text) and vertical_gap > paragraph_gap:
            separator = "\n\n"
        else:
            separator = "\n"
    else:
        vertical_gap = line.bbox[1] - previous.bbox[3]
        paragraph_gap = max(4.0, max(line.size, previous.size) * 0.4)
        if vertical_gap > paragraph_gap:
            separator = "\n\n"
        elif previous.text.rstrip().endswith(("-", "\u00ad")):
            separator = ""
        else:
            separator = " "
    node.body += separator + text


def _is_document_title(line: _TextLine, body_size: float, page_height: float) -> bool:
    return (
        line.page_number == 0
        and line.bbox[1] <= page_height * 0.34
        and line.size >= body_size * 1.4
        and _section_number(line.text) is None
    )


def _numbered_sibling_key(node: TreeNode) -> tuple[tuple[int, int | str], ...] | None:
    """Return a natural sort key for a numbered section heading."""
    parts = _section_number(node.heading)
    if parts is None:
        return None
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in parts
    )


def _order_numbered_siblings(nodes: dict[str, TreeNode], root_id: str) -> list[TreeNode]:
    """Sort numbered siblings while preserving unnumbered sibling positions.

    PDFs sometimes place numbered subsections out of numeric order. This changes
    only traversal order: parent IDs, node IDs, bodies, and content stay intact.
    """
    for node in nodes.values():
        numbered = [
            child_id for child_id in node.children
            if _numbered_sibling_key(nodes[child_id]) is not None
        ]
        ordered_numbered = sorted(
            numbered,
            key=lambda child_id: _numbered_sibling_key(nodes[child_id]),
        )
        iterator = iter(ordered_numbered)
        node.children = [
            next(iterator) if child_id in numbered else child_id
            for child_id in node.children
        ]

    ordered: list[TreeNode] = []

    def visit(node_id: str) -> None:
        node = nodes[node_id]
        ordered.append(node)
        for child_id in node.children:
            visit(child_id)

    visit(root_id)
    return ordered


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: str | Path) -> list[TreeNode]:
    """Parse a PDF and return a flat list of TreeNodes forming a tree.

    The first node is always a synthetic root (level=0) whose children
    are the top-level sections.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    try:
        page_heights = {index: float(page.rect.height) for index, page in enumerate(doc)}
        lines: list[_TextLine] = []
        for page_number, page in enumerate(doc):
            lines.extend(_extract_lines(page, page_number))
        lines = _filter_page_furniture(lines, page_heights)

        text_lines = [line for line in lines if line.kind == "text"]
        thresholds = _classify_font_sizes([line.size for line in text_lines if line.size > 0])
        root = TreeNode(id=_make_id("ROOT", None, 0), heading="ROOT", level=0)
        if not thresholds:
            root.compute_hash()
            return [root]

        body_size = thresholds["body"]
        lines = _merge_heading_fragments(lines, body_size)
        nodes: dict[str, TreeNode] = {root.id: root}
        node_list: list[TreeNode] = [root]
        stack: list[tuple[int, str]] = [(0, root.id)]
        heading_counter: dict[tuple[str, str], int] = {}
        number_to_node: dict[tuple[str, ...], str] = {}
        current_node_id = root.id
        previous_body_line: _TextLine | None = None
        has_section = False

        def add_node(text: str, level: int, parent_id: str) -> TreeNode:
            key = (parent_id, text)
            occurrence = heading_counter.get(key, 0)
            heading_counter[key] = occurrence + 1
            node_id = _make_id(text, parent_id, occurrence)
            while node_id in nodes:
                occurrence += 1
                heading_counter[key] = occurrence + 1
                node_id = _make_id(text, parent_id, occurrence)
            node = TreeNode(id=node_id, heading=text, level=level, parent_id=parent_id)
            nodes[node_id] = node
            node_list.append(node)
            nodes[parent_id].children.append(node_id)
            return node

        def restore_stack(parent_id: str, child: TreeNode) -> list[tuple[int, str]]:
            ancestry: list[tuple[int, str]] = []
            cursor = nodes[parent_id]
            while cursor.level > 0:
                ancestry.append((cursor.level, cursor.id))
                cursor = nodes[cursor.parent_id]  # type: ignore[index]
            ancestry.reverse()
            return [(0, root.id), *ancestry, (child.level, child.id)]

        for line in lines:
            text = line.text.strip()
            if line.kind == "table":
                current = nodes[current_node_id]
                _append_body_line(current, line, previous_body_line)
                previous_body_line = line
                continue

            desired_level = _heading_level(line.size, line.flags, thresholds, text)
            if desired_level == 0:
                _append_body_line(nodes[current_node_id], line, previous_body_line)
                previous_body_line = line
                continue

            if not has_section and _is_document_title(
                line, body_size, page_heights.get(line.page_number, float("inf"))
            ):
                _append_body_line(root, line, previous_body_line)
                previous_body_line = line
                continue

            number_parts = _section_number(text)
            if number_parts is not None:
                parent_id = root.id
                for prefix_length in range(len(number_parts) - 1, 0, -1):
                    candidate = number_to_node.get(number_parts[:prefix_length])
                    if candidate is not None:
                        parent_id = candidate
                        break
                parent_level = nodes[parent_id].level
                # Clamp missing numbering levels to the next actual tree level.
                actual_level = min(len(number_parts), parent_level + 1)
                node = add_node(text, actual_level, parent_id)
                number_to_node[number_parts] = node.id
                stack = restore_stack(parent_id, node)
            else:
                while stack and stack[-1][0] >= desired_level:
                    stack.pop()
                if not stack:
                    stack = [(0, root.id)]
                parent_level, parent_id = stack[-1]
                actual_level = min(desired_level, parent_level + 1)
                node = add_node(text, actual_level, parent_id)
                stack.append((node.level, node.id))

            current_node_id = node.id
            previous_body_line = None
            has_section = True

        node_list = _order_numbered_siblings(nodes, root.id)
        for node in node_list:
            node.compute_hash()
        return node_list
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def tree_to_dict(nodes: list[TreeNode]) -> list[dict]:
    """Convert to a JSON-serializable list of dicts."""
    return [asdict(n) for n in nodes]


def save_tree(nodes: list[TreeNode], output_path: str | Path) -> Path:
    """Save the tree to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = tree_to_dict(nodes)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def load_tree(json_path: str | Path) -> list[TreeNode]:
    """Load a previously-saved tree from JSON."""
    json_path = Path(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    nodes = []
    for d in data:
        node = TreeNode(
            id=d["id"],
            heading=d["heading"],
            level=d["level"],
            body=d.get("body", ""),
            content_hash=d.get("content_hash", ""),
            parent_id=d.get("parent_id"),
            children=d.get("children", []),
        )
        nodes.append(node)
    return nodes


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parser.py <path-to-pdf> [output.json]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/ct200_tree.json"

    print(f"Parsing: {pdf_path}")
    nodes = parse_pdf(pdf_path)

    out = save_tree(nodes, output_path)
    print(f"Saved {len(nodes)} nodes -> {out}")

    # Quick summary
    levels = {}
    for n in nodes:
        levels[n.level] = levels.get(n.level, 0) + 1
    for lvl in sorted(levels):
        print(f"  Level {lvl}: {levels[lvl]} nodes")


if __name__ == "__main__":
    main()
