"""
test_parser.py — Tests for the CT-200 PDF parser.

Covers:
  1. Duplicate headings produce distinct node IDs with correct parents.
  2. Parent-child relationships are consistent (every child's parent_id exists,
     every parent's children list references existing nodes).
  3. Malformed / empty input is handled gracefully.
  4. Content hash is deterministic and changes when body text changes.
  5. Section-numbering level inference works correctly.
"""

import hashlib
import fitz
import pytest
import parser as parser_module
from parser import (
    TreeNode,
    _TextLine,
    _make_id,
    _infer_level_from_numbering,
    parse_pdf,
    tree_to_dict,
    save_tree,
    load_tree,
    is_ct200_manual,
)
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_id_index(nodes):
    """Return {node.id: node} for quick lookup."""
    return {n.id: n for n in nodes}


def _make_pdf(tmp_path, draw_page):
    document = fitz.open()
    page = document.new_page()
    draw_page(page)
    path = tmp_path / "synthetic_manual.pdf"
    document.save(path)
    document.close()
    return path


# ---------------------------------------------------------------------------
# Test 1 — Duplicate headings
# ---------------------------------------------------------------------------

class TestDuplicateHeadings:
    """Two sections with the same heading under the same parent must get
    distinct IDs."""

    def test_make_id_different_occurrences(self):
        """_make_id with different occurrence counters → different IDs."""
        id_a = _make_id("Safety Warnings", "parent_1", 0)
        id_b = _make_id("Safety Warnings", "parent_1", 1)
        assert id_a != id_b, "Duplicate headings must produce distinct IDs"

    def test_make_id_same_heading_different_parents(self):
        """Same heading under different parents → different IDs."""
        id_a = _make_id("Overview", "section_1", 0)
        id_b = _make_id("Overview", "section_2", 0)
        assert id_a != id_b

    def test_make_id_deterministic(self):
        """Same inputs always produce the same ID."""
        id1 = _make_id("Foo", "bar", 0)
        id2 = _make_id("Foo", "bar", 0)
        assert id1 == id2


# ---------------------------------------------------------------------------
# Test 2 — Parent-child consistency
# ---------------------------------------------------------------------------

class TestParentChildRelationships:
    """If we can parse the real PDF, verify the tree is internally consistent."""

    @pytest.fixture
    def real_tree(self, monkeypatch):
        """Parse extractable fixture text without requiring system Tesseract."""
        monkeypatch.setattr(parser_module, "_should_ocr_page", lambda page, lines: False)
        candidates = [
            Path("data/ct200_manual.pdf"),
            Path("data/ct200_manual_v1.pdf"),
        ]
        for p in candidates:
            if p.exists():
                return parse_pdf(p)
        pytest.skip("CT-200 PDF not found in data/ — skipping real-PDF tests")

    def test_every_child_has_valid_parent(self, real_tree):
        idx = _build_id_index(real_tree)
        for node in real_tree:
            if node.parent_id is not None:
                assert node.parent_id in idx, (
                    f"Node {node.id!r} ({node.heading!r}) references "
                    f"non-existent parent {node.parent_id!r}"
                )

    def test_every_parent_lists_child(self, real_tree):
        idx = _build_id_index(real_tree)
        for node in real_tree:
            if node.parent_id is not None:
                parent = idx[node.parent_id]
                assert node.id in parent.children, (
                    f"Node {node.id!r} ({node.heading!r}) not in parent's children list"
                )

    def test_root_has_no_parent(self, real_tree):
        root = real_tree[0]
        assert root.parent_id is None
        assert root.level == 0

    def test_at_least_some_nodes_parsed(self, real_tree):
        """Sanity: the parser should find a meaningful number of nodes."""
        assert len(real_tree) >= 5, (
            f"Only {len(real_tree)} nodes found — parser may be broken"
        )


# ---------------------------------------------------------------------------
# Test 3 — Malformed / empty input
# ---------------------------------------------------------------------------

class TestMalformedInput:
    """The parser must not crash on bad input."""

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_pdf("does_not_exist.pdf")

    def test_non_pdf_file(self, tmp_path):
        """A file that isn't a valid PDF should raise a fitz error."""
        bad = tmp_path / "fake.pdf"
        bad.write_text("this is not a pdf")
        with pytest.raises(Exception):
            parse_pdf(bad)

    def test_empty_file(self, tmp_path):
        """A zero-byte file should raise."""
        empty = tmp_path / "empty.pdf"
        empty.write_bytes(b"")
        with pytest.raises(Exception):
            parse_pdf(empty)


# ---------------------------------------------------------------------------
# Test 4 — Content hashing
# ---------------------------------------------------------------------------

class TestContentHash:
    def test_hash_deterministic(self):
        node = TreeNode(id="test", heading="Test", level=1, body="hello world")
        h1 = node.compute_hash()
        h2 = node.compute_hash()
        assert h1 == h2

    def test_hash_changes_with_body(self):
        node_a = TreeNode(id="a", heading="X", level=1, body="version 1")
        node_b = TreeNode(id="b", heading="X", level=1, body="version 2")
        node_a.compute_hash()
        node_b.compute_hash()
        assert node_a.content_hash != node_b.content_hash

    def test_hash_is_sha256(self):
        node = TreeNode(id="t", heading="T", level=1, body="test")
        h = node.compute_hash()
        expected = hashlib.sha256(b"test").hexdigest()
        assert h == expected


# ---------------------------------------------------------------------------
# Test 5 — Section numbering inference
# ---------------------------------------------------------------------------

class TestSectionNumbering:
    def test_simple_number(self):
        result = _infer_level_from_numbering("3 Overview")
        assert result is not None
        level, _ = result
        assert level == 1

    def test_dotted_number(self):
        result = _infer_level_from_numbering("3.2 Pressure Limits")
        assert result is not None
        level, _ = result
        assert level == 2

    def test_deep_number(self):
        result = _infer_level_from_numbering("3.2.1 Calibration Procedure")
        assert result is not None
        level, _ = result
        assert level == 3

    def test_no_number(self):
        result = _infer_level_from_numbering("Just a sentence of text.")
        assert result is None

    def test_appendix(self):
        result = _infer_level_from_numbering("Appendix A: Error Codes")
        assert result is not None
        level, _ = result
        assert level == 1


# ---------------------------------------------------------------------------
# Test 6 — Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_round_trip(self, tmp_path):
        nodes = [
            TreeNode(id="root", heading="ROOT", level=0, content_hash="aaa"),
            TreeNode(id="ch1", heading="Ch1", level=1, parent_id="root",
                     body="body text", content_hash="bbb"),
        ]
        nodes[0].children = ["ch1"]

        out = tmp_path / "tree.json"
        save_tree(nodes, out)
        loaded = load_tree(out)

        assert len(loaded) == 2
        assert loaded[0].id == "root"
        assert loaded[1].parent_id == "root"
        assert loaded[0].children == ["ch1"]
        assert loaded[1].body == "body text"


class TestImprovedExtraction:
    def test_available_ct200_manual_has_clean_title_hierarchy_and_table(self, monkeypatch):
        manual = Path("data/ct200_manual.pdf")
        if not manual.exists():
            pytest.skip("CT-200 manual is not available in data/")
        monkeypatch.setattr(parser_module, "_should_ocr_page", lambda page, lines: False)
        nodes = parse_pdf(manual)
        root = nodes[0]
        headings = [node.heading for node in nodes]
        specs = next(node for node in nodes if node.heading == "2.1 General Specifications")

        assert root.body.startswith("CardioTrack CT-200 Home Blood Pressure Monitor")
        assert "Parameter" not in headings
        assert "Value" not in headings
        assert max(node.level for node in nodes) <= 3
        assert "| Measurement method | Oscillometric |" in specs.body
        assert all(node.content_hash for node in nodes)

    def test_wrapped_heading_and_paragraphs_and_bullets_are_preserved(self, tmp_path):
        def draw(page):
            page.insert_text(
                (40, 55),
                "1. Key Terms (if any of this is new to you, that's fine —",
                fontsize=15,
                fontname="hebo",
            )
            page.insert_text((40, 73), "quick primer)", fontsize=15, fontname="hebo")
            page.insert_text((40, 110), "First paragraph continues", fontsize=11)
            page.insert_text((40, 123), "on the next visual line", fontsize=11)
            page.insert_text((40, 154), "Second paragraph starts here", fontsize=11)
            page.insert_text((40, 181), "- First bullet item", fontsize=11)
            page.insert_text((40, 196), "- Second bullet item", fontsize=11)
            page.insert_text((40, 225), "Additional body text", fontsize=11)

        nodes = parse_pdf(_make_pdf(tmp_path, draw))
        section = next(node for node in nodes if node.level > 0)
        assert section.heading.startswith("1. Key Terms (if any of this is new to you, that's fine")
        assert section.heading.endswith("quick primer)")
        assert "First paragraph continues on the next visual line" in section.body
        assert "Second paragraph starts here\n\n- First bullet item\n- Second bullet item" in section.body
        assert section.content_hash == hashlib.sha256(section.body.encode("utf-8")).hexdigest()

    def test_irregular_numbering_is_clamped_and_duplicate_headings_stay_distinct(self, tmp_path):
        def draw(page):
            page.insert_text((40, 45), "1. Overview", fontsize=18, fontname="hebo")
            page.insert_text((40, 78), "1.1 Details", fontsize=14, fontname="hebo")
            page.insert_text((40, 110), "1.1.1.1 Deep A", fontsize=11, fontname="hebo")
            page.insert_text((40, 140), "deep A content line one", fontsize=11)
            page.insert_text((40, 153), "deep A content line two", fontsize=11)
            page.insert_text((40, 180), "1.1.1.2 Deep B", fontsize=11, fontname="hebo")
            page.insert_text((40, 210), "deep B content line one", fontsize=11)
            page.insert_text((40, 223), "deep B content line two", fontsize=11)
            page.insert_text((40, 250), "1.2 Repeated", fontsize=14, fontname="hebo")
            page.insert_text((40, 275), "first repeated content", fontsize=11)
            page.insert_text((40, 300), "1.2 Repeated", fontsize=14, fontname="hebo")
            page.insert_text((40, 325), "second repeated content", fontsize=11)
            page.insert_text((40, 360), "1.3 Empty Section", fontsize=14, fontname="hebo")

        nodes = parse_pdf(_make_pdf(tmp_path, draw))
        by_heading = {}
        for node in nodes:
            by_heading.setdefault(node.heading, []).append(node)
        deep_nodes = by_heading["1.1.1.1 Deep A"] + by_heading["1.1.1.2 Deep B"]
        assert [node.level for node in deep_nodes] == [3, 3]
        assert deep_nodes[0].parent_id == deep_nodes[1].parent_id
        duplicates = by_heading["1.2 Repeated"]
        assert len(duplicates) == 2
        assert duplicates[0].id != duplicates[1].id
        assert by_heading["1.3 Empty Section"][0].body == ""
        assert by_heading["1.3 Empty Section"][0].content_hash == hashlib.sha256(b"").hexdigest()

    def test_numbered_siblings_are_sorted_without_moving_their_content(self, tmp_path):
        def draw(page):
            page.insert_text((40, 45), "1. Device Operation", fontsize=18, fontname="hebo")
            page.insert_text((40, 80), "1.2 Second Step", fontsize=14, fontname="hebo")
            page.insert_text((40, 105), "second step content", fontsize=11)
            page.insert_text((40, 140), "1.1 First Step", fontsize=14, fontname="hebo")
            page.insert_text((40, 165), "first step content", fontsize=11)
            page.insert_text((40, 200), "1.3 Third Step", fontsize=14, fontname="hebo")
            page.insert_text((40, 225), "third step content", fontsize=11)

        nodes = parse_pdf(_make_pdf(tmp_path, draw))
        index = _build_id_index(nodes)
        parent = next(node for node in nodes if node.heading == "1. Device Operation")
        children = [index[child_id] for child_id in parent.children]

        assert [node.heading for node in children] == [
            "1.1 First Step", "1.2 Second Step", "1.3 Third Step",
        ]
        assert [node.body for node in children] == [
            "first step content", "second step content", "third step content",
        ]
        assert all(node.parent_id == parent.id for node in children)

    def test_browser_interface_artifacts_are_filtered(self, tmp_path):
        def draw(page):
            page.insert_text((35, 25), "Cette version du navigateur n'est plus compatible", fontsize=10)
            page.insert_text((35, 42), "Fichier Édition Affichage Outils Aide", fontsize=10)
            page.insert_text((40, 85), "1. Device Overview", fontsize=18, fontname="hebo")
            page.insert_text((40, 120), "Actual manual content", fontsize=11)
            page.insert_text((40, 133), "More actual content", fontsize=11)

        nodes = parse_pdf(_make_pdf(tmp_path, draw))
        all_text = " ".join(node.heading + " " + node.body for node in nodes)
        assert "n'est plus compatible" not in all_text
        assert "Fichier" not in all_text
        assert "Actual manual content" in all_text

    def test_detected_table_stays_in_section_body(self, tmp_path):
        def draw(page):
            page.insert_text((40, 45), "1. Specifications", fontsize=18, fontname="hebo")
            for x in (40, 200, 360):
                page.draw_line((x, 85), (x, 145), width=1)
            for y in (85, 105, 125, 145):
                page.draw_line((40, y), (360, y), width=1)
            page.insert_text((48, 100), "Parameter", fontsize=11, fontname="hebo")
            page.insert_text((208, 100), "Value", fontsize=11, fontname="hebo")
            page.insert_text((48, 120), "Pressure range", fontsize=11)
            page.insert_text((208, 120), "0-299 mmHg", fontsize=11)
            page.insert_text((48, 140), "Pulse range", fontsize=11)
            page.insert_text((208, 140), "40-199 bpm", fontsize=11)
            page.insert_text((40, 190), "Additional paragraph text", fontsize=11)
            page.insert_text((40, 203), "keeps body font classification stable", fontsize=11)

        nodes = parse_pdf(_make_pdf(tmp_path, draw))
        headings = [node.heading for node in nodes]
        section = next(node for node in nodes if node.heading == "1. Specifications")
        assert "Parameter" not in headings
        assert "Value" not in headings
        assert "| Pressure range | 0-299 mmHg |" in section.body
        assert "| Pulse range | 40-199 bpm |" in section.body

    def test_ct200_manual_identity_rejects_assignment_text(self):
        manual_text = (
            "CardioTrack CT-200 Home Blood Pressure Monitor. "
            "1. Device Overview. 1.1 Intended Use."
        )
        assignment_text = (
            "AI Engineering Internship Assignment - Tri9T AI. "
            "CardioTrack CT-200 Home Blood Pressure Monitor. "
            "Build a document tree API. Key Terms and Expected Tech Stack."
        )
        assert is_ct200_manual(manual_text)
        assert not is_ct200_manual(assignment_text)


class TestOCRFallback:
    def test_sufficient_embedded_text_does_not_invoke_ocr(self, tmp_path, monkeypatch):
        def draw(page):
            page.insert_text(
                (40, 80),
                "This ordinary text page already contains enough extractable content.",
                fontsize=11,
            )

        def unexpected_ocr(*args, **kwargs):
            pytest.fail("OCR should not run for a page with sufficient embedded text")

        monkeypatch.setattr(parser_module, "_ocr_page_lines", unexpected_ocr)
        nodes = parse_pdf(_make_pdf(tmp_path, draw))
        assert "ordinary text page" in nodes[0].body

    def test_insufficient_text_uses_ocr_lines_in_existing_tree_pipeline(self, tmp_path, monkeypatch):
        def draw(page):
            page.insert_text((40, 80), "weak", fontsize=11)

        observed = []

        def fake_ocr(page, page_number):
            observed.append(page_number)
            return [
                _TextLine("1. OCR Section", 18, 0, (40, 40, 180, 60), page_number),
                _TextLine("Recognized body text from the scanned page", 11, 0, (40, 85, 260, 100), page_number),
                _TextLine("A second line keeps the body font classification stable", 11, 0, (40, 105, 280, 120), page_number),
            ]

        monkeypatch.setattr(parser_module, "_ocr_page_lines", fake_ocr)
        nodes = parse_pdf(_make_pdf(tmp_path, draw))
        section = next(node for node in nodes if node.heading == "1. OCR Section")

        assert observed == [0]
        assert "Recognized body text" in section.body
        assert "weak" not in " ".join(node.heading + " " + node.body for node in nodes)
        assert all(node.content_hash == hashlib.sha256(node.body.encode("utf-8")).hexdigest() for node in nodes)

    def test_ocr_words_become_page_numbered_text_lines_with_scaled_boxes(self, monkeypatch):
        import pytesseract

        data = {
            "text": ["CT-200", "Manual", "Section"],
            "left": [20, 80, 20],
            "top": [30, 30, 60],
            "width": [40, 50, 70],
            "height": [20, 20, 20],
            "block_num": [1, 1, 1],
            "par_num": [1, 1, 1],
            "line_num": [1, 1, 2],
        }
        monkeypatch.setenv("TESSERACT_CMD", "mock-tesseract")
        monkeypatch.setattr(parser_module.shutil, "which", lambda command: "mock-tesseract.exe")
        monkeypatch.setattr(pytesseract, "image_to_data", lambda image, **kwargs: data)

        document = fitz.open()
        try:
            page = document.new_page(width=200, height=300)
            lines = parser_module._ocr_page_lines(page, 4)
        finally:
            document.close()

        assert [line.text for line in lines] == ["CT-200 Manual", "Section"]
        assert all(isinstance(line, _TextLine) and line.page_number == 4 for line in lines)
        assert lines[0].bbox == pytest.approx((10, 15, 65, 25))
        assert lines[0].size == pytest.approx(8)

    def test_missing_tesseract_raises_clear_error(self, tmp_path, monkeypatch):
        def draw(page):
            page.insert_text((40, 80), "short", fontsize=11)

        monkeypatch.setenv("TESSERACT_CMD", str(tmp_path / "missing-tesseract.exe"))
        monkeypatch.setattr(parser_module.shutil, "which", lambda command: None)

        with pytest.raises(RuntimeError, match="Tesseract OCR executable was not found.*TESSERACT_CMD"):
            parse_pdf(_make_pdf(tmp_path, draw))

    def test_tesseract_runtime_error_is_reported_with_page_number(self, tmp_path, monkeypatch):
        import pytesseract

        def draw(page):
            page.insert_text((40, 80), "short", fontsize=11)

        monkeypatch.setenv("TESSERACT_CMD", "mock-tesseract")
        monkeypatch.setattr(
            parser_module.shutil,
            "which",
            lambda command: str(tmp_path / "mock-tesseract.exe"),
        )

        def fail_ocr(*args, **kwargs):
            raise RuntimeError("mocked engine failure")

        monkeypatch.setattr(pytesseract, "image_to_data", fail_ocr)
        with pytest.raises(RuntimeError, match="Tesseract OCR failed on page 1: mocked engine failure"):
            parse_pdf(_make_pdf(tmp_path, draw))

    def test_empty_ocr_result_does_not_silently_return_a_tree(self, tmp_path, monkeypatch):
        def draw(page):
            page.insert_text((40, 80), "short", fontsize=11)

        monkeypatch.setattr(parser_module, "_ocr_page_lines", lambda page, page_number: [])
        with pytest.raises(RuntimeError, match="Tesseract OCR returned no text for page 1"):
            parse_pdf(_make_pdf(tmp_path, draw))
