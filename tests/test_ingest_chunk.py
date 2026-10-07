"""Prose chunking tests: the boundary, not the count.

A table chunk that loses its header is obviously broken. A prose chunk that ends in the
wrong place is well-formed, retrievable, embeds fine, and silently wrong - so these tests
assert on where the boundaries LAND, against text constants declared here rather than
against a real PDF whose line breaks the reader cannot see. Every size decision below is
arithmetic stated in the docstring of the test that depends on it, and
``test_fixture_lengths_are_what_the_arithmetic_assumes`` pins the premises.

The corpus fact under test throughout is R2. ``def.hospital`` on page 4 of
star-comprehensive-2025 qualifies a facility two ways joined by the word "Or": registered
with the local authorities, OR meeting five listed criteria. A chunk carrying only the
second limb answers "not a hospital" for a facility that qualifies under the first, with a
correct-looking citation. So a definition is never split, and a clause is split only
between sub-clauses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arag.ingest.chunk import (
    HEADING_PATTERNS,
    MAX_PROSE_CHUNK_CHARS,
    RUNNING_LINE_MIN_PAGES,
    chunk_document,
)
from arag.ingest.chunk_types import ChunkKind
from arag.ingest.clause_index import ID_PATTERNS
from arag.ingest.normalise import NumericCorruptionError


class FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_text(self, option: str = "text") -> str:
        return self._text


class FakeDoc:
    """A ``pymupdf.Document`` stand-in whose pages are literal strings.

    The chunker accepts a Protocol instead of importing pymupdf precisely so these tests
    can state their input as lines. "The split landed between B and C" is only a
    meaningful assertion if the reader can see where B ends, and a real PDF hides that.
    """

    def __init__(self, *pages: str) -> None:
        self._pages = pages

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def __getitem__(self, index: int) -> FakePage:
        return FakePage(self._pages[index])


# --- R2 fixture: one clause, three lettered sub-clauses, exact lengths -----------------

HEAD = "5. Exclusions apply as follows:"  # 31 chars
SUB_A = "A. " + "a" * 57  # 3 + 57 = 60
SUB_B = "B. " + "b" * 57  # 60
SUB_C = "C. " + "c" * 57  # 60
# 31 + 60 + 60 + 60 + 3 newlines = 214
CLAUSE_PAGE = "\n".join([HEAD, SUB_A, SUB_B, SUB_C])

# The same clause with an over-budget middle limb: 3 + 297 = 300 chars.
BIG_SUB_B = "B. " + "b" * 297
BIG_CLAUSE_PAGE = "\n".join([HEAD, SUB_A, BIG_SUB_B, SUB_C])

# --- R2 fixture: a definition whose two qualifying limbs are joined by "Or" ------------

DEF_SECTION = "I - DEFINITIONS"
DEF_HOSPITAL = (
    "Hospital: A Hospital means any institution registered as a hospital with the local "
    "authorities under the Clinical Establishments Act, 2010 Or complies with all "
    "minimum criteria as under:"
)
LIMB_I = "i. has qualified nursing staff under its employment round the clock;"
LIMB_II = "ii. has at least 10 in-patient beds in towns having a population under 10,00,000;"
LIMB_III = "iii. has qualified medical practitioner(s) in charge round the clock;"
DEF_HOSPITALIZATION = (
    "Hospitalization: Hospitalization means admission in a Hospital for a minimum period "
    "of 24 consecutive hours."
)
DEFINITIONS_PAGE = "\n".join(
    [DEF_SECTION, DEF_HOSPITAL, LIMB_I, LIMB_II, LIMB_III, DEF_HOSPITALIZATION]
)

# --- refused-region fixture -----------------------------------------------------------

TABLE_PAGE = "\n".join(
    [
        "Annexure II - List of expenses not covered",
        "Sl. No.  Item",
        "1  BABY FOOD",
        "2  BABY UTILITIES CHARGES",
        "3  BEAUTY SERVICES",
    ]
)
# One declared table region on page 2, in the (page, bbox) shape the extractor produces.
TABLE_REGIONS: list[tuple[int, tuple[float, float, float, float]]] = [
    (2, (56.0, 120.0, 540.0, 400.0))
]


def test_fixture_lengths_are_what_the_arithmetic_assumes() -> None:
    """Pins the premises every packing calculation below is built on.

    Without this, a one-character edit to a fixture string turns the arithmetic in the
    other docstrings into fiction while the tests still pass for the wrong reason.
    """
    assert (len(HEAD), len(SUB_A), len(SUB_B), len(SUB_C)) == (31, 60, 60, 60)
    assert len(CLAUSE_PAGE) == 214
    assert len(BIG_SUB_B) == 300


class TestR2ClauseSplitting:
    """An over-budget clause splits between sub-clauses and nowhere else."""

    def test_oversized_clause_splits_at_subclause_boundaries(self) -> None:
        """max_chars=120 over HEAD(31) + A(60) + B(60) + C(60).

        Packing, where a piece's length is the sum of its lines plus one newline each:

            HEAD                31
            HEAD + A       31+1+60 =  92  <= 120, so A joins HEAD
            HEAD + A + B   92+1+60 = 153  >  120, so the piece closes and B starts one
            B + C          60+1+60 = 121  >  120 by exactly the joining newline

        giving three chunks of 92, 60 and 60 characters. The boundaries fall between A
        and B and between B and C - never inside a limb.
        """
        chunks, report = chunk_document("doc", FakeDoc(CLAUSE_PAGE), max_chars=120)

        assert [chunk.text for chunk in chunks] == [f"{HEAD}\n{SUB_A}", SUB_B, SUB_C]
        assert [chunk.char_len for chunk in chunks] == [92, 60, 60]
        # No text was lost or duplicated by the split, and every cut fell on a line start.
        assert "\n".join(chunk.text for chunk in chunks) == CLAUSE_PAGE
        assert [chunk.clause_id for chunk in chunks] == ["5", "5", "5"]
        assert [chunk.detail["part"] for chunk in chunks] == [0, 1, 2]
        assert all(chunk.detail["parts"] == 3 for chunk in chunks)
        assert all(chunk.kind is ChunkKind.PROSE for chunk in chunks)
        assert report.prose_chunks == 3
        assert report.oversized_split == 0

    def test_no_chunk_ever_begins_or_ends_inside_a_subclause(self) -> None:
        """The property the previous test's exact boundaries are an instance of.

        Every limb body is 57 identical characters, so a mid-sub-clause cut would show up
        as a chunk holding some but not all of one letter's run.
        """
        chunks, _ = chunk_document("doc", FakeDoc(CLAUSE_PAGE), max_chars=120)
        for letter in ("a", "b", "c"):
            holding = [chunk for chunk in chunks if letter * 2 in chunk.text]
            assert len(holding) == 1, f"limb {letter.upper()} appears in {len(holding)} chunks"
            assert holding[0].text.count(letter * 57) == 1
            assert letter * 58 not in holding[0].text

    def test_a_single_oversized_subclause_is_emitted_whole_and_counted(self) -> None:
        """max_chars=120 over HEAD(31) + A(60) + B(300) + C(60).

        B alone is 300 characters, so there is no split point that would bring it under
        budget. R2 wins over the budget: it is emitted whole and counted.

            HEAD + A         31+1+60 =  92  <= 120
            HEAD + A + B     92+1+300 = 393 >  120  -> piece closes, B starts its own
            B + C           300+1+60 = 361  >  120  -> B is emitted alone, at 300

        So three chunks of 92, 300 and 60, and oversized_split == 1.
        """
        chunks, report = chunk_document("doc", FakeDoc(BIG_CLAUSE_PAGE), max_chars=120)

        assert [chunk.char_len for chunk in chunks] == [92, 300, 60]
        assert chunks[1].text == BIG_SUB_B
        assert report.oversized_split == 1
        assert chunks[1].detail["oversized"] is True
        assert chunks[0].detail["oversized"] is False

    def test_a_clause_that_fits_is_not_split_at_all(self) -> None:
        """214 characters against the 1200 default: the sub-clause boundaries exist and
        are deliberately unused. Splitting a clause that fits would scatter A, B and C
        across three chunks for no reason.
        """
        chunks, report = chunk_document("doc", FakeDoc(CLAUSE_PAGE))
        assert MAX_PROSE_CHUNK_CHARS == 1200
        assert len(chunks) == 1
        assert chunks[0].text == CLAUSE_PAGE
        assert chunks[0].detail["parts"] == 1
        assert report.oversized_split == 0


class TestR2Definitions:
    """A definition is emitted whole, over budget or not."""

    def test_definition_with_two_or_limbs_is_never_split(self) -> None:
        """The def.hospital failure, reproduced at a budget that would force a split.

        The definition is 293 + 3 limbs = well over max_chars=80, and it contains three
        sub-clause markers (i., ii., iii.) that the clause splitter would happily cut at.
        It must still arrive as one chunk carrying BOTH qualifying limbs: the registration
        limb before "Or" and the criteria limb after it.
        """
        chunks, report = chunk_document("doc", FakeDoc(DEFINITIONS_PAGE), max_chars=80)

        hospital = [chunk for chunk in chunks if chunk.clause_id == "def.hospital"]
        assert len(hospital) == 1, "the definition was split"
        assert hospital[0].kind is ChunkKind.DEFINITION
        assert hospital[0].detail["parts"] == 1
        assert hospital[0].text == "\n".join([DEF_HOSPITAL, LIMB_I, LIMB_II, LIMB_III])
        # Both limbs of the disjunction, in one chunk.
        assert "registered as a hospital with the local" in hospital[0].text
        assert "Or complies with all" in hospital[0].text
        assert "iii. has qualified medical practitioner" in hospital[0].text
        # It is over budget, and the report says so rather than hiding it.
        assert hospital[0].char_len > 80
        assert hospital[0].detail["oversized"] is True
        assert report.oversized_split >= 1

    def test_each_defined_term_is_its_own_chunk(self) -> None:
        """One chunk per term, and the next term ends the previous one.

        Three blocks on this page: the section heading, Hospital (with its limbs), and
        Hospitalization. Hospitalization must not be swallowed by Hospital - and equally
        must not be cut out of it, which is what "one chunk per defined term, whole" means
        in both directions.
        """
        chunks, report = chunk_document("doc", FakeDoc(DEFINITIONS_PAGE), max_chars=80)

        assert [chunk.clause_id for chunk in chunks] == [
            None,
            "def.hospital",
            "def.hospitalization",
        ]
        assert [chunk.kind for chunk in chunks] == [
            ChunkKind.PROSE,
            ChunkKind.DEFINITION,
            ChunkKind.DEFINITION,
        ]
        assert chunks[0].text == DEF_SECTION
        assert chunks[2].text == DEF_HOSPITALIZATION
        assert report.definition_chunks == 2
        assert report.prose_chunks == 1

    def test_a_term_outside_the_definitions_section_stays_prose(self) -> None:
        """ "Note:" and "Important:" are list labels, not defined terms.

        Without the section gate, every line-start "Word:" in 197 pages of policy wording
        would become a citable ``def.*`` id, and a labeller would have thousands of
        identifiers to choose from of which a handful are real.
        """
        page = "\n".join(
            [
                "6. Claims procedure",
                "Note: intimation must reach the Company within 24 hours of admission.",
            ]
        )
        chunks, report = chunk_document("doc", FakeDoc(page))
        assert report.definition_chunks == 0
        assert [chunk.clause_id for chunk in chunks] == ["6"]
        assert "Note: intimation" in chunks[0].text

    def test_a_term_whose_id_will_not_canonicalise_stays_prose(self) -> None:
        """A 40-character term slugs to 40 characters; ``def.<slug>`` allows 39.

        ``CLAUSE_ID_PATTERN`` accepts ``def\\.[a-z][a-z0-9_]{1,38}`` - a slug of 2 to 39
        characters. The term below is exactly 40, so ``validate_clause_id`` rejects it.
        The line stays prose and is still indexed: an uncitable chunk is a gap in the
        citation surface, while dropping the text would be a gap in the corpus.
        """
        term = "Abcdefghij Abcdefghij Abcdefghij Abcdefg"
        assert len(term) == 40
        page = "\n".join([DEF_SECTION, f"{term}: this term is too long to canonicalise."])

        chunks, report = chunk_document("doc", FakeDoc(page))
        assert report.definition_chunks == 0
        assert any(term in chunk.text for chunk in chunks)


class TestClauseBoundaries:
    def test_a_clause_runs_to_the_next_heading_of_the_same_or_higher_level(self) -> None:
        """Clause 4 keeps 4.1 and 4.2; clause 5 starts a new chunk.

        The alternative - a new chunk at every heading regardless of level - emits "4.
        Coverage" as a chunk containing only its own title, and separates each sub-clause
        from the clause that governs it.
        """
        page = "\n".join(
            [
                "4. Coverage",
                "4.1. Inpatient care is covered.",
                "4.2. Day care treatment is covered.",
                "5. Exclusions",
                "5.1. War is excluded.",
            ]
        )
        chunks, _ = chunk_document("doc", FakeDoc(page))

        assert [chunk.clause_id for chunk in chunks] == ["4", "5"]
        assert "4.1. Inpatient care" in chunks[0].text
        assert "4.2. Day care" in chunks[0].text
        assert "5.1. War" in chunks[1].text

    def test_deeper_headings_become_split_points_for_the_clause_they_belong_to(self) -> None:
        """The other half of the same rule: 4.1 and 4.2 are not chunks, they are the
        boundaries clause 4 is cut at once it exceeds the budget.

        Lines are 4 + 3*31 = 97 characters plus 3 newlines = 100. At max_chars=70 the
        heading plus 4.1 (4 + 1 + 31 = 36) fits, adding 4.2 (36 + 1 + 31 = 68) also fits,
        adding 4.3 (68 + 1 + 31 = 100) does not - so the cut lands before 4.3.
        """
        sub = ["4.1. " + "x" * 26, "4.2. " + "y" * 26, "4.3. " + "z" * 26]
        assert [len(line) for line in sub] == [31, 31, 31]
        chunks, _ = chunk_document("doc", FakeDoc("\n".join(["4. C", *sub])), max_chars=70)

        assert [chunk.text for chunk in chunks] == ["\n".join(["4. C", sub[0], sub[1]]), sub[2]]
        assert [chunk.clause_id for chunk in chunks] == ["4", "4"]

    def test_a_running_page_number_does_not_open_a_clause(self) -> None:
        """Every page of this corpus carries "12 / 47" or "17 of 18", which is
        indistinguishable from a bare clause number. Treating it as a heading would give
        every document a clause named after each of its own page numbers - and would cut
        the clause that straddles the page break in half at the footer.
        """
        page_one = "\n".join(["7. Waiting periods apply as follows.", "12 / 47"])
        page_two = "\n".join(["The waiting period continues here.", "13 / 47"])
        chunks, _ = chunk_document("doc", FakeDoc(page_one, page_two))

        assert [chunk.clause_id for chunk in chunks] == ["7"]
        assert chunks[0].pages == (1, 2)
        # The footer text is still indexed - excluded from heading detection, not deleted.
        assert "12 / 47" in chunks[0].text

    def test_a_running_header_does_not_open_a_definition(self) -> None:
        """Measured: nivabupa-reassure2's page header is "Product Name: ReAssure 2.0 |
        Product UIN: ...", which matches the definition pattern on all 34 pages. Before
        this filter it produced five ``def.product_name`` chunks - all claiming the same
        clause id, each swallowing the real numbered definitions that followed it.

        Four pages here, so the threshold is max(RUNNING_LINE_MIN_PAGES=3, 4*0.5=2) = 3
        and a line on all four pages is furniture.

        The leading ``None`` in the expected clause ids is the page-1 header itself: it
        precedes every heading in the document, so it lands in an unheaded block. Text the
        chunker cannot attribute to a clause is still indexed, just not citable by clause.
        """
        header = "Product Name: TestCover | Product UIN: ABC123"
        doc = FakeDoc(
            "\n".join([header, "2. Definitions", "Hospital: A Hospital means an institution."]),
            "\n".join([header, "Grace Period: Grace Period means thirty days."]),
            "\n".join([header, "Injury: Injury means accidental physical bodily harm."]),
            "\n".join([header, "3. Benefits are payable as follows."]),
        )
        chunks, report = chunk_document("doc", doc)

        assert RUNNING_LINE_MIN_PAGES == 3
        clause_ids = [chunk.clause_id for chunk in chunks]
        assert "def.product_name" not in clause_ids
        assert clause_ids == [None, "2", "def.hospital", "def.grace_period", "def.injury", "3"]
        assert report.pages_uncovered == []
        # Furniture is barred from opening a clause, not removed from the text.
        assert any(header in chunk.text for chunk in chunks)

    def test_a_definition_records_the_numbered_section_that_defined_it(self) -> None:
        """``section_path`` is the only place a definition's numbered parent survives.

        "2. Definitions" is pushed as the enclosing clause; a term sits one level below it,
        so the term's chunk carries ("2",) while clause 2's own chunk carries ().
        """
        doc = FakeDoc(
            "\n".join(
                [
                    "2. Definitions",
                    "Hospital: A Hospital means an institution.",
                    "3. Benefits",
                ]
            )
        )
        chunks, _ = chunk_document("doc", doc)
        by_clause = {chunk.clause_id: chunk for chunk in chunks}
        assert by_clause["2"].meta.section_path == ()
        assert by_clause["def.hospital"].meta.section_path == ("2",)
        assert by_clause["3"].meta.section_path == ()

    def test_only_line_anchored_id_patterns_are_used_as_boundaries(self) -> None:
        """``compound`` ("II - Section 6 m.") and ``excl_code`` ("Code Excl 02") are
        *searched* rather than line-anchored in clause_index, because both trail a clause
        title mid-line. Opening a chunk at a mid-line match would start a clause in the
        middle of a sentence, so they stay identifiers and are not boundaries.
        """
        assert set(HEADING_PATTERNS) <= set(ID_PATTERNS)
        assert "compound" not in HEADING_PATTERNS
        assert "excl_code" not in HEADING_PATTERNS


class TestRefusedRegionFallback:
    """A table region is not a hole in the prose (R2).

    Two tables in irdai-master-circular-2024 are refused for having no header. If the
    prose chunker skipped table regions - the obvious "don't index it twice" saving - that
    content would be in neither index, and the gap would look like a relevance problem.
    """

    def _doc(self) -> FakeDoc:
        return FakeDoc(
            "1. Coverage of hospitalisation expenses is subject to the annexure below.",
            TABLE_PAGE,
            "2. Exclusions are listed in this clause.",
        )

    def test_a_page_carrying_a_declared_table_still_produces_prose(self) -> None:
        chunks, report = chunk_document("doc", self._doc(), table_regions=TABLE_REGIONS)

        assert report.pages_uncovered == []
        assert 2 in report.pages_covered
        covering = [chunk for chunk in chunks if 2 in chunk.pages]
        assert covering, "the table page produced no prose chunk"
        assert any("BABY UTILITIES CHARGES" in chunk.text for chunk in covering)

    def test_chunks_on_a_declared_region_are_flagged_and_others_are_not(self) -> None:
        chunks, _ = chunk_document("doc", self._doc(), table_regions=TABLE_REGIONS)
        for chunk in chunks:
            assert chunk.detail["covers_table_region"] is (2 in chunk.pages)

    def test_declaring_a_region_never_changes_the_text_that_is_indexed(self) -> None:
        """The guarantee stated as an identity: table_regions is an input the chunker
        checks coverage against, never one it subtracts.
        """
        with_regions, report_with = chunk_document("doc", self._doc(), table_regions=TABLE_REGIONS)
        without, report_without = chunk_document("doc", self._doc())

        assert [chunk.text for chunk in with_regions] == [chunk.text for chunk in without]
        assert report_with.chars_out == report_without.chars_out
        assert report_without.pages_uncovered == []


class TestNormalisation:
    def test_a_ligature_does_not_survive_into_a_chunk(self) -> None:
        """Measured: 269 ligature codepoints in this corpus. "bene\\ufb01t" and "benefit"
        share no BM25 token, so an unnormalised chunk is unretrievable by the most-queried
        word in the domain - and nothing errors.
        """
        chunks, _ = chunk_document(
            "doc", FakeDoc("1. The beneﬁt is payable after the waiting period.")
        )
        assert len(chunks) == 1
        assert "ﬁ" not in chunks[0].text
        assert "benefit is payable" in chunks[0].text

    def test_a_numeric_corruption_stops_chunking(self) -> None:
        """The N2 hazard, and the reason strict_numerics is on.

        The superscript here is NOT preceded by a digit, so the conservative footnote
        stripper leaves it in place and NFKC folds it into the value: 5,00,000 becomes
        15,00,000. Chunking must stop rather than index a sub-limit that is wrong by 10x.
        """
        doc = FakeDoc("1. Sum insured of Rs ¹5,00,000 applies to this benefit.")
        with pytest.raises(NumericCorruptionError):
            chunk_document("doc", doc)


class TestChunkReport:
    def test_char_accounting_is_exact(self) -> None:
        """chars_in is the 214 characters of the page; chars_out is 212.

        The two newlines that were the split points are consumed by the split (214 - 2),
        which is the whole difference. A wider gap than that means text was dropped, which
        is what this field exists to make visible.
        """
        _, report = chunk_document("doc", FakeDoc(CLAUSE_PAGE), max_chars=120)
        assert report.chars_in == 214
        assert report.chars_out == 212
        assert report.chunks == 3
        assert report.pages_covered == {1}
        assert report.ok

    def test_a_page_with_no_text_is_reported_uncovered(self) -> None:
        """Proves pages_uncovered is not vacuously empty.

        Without this, the corpus-wide "pages_uncovered == []" assertion would pass just as
        well against a field that is hardcoded empty.
        """
        doc = FakeDoc("1. First page.", "", "2. Third page.")
        _, report = chunk_document("doc", doc)

        assert report.pages_uncovered == [2]
        assert report.pages_covered == {1, 3}
        assert not report.ok

    def test_chunk_ids_are_unique_and_carry_their_page(self) -> None:
        chunks, _ = chunk_document("doc", FakeDoc(CLAUSE_PAGE), max_chars=120)
        ids = [chunk.chunk_id for chunk in chunks]
        assert ids == ["doc:p1:s0", "doc:p1:s1", "doc:p1:s2"]
        assert len(set(ids)) == len(ids)

    def test_every_chunk_is_citable_by_document_and_page(self) -> None:
        """A chunk with no span is retrievable and unscoreable - present in the index,
        absent from every retrieval metric, and erroring nowhere.
        """
        chunks, _ = chunk_document("doc", FakeDoc(DEFINITIONS_PAGE), max_chars=80)
        for chunk in chunks:
            assert chunk.span.document_id == "doc"
            assert chunk.span.page == chunk.pages[0]
            assert chunk.meta.is_table is False


@pytest.mark.slow
class TestRealCorpus:
    """The claims that only 197 real pages can support.

    Marked slow: a full pass over six PDFs is seconds standalone and minutes under
    coverage instrumentation, so it runs in the nightly rather than the PR gate.
    """

    @staticmethod
    def _regions(
        source_id: str, doc: object
    ) -> list[tuple[int, tuple[float, float, float, float]]]:
        """Real table bboxes, from the table extractor, in the shape the API takes.

        Costs a full ``find_tables`` pass - 12 to 16 seconds per document, roughly 30x
        what chunking the same document costs - so it is paid only where the regions carry
        signal, which is the document with the refused tables.
        """
        from arag.ingest.tables import extract_fragments

        regions: list[tuple[int, tuple[float, float, float, float]]] = []
        for index in range(doc.page_count):  # type: ignore[attr-defined]
            for fragment in extract_fragments(source_id, doc[index], index + 1):  # type: ignore[index]
                regions.append((fragment.page, fragment.bbox))
        return regions

    def test_no_page_of_any_document_is_left_uncovered(self, repo_root: Path) -> None:
        """The R2 corpus-level requirement, plus the invariants that make it meaningful.

        A page can be "covered" trivially by a chunk holding only its footer, so this also
        asserts the character ratio: at least 99% of every document's extracted characters
        reach a chunk. Measured today, chars_out/chars_in is 0.994 to 1.000 - the shortfall
        is the leading and trailing whitespace stripped off each chunk.

        No ``table_regions`` here, deliberately: declaring a region provably does not
        change what is indexed (``test_declaring_a_region_never_changes_the_text_that_is_
        indexed``), so a second table-extraction pass over 197 pages would add two minutes
        to the nightly for no extra signal. The regions earn their cost in the refused-
        table test below, which is the only place they can tell us anything.
        """
        from arag.ingest.manifest import Manifest

        pymupdf = pytest.importorskip("pymupdf")
        raw = repo_root / "data" / "raw"
        if not raw.exists() or not any(raw.glob("*.pdf")):
            pytest.skip("corpus not fetched")

        # Only the PRODUCTION sources - the adversarial fixture PDF lives in data/raw too
        # but is not part of the 197-page corpus, and ingesting it here would both break the
        # page census and chunk a poisoned document in a test about real coverage.
        manifest = Manifest.load(repo_root / "data" / "manifest" / "sources.jsonl")
        pdfs = sorted(raw / f"{s.id}.pdf" for s in manifest.production_sources)
        assert len(pdfs) == 6, "the manifest declares six production sources"
        if not all(p.exists() for p in pdfs):
            pytest.skip("production corpus not fully fetched")

        seen_ids: set[str] = set()
        total_pages = 0
        for pdf in pdfs:
            doc = pymupdf.open(pdf)
            with doc:
                chunks, report = chunk_document(pdf.stem, doc)
                total_pages += doc.page_count

            assert report.pages_uncovered == [], (
                f"{pdf.stem}: pages {report.pages_uncovered} produced no prose chunk. A "
                "page in neither the table index nor the prose index is a silent gap."
            )
            assert report.chars_out >= report.chars_in * 0.99, (
                f"{pdf.stem}: only {report.chars_out}/{report.chars_in} characters reached a chunk"
            )
            # Over-budget chunks are allowed by R2 but must all be accounted for.
            assert (
                sum(1 for chunk in chunks if chunk.char_len > MAX_PROSE_CHUNK_CHARS)
                == report.oversized_split
            )
            # R2: no definition is ever a fragment of one.
            assert all(
                chunk.detail["parts"] == 1 for chunk in chunks if chunk.kind is ChunkKind.DEFINITION
            )
            ids = {chunk.chunk_id for chunk in chunks}
            assert len(ids) == len(chunks), f"{pdf.stem}: duplicate chunk ids"
            assert not (ids & seen_ids), f"{pdf.stem}: chunk ids collide with another document"
            seen_ids |= ids

        assert total_pages == 197, "the corpus is 197 pages"
        assert len(seen_ids) >= total_pages, "full coverage implies at least one chunk per page"

    def test_the_refused_table_pages_are_covered_by_prose(self, repo_root: Path) -> None:
        """Measured: the tables on pages 10 and 11 of irdai-master-circular-2024 have no
        detectable header and are refused by the table chunker. Those two pages are the
        exact place where a "skip detected tables" prose chunker would lose content, so
        they are asserted individually rather than left to the corpus-wide check.
        """
        pymupdf = pytest.importorskip("pymupdf")
        pdf = repo_root / "data" / "raw" / "irdai-master-circular-2024.pdf"
        if not pdf.exists():
            pytest.skip("corpus not fetched")

        doc = pymupdf.open(pdf)
        with doc:
            regions = self._regions(pdf.stem, doc)
            chunks, report = chunk_document(pdf.stem, doc, table_regions=regions)

        assert {10, 11} <= report.pages_covered
        for page in (10, 11):
            covering = [chunk for chunk in chunks if page in chunk.pages]
            assert covering, f"page {page} carries a refused table and no prose chunk"
            # The extractor did detect a table region on these pages...
            assert all(chunk.detail["covers_table_region"] for chunk in covering)
            # ...and the prose fallback still carries real text off them.
            assert sum(chunk.char_len for chunk in covering) > 500

    def test_def_hospital_survives_a_budget_that_would_split_it(self, repo_root: Path) -> None:
        """R2 on the document that motivated it, at max_chars=400 against a 921-character
        definition - so the definition is kept whole by the rule, not by luck.
        """
        pymupdf = pytest.importorskip("pymupdf")
        pdf = repo_root / "data" / "raw" / "star-comprehensive-2025.pdf"
        if not pdf.exists():
            pytest.skip("corpus not fetched")

        doc = pymupdf.open(pdf)
        with doc:
            chunks, _ = chunk_document(pdf.stem, doc, max_chars=400)

        hospital = [chunk for chunk in chunks if chunk.clause_id == "def.hospital"]
        assert len(hospital) == 1, "def.hospital was split"
        assert hospital[0].pages == (4,)
        assert hospital[0].char_len > 400
        assert hospital[0].kind is ChunkKind.DEFINITION
        # Both qualifying limbs, in the one chunk.
        assert "registered as a hospital with the local" in hospital[0].text
        assert "Or" in hospital[0].text
        assert "complies with all minimum criteria" in hospital[0].text
        assert "Maintains daily records of patients" in hospital[0].text
