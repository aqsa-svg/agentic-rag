"""Manifest tests.

Three fields on a source are retrieval *logic*, not bookkeeping — authority rank, the
supersession graph, and the mirror chain — so the invariants that keep them meaningful are
enforced by tests rather than by care.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from arag.ingest.manifest import (
    RANK_INSURER,
    RANK_REGULATOR,
    Manifest,
    Source,
    SourceKind,
)


def source(**overrides: object) -> Source:
    base: dict[str, object] = {
        "id": "s1",
        "url": "https://example.test/a.pdf",
        "kind": SourceKind.POLICY_WORDING,
        "publisher": "Example Insurer",
        "authority_rank": RANK_INSURER,
    }
    return Source.model_validate(base | overrides)


class TestSource:
    def test_valid(self) -> None:
        assert source().authority_rank == RANK_INSURER

    def test_rejects_unknown_authority_rank(self) -> None:
        with pytest.raises(ValidationError, match="authority_rank"):
            source(authority_rank=7)

    def test_rejects_brochures_outright(self) -> None:
        """Brochures assert cover that the policy wording does not grant.

        Ingesting one would let the system answer a coverage question from marketing copy,
        which is the single most damaging thing that could enter this corpus.
        """
        with pytest.raises(ValidationError, match="marketing copy"):
            source(kind=SourceKind.BROCHURE)

    def test_rejects_self_supersession(self) -> None:
        with pytest.raises(ValidationError, match="cannot supersede itself"):
            source(superseded_by="s1")

    def test_urls_puts_primary_before_mirrors(self) -> None:
        s = source(mirror_urls=["https://mirror.test/a.pdf"])
        assert s.urls == ("https://example.test/a.pdf", "https://mirror.test/a.pdf")

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            source(authorty_rank=1)


class TestManifestGraph:
    def test_duplicate_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate source ids"):
            Manifest(sources=(source(id="a"), source(id="a")))

    def test_dangling_supersedes_edge_rejected(self) -> None:
        """A dangling edge silently disables the version filter for that document, which is
        the exact failure the edge exists to prevent.
        """
        with pytest.raises(ValidationError, match="supersedes unknown source"):
            Manifest(sources=(source(id="a", supersedes=["nope"]),))

    def test_dangling_superseded_by_rejected(self) -> None:
        with pytest.raises(ValidationError, match="superseded_by unknown source"):
            Manifest(sources=(source(id="a", superseded_by="nope"),))

    def test_dangling_parent_rejected(self) -> None:
        with pytest.raises(ValidationError, match="parent unknown"):
            Manifest(sources=(source(id="a", parent_document_id="nope"),))

    def test_superseded_ids_collects_both_directions(self) -> None:
        m = Manifest(
            sources=(
                source(id="new", supersedes=["old"]),
                source(id="old", superseded_by="new", effective_to=date(2025, 3, 31)),
            )
        )
        assert m.superseded_ids() == {"old"}

    def test_a_supersession_edge_without_a_date_is_refused(self) -> None:
        """The invariant `arag.retrieval.filtering.in_force()` is written against.

        That function decides supersession on DATES ALONE and refuses to infer one from the
        edge, because a chunk that cannot be placed in time must not be guessed about at
        query time. Nothing enforced the other half of that contract.

        Measured 2026-10-02: star-comprehensive-2021 carried `superseded_by` with no
        `effective_to`, so retrieving h-01 with and without `as_of=2026-09-04` returned the
        IDENTICAL top 10 - three chunks of it from the replaced wording. The runner passed
        as_of, the engine built the filter, both retrievers honoured it, and the feature was
        inert because every component was behaving correctly against absent data.
        """
        with pytest.raises(ValidationError, match="effective_to is unset"):
            Manifest(sources=(source(id="new"), source(id="old", superseded_by="new")))

    def test_by_id_raises_for_unknown(self) -> None:
        with pytest.raises(KeyError, match="unknown source id"):
            Manifest(sources=(source(id="a"),)).by_id("b")


class TestShippedManifest:
    @pytest.fixture
    def manifest(self, repo_root: Path) -> Manifest:
        return Manifest.load(repo_root / "data" / "manifest" / "sources.jsonl")

    def test_loads(self, manifest: Manifest) -> None:
        # Six REAL documents plus one adversarial fixture. The split is asserted explicitly
        # because the whole point of the fixture's separate kind is that it never gets
        # counted among the real corpus by accident - see production_sources.
        assert len(manifest.production_sources) == 6
        assert len(manifest.adversarial_fixtures) == 2
        assert len(manifest.sources) == 8

    def test_regulator_outranks_every_insurer_document(self, manifest: Manifest) -> None:
        """The IRDAI circular repeals 55 prior circulars and overrides insurer wording.

        Encoding that as a comparable number is what lets the agent resolve a
        wording-versus-circular conflict instead of averaging two contradictory passages.
        """
        # Partition the PRODUCTION sources: the adversarial fixture is rank 3 by design and
        # is not an insurer document, so including it here would be comparing the wrong set.
        regulator = [s for s in manifest.production_sources if s.publisher == "IRDAI"]
        insurers = [s for s in manifest.production_sources if s.publisher != "IRDAI"]
        assert regulator, "the corpus must contain a regulator instrument"
        assert all(s.authority_rank == RANK_REGULATOR for s in regulator)
        assert all(s.authority_rank == RANK_INSURER for s in insurers)
        assert max(s.authority_rank for s in regulator) < min(s.authority_rank for s in insurers)

    def test_two_star_versions_are_linked_as_a_version_pair(self, manifest: Manifest) -> None:
        """The deliberately-ingested older Star wording must be marked superseded.

        Without this link, retrieving the 2021 terms for a 2026 question is invisible
        noise instead of a measurable failure.
        """
        new = manifest.by_id("star-comprehensive-2025")
        old = manifest.by_id("star-comprehensive-2021")
        assert old.id in new.supersedes
        assert old.superseded_by == new.id
        assert old.uin != new.uin
        assert manifest.superseded_ids() == {"star-comprehensive-2021"}

    def test_annexure_points_at_its_parent_circular(self, manifest: Manifest) -> None:
        annexure = manifest.by_id("irdai-annexure-2024")
        assert annexure.parent_document_id == "irdai-master-circular-2024"

    def test_circular_declares_a_fallback_mirror(self, manifest: Manifest) -> None:
        circular = manifest.by_id("irdai-master-circular-2024")
        assert circular.mirror_urls, "a regulatory source should have a fallback"
        assert "irdai.gov.in" in circular.url, "the regulator's own copy must be primary"

    def test_no_brochures_in_the_corpus(self, manifest: Manifest) -> None:
        assert all(s.kind is not SourceKind.BROCHURE for s in manifest.sources)

    def test_load_reports_all_bad_lines_at_once(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text(
            "# comment\n"
            '{"id": "a", "url": "u", "kind": "policy_wording", "publisher": "p", '
            '"authority_rank": 9}\n'
            '{"id": "b", "url": "u", "kind": "brochure", "publisher": "p", '
            '"authority_rank": 3}\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="2 invalid manifest entr"):
            Manifest.load(path)
