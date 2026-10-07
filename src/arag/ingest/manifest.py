"""The corpus manifest: what we ingest, from where, and how it relates to everything else.

This is a declarative file rather than a hardcoded list because three fields on it are
*retrieval logic*, not bookkeeping:

* ``authority_rank`` — 1 for a regulator instrument, 2 for an insurer's policy wording.
  The IRDAI master circular repeals 55 prior circulars and overrides insurer wording, so
  when a wording and the circular disagree the circular governs. Encoding that as a
  number on the source lets the agent resolve the conflict and say which rank it relied
  on, instead of averaging two contradictory passages into confident mush.
* ``supersedes`` / ``superseded_by`` — the version graph. We deliberately ingest two
  versions of the same Star product so that retrieving the older one in preference to the
  current one is a *measurable failure*, not invisible noise.
* ``mirror_urls`` — a fallback chain. A source going dead is an expected state.

Nothing here is derived from the PDFs. The manifest is the input to ingest, and the hash
recorded after a successful fetch is what makes a rebuild reproducible without ever
committing a copyrighted document.
"""

from __future__ import annotations

import json
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceKind(StrEnum):
    POLICY_WORDING = "policy_wording"
    CIRCULAR = "circular"
    ANNEXURE = "annexure"
    BROCHURE = "brochure"
    # A deliberately poisoned document, built by tools/build_injection_fixture.py to carry
    # prompt-injection payloads for the injection stratum. It is declared in the manifest so
    # it is documented and loadable by a deliberate opt-in, and EXCLUDED from the production
    # corpus by kind (see Manifest.production_sources) so the only way a payload reaches a
    # retriever is that opt-in. A fixture must never be indistinguishable from a real source,
    # which is the whole reason it is its own kind rather than a flag on an id.
    ADVERSARIAL_FIXTURE = "adversarial_fixture"


# Plain integer constants rather than an enum. The value is compared *numerically* in
# retrieval ("prefer the lower rank"), and wrapping it in an enum invites `rank == 1`
# equality checks that silently break the moment a rank is inserted between two others.
RANK_REGULATOR = 1
RANK_INSURER = 2
RANK_MARKETING = 3
VALID_RANKS = (RANK_REGULATOR, RANK_INSURER, RANK_MARKETING)


class Source(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    url: str
    kind: SourceKind
    publisher: str
    authority_rank: int

    insurer: str | None = None
    product: str | None = None
    uin: str | None = None
    version_label: str | None = None
    effective_from: date | None = None
    # The date this document STOPPED governing. Required whenever `superseded_by` is set:
    # `arag.retrieval.filtering.in_force()` decides supersession on dates alone and will
    # not infer one from the edge, so without this the version filter silently passes the
    # replaced wording through.
    effective_to: date | None = None

    mirror_urls: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()
    superseded_by: str | None = None
    parent_document_id: str | None = None

    expected_sha256: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.authority_rank not in VALID_RANKS:
            raise ValueError(f"{self.id}: authority_rank must be one of {VALID_RANKS}")
        if self.kind is SourceKind.BROCHURE:
            raise ValueError(
                f"{self.id}: brochures are marketing copy that asserts cover the policy "
                "wording does not grant. Ingesting one poisons the corpus."
            )
        if self.kind is SourceKind.ADVERSARIAL_FIXTURE and self.authority_rank != RANK_MARKETING:
            # A poisoned document must sit at the lowest rank, so that if it ever DID leak
            # into retrieval the conflict logic would prefer every real source over it. The
            # rank is defence in depth behind the production_sources exclusion, not instead
            # of it.
            raise ValueError(
                f"{self.id}: an adversarial_fixture must have authority_rank "
                f"{RANK_MARKETING} (lowest), got {self.authority_rank}"
            )
        if self.superseded_by == self.id:
            raise ValueError(f"{self.id}: cannot supersede itself")
        return self

    @property
    def urls(self) -> tuple[str, ...]:
        """Primary first, then mirrors, in fetch order."""
        return (self.url, *self.mirror_urls)


class Manifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    sources: tuple[Source, ...]
    path: Path | None = Field(default=None)

    @model_validator(mode="after")
    def _check_graph(self) -> Self:
        ids = {s.id for s in self.sources}
        if len(ids) != len(self.sources):
            raise ValueError("duplicate source ids in manifest")

        for s in self.sources:
            # A dangling supersession edge would silently disable the version filter for
            # that document, which is the exact failure the edge exists to prevent.
            for target in s.supersedes:
                if target not in ids:
                    raise ValueError(f"{s.id}: supersedes unknown source {target!r}")
            if s.superseded_by and s.superseded_by not in ids:
                raise ValueError(f"{s.id}: superseded_by unknown source {s.superseded_by!r}")
            # THE invariant `arag.retrieval.filtering.in_force()` is written against.
            #
            # That function decides supersession on DATES ALONE and deliberately refuses to
            # consult `superseded_by`, because a supersession edge carries no date and a
            # chunk that cannot be placed in time must not be guessed about at query time.
            # It documents the counterpart obligation - "the ingest invariant is that
            # supersession sets effective_to" - and nothing enforced it.
            #
            # Measured 2026-10-02: star-comprehensive-2021 carried `superseded_by` with no
            # `effective_to`, so `in_force()` returned True for it at every date and the
            # as_of filter was a COMPLETE NO-OP for the only superseded document in the
            # corpus. Retrieving h-01 with and without `as_of=2026-09-04` returned the
            # identical top 10, three chunks of it from the replaced wording.
            #
            # The whole chain - runner passes as_of, engine builds the filter, both
            # retrievers honour it - was correct and inert. Enforced here so the next
            # supersession edge cannot be added without the date that makes it mean
            # anything.
            if s.superseded_by and s.effective_to is None:
                raise ValueError(
                    f"{s.id}: superseded_by={s.superseded_by!r} but effective_to is unset. "
                    "Supersession is decided by dates (arag.retrieval.filtering.in_force), "
                    "so an edge without a date disables the version filter for this "
                    "document instead of enabling it - silently, because every component "
                    "on the path is behaving correctly."
                )
            if s.parent_document_id and s.parent_document_id not in ids:
                raise ValueError(f"{s.id}: parent unknown {s.parent_document_id!r}")
        return self

    @property
    def production_sources(self) -> tuple[Source, ...]:
        """The sources that may be ingested into the production corpus.

        Everything except adversarial fixtures. `build_corpus` iterates this, not
        `sources`, so a poisoned document is unreachable from the retrieval path unless a
        caller deliberately opts in. Filtering by KIND rather than by id means a future
        fixture inherits the exclusion without anyone remembering to list it.
        """
        return tuple(s for s in self.sources if s.kind is not SourceKind.ADVERSARIAL_FIXTURE)

    @property
    def adversarial_fixtures(self) -> tuple[Source, ...]:
        """The poisoned documents, for the injection eval's deliberate opt-in only."""
        return tuple(s for s in self.sources if s.kind is SourceKind.ADVERSARIAL_FIXTURE)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        sources: list[Source] = []
        errors: list[str] = []
        for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                sources.append(Source.model_validate(json.loads(line)))
            except Exception as exc:
                errors.append(f"  {path.name}:{lineno}: {exc}")
        if errors:
            raise ValueError(f"{len(errors)} invalid manifest entr(ies):\n" + "\n".join(errors))
        return cls(sources=tuple(sources), path=path)

    def by_id(self, source_id: str) -> Source:
        for s in self.sources:
            if s.id == source_id:
                return s
        raise KeyError(f"unknown source id {source_id!r}")

    def superseded_ids(self) -> set[str]:
        """Documents that a later version replaces. Retrieval filters these out unless the
        caller asks for a historical ``as_of`` date.
        """
        out = {s.id for s in self.sources if s.superseded_by}
        for s in self.sources:
            out.update(s.supersedes)
        return out
