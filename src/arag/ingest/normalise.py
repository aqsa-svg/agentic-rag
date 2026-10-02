"""Text normalisation. The first thing that touches extracted text, and a correctness
requirement rather than tidying.

## Why this module is load-bearing

Spike S5 measured **269 ligature codepoints** in the corpus (254 of them in
`star-comprehensive-2021`, on every one of its 18 pages). A ligature is a single character
that renders as two letters:

    "bene\\ufb01t"  is 6 characters, one of which is U+FB01 LATIN SMALL LIGATURE FI
    "benefit"      is 7 characters

They are different strings. So:

* **BM25 cannot match them.** A user querying "benefit" produces the token `benefit`; the
  index contains `bene\\ufb01t`. Zero lexical overlap on the most-queried word in the
  domain.
* **The embedder degrades.** A subword tokeniser has never seen U+FB01 in this position and
  falls back to a byte-level or unknown token, so the vector is subtly wrong.

None of this produces an error. The text looks correct in a PDF viewer and in most
terminals. It is exactly the silent-corruption class this project exists to catch.

## N1: normalise at query time as well as index time

Non-obvious and easy to get wrong. A user who copy-pastes a phrase out of the PDF submits
a query containing U+FB01 too. Normalising only at index time leaves that query broken —
in precisely the case where the user is most likely to be quoting the document verbatim.

There is therefore exactly one public entry point, ``normalise_text``, and both the ingest
path and the query path call it. ``tests/test_ingest_normalise.py`` asserts that.

## N2: strip footnote markers BEFORE folding, and assert numeric invariance

``unicodedata.normalize("NFKC", ...)`` folds superscript digits to ordinary digits:

    "5,00,000/-\\u00b9"   --NFKC-->   "5,00,000/-1"

In this corpus the tables *are* the answers, so a footnote marker fusing onto a monetary
value is a silent numeric corruption strictly worse than the ligature problem: the answer
is confidently wrong and there is no detection path. So superscript markers are removed
before NFKC runs, and afterwards every numeric span is compared before/against after. A
changed value raises ``NumericCorruptionError`` and fails ingest loudly.

**Measured incidence in the current corpus: zero**, across five detectors (see
LIMITATIONS.md). This guard is insurance against a future document, not a fix for an
observed defect — the distinction matters, because the guard passing today proves nothing
about the guard working. The unit tests construct the hazard synthetically so the guard is
actually exercised.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from arag.obs import get_logger

log = get_logger(__name__)

# --- characters we deliberately handle -------------------------------------------------

SOFT_HYPHEN = "­"  # invisible; splits words for BM25
NBSP = " "  # renders as a space, is not a space
ZERO_WIDTH = ("​", "‌", "‍", "⁠", "﻿")

# Superscript/subscript digits and the ordinal markers used as footnote references.
# NFKC folds every one of these into an ordinary digit, which is the hazard in N2.
_SUPERSCRIPT_DIGITS = "¹²³" + "".join(chr(c) for c in range(0x2070, 0x207A))
_SUBSCRIPT_DIGITS = "".join(chr(c) for c in range(0x2080, 0x208A))
FOOTNOTE_MARKERS = _SUPERSCRIPT_DIGITS + _SUBSCRIPT_DIGITS

# Quote/dash folding. Not a correctness issue like ligatures, but "policy's" and
# "policy’s" are different BM25 tokens, so folding them is free recall.
_PUNCT_FOLD = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "–": "-",  # en dash
    "—": "-",  # em dash
    "−": "-",  # minus sign
    "…": "...",
}

# A numeric span: Indian grouping (5,00,000), decimals, and bare runs. Percentages and
# currency symbols are excluded on purpose - only the DIGITS must be invariant.
NUMERIC_SPAN = re.compile(r"\d[\d,]*(?:\.\d+)?")

_FOOTNOTE_AFTER_NUMBER = re.compile(rf"(?<=[\d)/\-])[{re.escape(FOOTNOTE_MARKERS)}]+")
_FOOTNOTE_ANYWHERE = re.compile(rf"[{re.escape(FOOTNOTE_MARKERS)}]+")


class NumericCorruptionError(RuntimeError):
    """Normalisation changed a numeric value. Ingest must stop.

    Raised rather than logged because a wrong sub-limit is not degraded quality, it is a
    confidently wrong coverage answer with no detection path downstream. The cost of
    stopping ingest is minutes; the cost of shipping a corrupted benefit table is an
    answer nobody can tell is wrong.
    """

    def __init__(self, before: list[str], after: list[str], context: str) -> None:
        changed = [f"{b!r} -> {a!r}" for b, a in zip(before, after, strict=False) if b != a]
        detail = "; ".join(changed[:5]) or f"{len(before)} spans -> {len(after)} spans"
        super().__init__(
            f"normalisation altered a numeric value in {context}: {detail}. "
            "This is the footnote-marker fusion hazard (N2): a superscript reference "
            "folding onto a monetary value. Ingest stopped rather than index a corrupted "
            "figure."
        )
        self.before = before
        self.after = after


@dataclass
class NormalisationStats:
    """What normalisation actually changed. Reported by the ingest coverage report so the
    effect is a measured number rather than a claim that it ran.
    """

    ligatures_folded: int = 0
    footnote_markers_stripped: int = 0
    soft_hyphens_removed: int = 0
    nbsp_folded: int = 0
    zero_width_removed: int = 0
    punctuation_folded: int = 0
    chars_in: int = 0
    chars_out: int = 0
    per_document: dict[str, int] = field(default_factory=dict)

    @property
    def total_changes(self) -> int:
        return (
            self.ligatures_folded
            + self.footnote_markers_stripped
            + self.soft_hyphens_removed
            + self.nbsp_folded
            + self.zero_width_removed
            + self.punctuation_folded
        )

    def merge(self, other: NormalisationStats) -> None:
        self.ligatures_folded += other.ligatures_folded
        self.footnote_markers_stripped += other.footnote_markers_stripped
        self.soft_hyphens_removed += other.soft_hyphens_removed
        self.nbsp_folded += other.nbsp_folded
        self.zero_width_removed += other.zero_width_removed
        self.punctuation_folded += other.punctuation_folded
        self.chars_in += other.chars_in
        self.chars_out += other.chars_out


LIGATURE_RANGE = re.compile("[ﬀ-ﬆ]")


def numeric_spans(text: str) -> list[str]:
    """Digit-only view of the text, used by the N2 invariance check.

    Commas are stripped before comparison so that a thousands-separator change is not
    reported as corruption, while a changed *value* still is.
    """
    return [m.group(0).replace(",", "") for m in NUMERIC_SPAN.finditer(text)]


def strip_footnote_markers(text: str, *, only_after_numbers: bool = True) -> tuple[str, int]:
    """Remove superscript/subscript footnote references before NFKC folds them to digits.

    ``only_after_numbers`` defaults to True and is the conservative choice: a superscript
    in running prose may carry meaning (an exponent, an ordinal), whereas one immediately
    following a digit, ``)``, ``/`` or ``-`` in a policy wording is a footnote reference.
    Stripping every superscript unconditionally would silently destroy "m\\u00b2" in a
    hospital-area definition.
    """
    pattern = _FOOTNOTE_AFTER_NUMBER if only_after_numbers else _FOOTNOTE_ANYWHERE
    stripped = 0

    def _drop(match: re.Match[str]) -> str:
        nonlocal stripped
        stripped += len(match.group(0))
        return ""

    return pattern.sub(_drop, text), stripped


def normalise_text(
    text: str,
    *,
    context: str = "<text>",
    strict_numerics: bool = True,
    fold_punctuation: bool = True,
) -> tuple[str, NormalisationStats]:
    """The single normalisation entry point. Called by BOTH ingest and query paths (N1).

    Order is deliberate and each step depends on the previous one:

    1. Strip footnote markers, *before* NFKC can fold them into digits (N2).
    2. Remove zero-width characters and soft hyphens, which otherwise survive NFKC and
       split words.
    3. NFKC — this is what folds ligatures to their component letters.
    4. Fold quotes and dashes, which NFKC leaves alone.
    5. Collapse NBSP and normalise whitespace runs.
    6. Verify no numeric value changed, and raise if one did.

    Returns the normalised text and a stats record of what changed.
    """
    stats = NormalisationStats(chars_in=len(text))
    if not text:
        return text, stats

    before_numerics = numeric_spans(text)

    # 1. footnote markers first
    out, stats.footnote_markers_stripped = strip_footnote_markers(text)

    # 2. invisible characters
    for ch in ZERO_WIDTH:
        count = out.count(ch)
        if count:
            stats.zero_width_removed += count
            out = out.replace(ch, "")
    stats.soft_hyphens_removed = out.count(SOFT_HYPHEN)
    out = out.replace(SOFT_HYPHEN, "")

    # 3. NFKC: the ligature fix
    stats.ligatures_folded = len(LIGATURE_RANGE.findall(out))
    out = unicodedata.normalize("NFKC", out)

    # 4. quotes and dashes (NFKC does not touch these)
    if fold_punctuation:
        for src, dst in _PUNCT_FOLD.items():
            count = out.count(src)
            if count:
                stats.punctuation_folded += count
                out = out.replace(src, dst)

    # 5. NBSP -> space. NFKC already converts NBSP, so count on the pre-NFKC text.
    stats.nbsp_folded = text.count(NBSP)
    out = out.replace(NBSP, " ")

    # 6. the N2 guard
    if strict_numerics:
        after_numerics = numeric_spans(out)
        if before_numerics != after_numerics:
            raise NumericCorruptionError(before_numerics, after_numerics, context)

    stats.chars_out = len(out)
    if stats.total_changes:
        log.debug(
            "normalised",
            context=context,
            ligatures=stats.ligatures_folded,
            footnotes=stats.footnote_markers_stripped,
            soft_hyphens=stats.soft_hyphens_removed,
            nbsp=stats.nbsp_folded,
        )
    return out, stats


def normalise_query(text: str) -> str:
    """Query-side normalisation (N1).

    A thin wrapper over ``normalise_text`` rather than a second implementation, because
    two implementations of normalisation is how the index and the query drift apart — and
    that drift is invisible: retrieval simply returns nothing for affected terms and looks
    like a relevance problem.

    ``strict_numerics=False``: a user's query is not corpus content, and refusing to answer
    because their typing tripped an ingest-time invariant would be absurd.
    """
    out, _ = normalise_text(text, context="<query>", strict_numerics=False)
    return out.strip()
