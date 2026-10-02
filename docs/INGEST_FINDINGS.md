# Spike S5 — what actually breaks in the corpus

Date: 2026-09-04 · Reproduce with `arag-ingest fetch && arag-ingest probe`
Machine-readable: `data/manifest/fetch_report.json`, `data/manifest/probe_report.json`

Read-only diagnostic. **Nothing was fixed.** The purpose is to replace assumptions in
DESIGN.md with measurements before any chunking code is written.

---

## 1. Fetch: 6/6 usable

| source | pages | size | outcome |
|---|---|---|---|
| star-comprehensive-2025 | 48 | 2,052 KB | ok |
| star-comprehensive-2021 | 18 | 1,522 KB | ok |
| nivabupa-reassure2 | 34 | 681 KB | ok |
| nivabupa-rise | 34 | 200 KB | ok |
| irdai-master-circular-2024 | 17 | 940 KB | ok (primary) |
| irdai-annexure-2024 | 46 | 1,091 KB | ok |

Two fetch-layer findings, both now handled and tested:

- **nivabupa.com returns HTTP 406 to a default client.** Content negotiation, not
  authentication — the documents are public. Fixed with a browser `User-Agent` and an
  explicit `Accept`. This is exactly the class of failure that would have been reported as
  "source dead" by a naive fetcher.
- **The `actuariesindia.org` mirror fails TLS chain verification**
  (`CERTIFICATE_VERIFY_FAILED`). Recorded as an attempt error while the primary succeeded,
  which is the dead-source case working as designed. It is also why the manifest now uses
  IRDAI's **own** copy as primary rather than that third-party mirror: a regulatory
  instrument should be read from the regulator.

The IRDAI landing page supplied resolved to two separate English documents (circular +
annexure), both ingested as separate sources per instruction. IRDAI's own filename for the
annexure contains the typo `circualr`; the URL keeps it verbatim, since normalising it
breaks the fetch.

---

## 2. Probe summary

```
source                        pp    chars  med/pp  ocr  2col  splic   lig  fused  tbl  heading
star-comprehensive-2025       48  134,147   3,016    0    32      0    15      0   30  numbered_clause
star-comprehensive-2021       18  135,032   8,277    0    15      4   254      0   20  all_caps
nivabupa-reassure2            34   84,281   2,547    0     4      0     0      0   29  all_caps
nivabupa-rise                 34   84,520   2,574    0     5      0     0      0   30  all_caps
irdai-master-circular-2024    17   29,017   1,831    0     0      0     0      0   15  numbered_clause
irdai-annexure-2024           46   62,084   1,328    0     5      0     0      0   41  numbered_clause
```

197 pages, 529k characters, 165 table candidates.

---

## 3. Three assumptions in DESIGN.md are now disproven

### 3.1 No page in this corpus needs OCR — the VLM stage is not required

DESIGN §2 assumed ~20% scanned, image-only pages, and §4 budgets a Qwen2.5-VL OCR stage
for them. **Measured: zero.** All 197 pages carry a usable text layer. The single
image-dominant page (star-2025 p1) is a full-bleed cover carrying the words "Policy
Wordings" and nothing else; OCR would recover nothing from it.

**Consequence:** the OCR branch comes out of the v1 critical path. It stays in the design
as a documented capability with a stated trigger (`pages_needing_ocr > 0` in the probe
report) rather than being built speculatively for a corpus that does not need it. This is
the single largest scope reduction the spike produced.

### 3.2 The corpus is English-only — confirmed, not assumed

DESIGN §13 assumption 2 held. Zero Devanagari codepoints across all six documents. The
`bge-base-en-v1.5` choice in §5.3 stands and no multilingual model is needed. Note that
Hindi versions of both IRDAI documents exist on the same landing page and were **not**
ingested; adding them later would invalidate this.

### 3.3 Two-column layout is not the problem — 4 pages, not 61

DESIGN §2 lists multi-column layout as a headline mess factor. 56 pages are two-column,
but only **4** extract in genuinely wrong order, all in `star-comprehensive-2021`
(pp. 5, 7, 8, 9). The rest emit one whole column before the other, which is correct.

---

## 4. The real defects, worst first

### 4.1 Ligature contamination — 269 occurrences (the highest-impact finding)

| document | ligature codepoints | pages affected |
|---|---|---|
| star-comprehensive-2021 | 254 | 18 of 18 (all) |
| star-comprehensive-2025 | 15 | 7 |
| everything else | 0 | — |

Observed tokens: `beneﬁt`, `Identiﬁcation`, `Ofﬁce`, `speciﬁed`, `ﬁnger`, `Beneﬁts`.

These contain **U+FB01**, a single codepoint that is *not* the letters `f` and `i`. So:

```python
"beneﬁt" != "benefit"        # BM25 cannot match a user's query
"fi" not in "beneﬁt"          # substring search fails too
unicodedata.normalize("NFKC", "beneﬁt") == "benefit"   # the fix
```

This silently breaks the lexical half of the hybrid retriever on the most
frequently-queried word in the domain, and it degrades the dense half by feeding the
tokeniser an out-of-vocabulary character. It produces no error and no visible garbling —
the text *looks* correct in a PDF viewer and in most terminals.

**Fix (v1, ingest):** Unicode NFKC normalisation before chunking, plus soft-hyphen removal
and NBSP folding. Cheap, and it must land before the first index is built, because
re-indexing after the fact means every recorded retrieval number is incomparable.

**Eval consequence:** this deserves its own golden-set probe. A question phrased with
"benefit" against `star-comprehensive-2021` is a lexical-retrieval trap, and v1-without-
normalisation versus v1-with-normalisation is a clean measured delta for the write-up.

### 4.2 Tables are the dominant ingest problem — 165 candidates on 131 of 197 pages

| document | table candidates | pages with tables |
|---|---|---|
| irdai-annexure-2024 | 41 | 41 of 46 |
| nivabupa-rise | 30 | 25 of 34 |
| star-comprehensive-2025 | 30 | 18 of 48 |
| nivabupa-reassure2 | 29 | 24 of 34 |
| star-comprehensive-2021 | 20 | 12 of 18 |
| irdai-master-circular-2024 | 15 | 11 of 17 |

Two thirds of all pages contain a table. In this domain tables *are* the answer to most
high-value questions — room-rent limits, sub-limits, waiting-period grids, discount
schedules. DESIGN §5.2's "tables extracted separately and never split" moves from a
sensible precaution to the load-bearing ingest decision.

### 4.3 Interleaved prose columns — 4 pages, one document

`star-comprehensive-2021` pp. 5, 7, 8, 9. Clauses from the left and right columns are
spliced together in extraction order. The result reads fluently and is wrong, which makes
it worse than a loud failure. Confined to one document, so a per-document
column-aware reading-order pass is sufficient; a corpus-wide layout engine is not.

### 4.4 Minor normalisation

2 soft hyphens (star-2025), 58 non-breaking spaces (42 of them in nivabupa-rise). Folded
by the same NFKC pass as 4.1.

---

## 5. Two structural findings that change the chunker design

### 5.1 Heading grammar differs per document — one regex cannot work

| document | dominant pattern | full distribution |
|---|---|---|
| star-comprehensive-2025 | `numbered_clause` | numbered=144, caps=100, annexure=5, roman=4 |
| star-comprehensive-2021 | `all_caps` | caps=131, numbered=125, section_numeric=22, alpha=12 |
| nivabupa-reassure2 | `all_caps` | caps=170, numbered=94, annexure=3 |
| nivabupa-rise | `all_caps` | caps=195, numbered=135, annexure=5 |
| irdai-master-circular-2024 | `numbered_clause` | numbered=44, section_numeric=4, caps=4 |
| irdai-annexure-2024 | `numbered_clause` | numbered=70, caps=22, roman=13, annexure=7 |

The two Star wordings are **the same product** and disagree on heading style. So the
clause-aware chunker in DESIGN §5.2 needs a per-document heading strategy selected by
measurement at ingest, not one hand-tuned regex. Only `nivabupa-reassure2` ships a PDF
outline (2 entries — too thin to use).

### 5.2 Page numbers are not comparable across versions — cite clauses, not pages

`star-comprehensive-2021`: 18 pages, 135,032 chars (8,277 median chars/page).
`star-comprehensive-2025`: 48 pages, 134,147 chars (3,016 median chars/page).

Near-identical content volume, 2.7x difference in pagination. The 2021 document crams the
same wording into a third of the pages.

**Consequence for the eval schema:** `ground_truth_spans` and `must_not_cite` must key on
`clause_id`, and page must be treated as a weak fallback only. A page-keyed supersession
trap would be meaningless between these two documents. The `SpanMatcher` default
(`clause_prefix`, with page as fallback and `page_tolerance=0`) is already correct for
this — now for a measured reason rather than a guessed one.

Suspected cause of 4.1 and 4.3 both landing on the 2021 document: its producer is
`PlotSoft PDFill 9.0`, a re-processing tool, whereas the clean documents came from Adobe
PDF Library and Word 2016.

---

## 6. My own probe was wrong three times

Reported because the corrections are the reason the numbers above can be trusted. The
first version of `probe.py` over-reported broken pages by roughly **15x** (61 → 4).

| # | Bad heuristic | What it wrongly flagged | Verification that caught it | Fix |
|---|---|---|---|---|
| 1 | Low space ratio ⇒ fused text | `nivabupa-rise` p34, a clean discount-rate table | Dumped the page; cells extract one per line, so a low space ratio is normal | Measure abnormally *long* tokens instead |
| 2 | Two-column layout ⇒ broken reading order | All 32 two-column pages of star-2025 | Printed block x-ranges and emission order; the PDF emits one full column then the other | Count left↔right transitions; 1 switch is correct, >1 is damage |
| 3 | Column transitions ⇒ prose splicing | 18 of 22 flagged pages | All 22 flagged pages contained tables; zero table-free flagged pages | Exclude blocks intersecting a detected table before counting |

Each has a regression test in `tests/test_ingest_probe.py` asserting against the real
corpus, so reintroducing any of them fails the build.

The general lesson, and the reason the verification rounds were worth the time: a
diagnostic that has not been checked against the artefact it measures is an opinion with a
number attached.

---

## 7. What v1 ingest must do, in priority order

1. **NFKC normalisation** (ligatures, soft hyphens, NBSP) — before the first index exists.
2. **Table isolation** — two thirds of pages; never split a table; serialise to Markdown
   plus a generated natural-language gloss.
3. **Per-document heading strategy**, selected from the measured pattern distribution.
4. **Column-aware reading order** for the 4 affected pages only.
5. **No OCR stage.** Gated on `pages_needing_ocr > 0`, which is currently zero.
6. **Clause-keyed provenance**, page as weak fallback.
