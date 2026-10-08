# v1r — the re-frozen baseline

**`data/baseline_v1r.json` replaces `data/baseline_v1.json` as the origin for every
subsequent delta.** The original file is kept, unedited, and `docs/BASELINE_V1.md` keeps its
text and its correction banner — a frozen baseline that is quietly corrected is no longer a
baseline, so the wrong numbers stay visible next to the reason they were wrong.

## Why a re-freeze happened

Silent-wrongness instance 9. The chunker recorded a clause's heading number (`3`) while the
golden set used the IRDAI code (`excl.03`), so **0 of 1,024 chunks carried any `excl.NN`
id** and three of the seven labelled items could not match a chunk at all, however well
retrieval performed. The v1 baseline was measured through that matcher.

Nothing about retrieval changed between the two runs. Same corpus, same BM25 parameters
(`k1=1.5`, `b=0.75`, no stemming), same `top_k=10`, same labels, same two serialisations.
The only difference is that chunks now carry the clause ids their text contains.

- **Run:** BM25-only, no rerank, no query expansion, no generation.
- **Golden set:** `data/golden/v1.jsonl`, 7 items — 5 with ground-truth spans, 2
  `unanswerable` whose retrieval metrics are undefined rather than zero.
- **Recorded:** `data/baseline_v1r.json`.
- **Profile:** `v1r_baseline`, derived from the markdown figures below and now active.

---

## Aggregate

| metric | markdown | ~~v1 as published~~ | row_NL | ~~v1 as published~~ |
|---|---|---|---|---|
| recall@10 | **0.400** | ~~0.300~~ | 0.300 | ~~0.300~~ |
| nDCG@10 | **0.235** | ~~0.188~~ | 0.179 | ~~0.179~~ |
| MRR | **0.207** | ~~0.167~~ | 0.150 | ~~0.150~~ |

> **nDCG re-checked against the corrected metric (2026-10-08) and UNCHANGED.** The nDCG bug
> fixed on 2026-10-08 over-credited a span covered by several retrieved chunks; re-scored from
> the stored per-span ranks, **both BM25 figures above (markdown 0.235, row-NL 0.179) are
> identical to 3+ decimal places.** BM25 returned no duplicate per-span chunks on this set, so
> there was nothing to over-credit - the bug only moved the dense and hybrid runs (v2, v4).
> Recorded because a reader who knows the metric was wrong will wonder whether these were
> re-checked: they were, and they did not move.
| ctx precision@10 | **0.060** | ~~0.040~~ | 0.040 | ~~0.040~~ |
| hit rate | **0.600** | ~~0.400~~ | 0.400 | ~~0.400~~ |

Cross-validated: these markdown figures match the `bm25` column of the v2 run
(`data/v2_measure.json`) exactly, produced by a different script on the same corpus.

**row_NL did not move at all.** The fix recovered h-28's `excl.01` span under markdown and
not under row_NL, where that chunk stays outside the top 10. So the serialisation gap, which
the original baseline reported as a tie on recall, is real and now visible: **markdown 0.400
against row_NL 0.300.** The v1 note that T3's quality columns were unusable still stands for
the table question — no labelled item targets a table — but the two serialisations are no
longer indistinguishable on quality, and the difference runs the opposite way to what the
smaller index and lower token count would suggest.

## Per item (markdown)

| item | spans retrieved | ranks | ~~v1 as published~~ |
|---|---|---|---|
| h-01 | 1/1 | `star-2025 p4 def.hospital` #2 | ~~1/1, #2~~ (unchanged) |
| h-02 | 0/2 | both missed | ~~0/2~~ (unchanged under BM25) |
| h-03 | 0/1 | missed | ~~0/1~~ (unchanged) |
| h-04 | 1/2 | `p42 cl.12` #3 · `p41 cl.9` missed | ~~1/2~~ (unchanged) |
| h-28 | **1/2** | **`p31 excl.01` #5** · `p30 cl.25` missed | ~~0/2, both missed~~ |
| h-23 | n/a | unanswerable — undefined, not zero | — |
| h-24 | n/a | unanswerable — undefined, not zero | — |

h-28 is the whole delta under BM25: its `excl.01` span was retrieved at rank 5 all along and
scored as a miss.

## Expectation categories (markdown)

| | PASSED | FAILED | FAILED_AS_EXPECTED | FIXED |
|---|---|---|---|---|
| as shipped | 2 | 5 | 0 | 0 |
| retrieval-only | **5** | 2 | 0 | 0 |

`as shipped` PASSED 2 is h-23 and h-24 only, and is **not a quality signal**: they pass
because the engine abstains on everything, the same fact that fails all five `answer` items.

## Per serialisation

| | markdown | row_NL |
|---|---|---|
| chunks | 1,024 (703 prose / 99 def / 222 table) | 2,537 (703 prose / 99 def / 1,735 table) |
| index bytes | 2,390,950 | 3,259,401 |
| build seconds | 110.52 | 121.49 |
| latency mean / p50 / p95 ms | 8.92 / 8.23 / 19.75 | 9.58 / 10.22 / 11.92 |
| tokens at top-10 | 3,744 | 2,145 |

Timing remains load-dependent and indicative, not a measurement — see the range discussion
in `docs/BASELINE_V1.md`. Quality figures reproduced exactly across two independent scripts.

---

## The gate was disarmed, not merely understated

This is the part worth stating on its own, because it is the clearest answer to "why did the
chunker gap matter?"

`v1_baseline` set `min_recall_at_k: 0.30`, derived from the measured 0.300. The true figure
was **0.400**. So the gate sat a full tenth below the real floor — and the arithmetic of what
that permitted is exact:

> Corrected per-item recall under BM25: h-01 `1.0`, h-02 `0.0`, h-03 `0.0`, h-04 `0.5`,
> h-28 `0.5` → mean **0.400**.
>
> Lose h-28 entirely to a regression: `(1.0 + 0 + 0 + 0.5 + 0) / 5` = **0.300**.
>
> `0.300 >= 0.30` → **GATE PASS.**

A change that destroyed one item's retrieval completely would have been reported as green.
The guard built to catch a regression in these numbers could not catch a regression in these
numbers, and nothing about it looked wrong from outside: it was passing throughout.

That is the consequence recorded under instance 9 in `docs/SILENT_WRONGNESS.md`. The chunker
gap did not just understate the measurements — it disarmed the guard derived from them.

`v1r_baseline` sets `min_recall_at_k: 0.40` at the measured floor with no tolerance, and the
same regression now fails.

## What has NOT been fixed by the re-freeze

- **h-03 still retrieves nothing.** The chunk carrying `excl.03` is reachable — the
  agreement test proves a span on it matches `star-comprehensive-2025:p32:s146` — and it is
  outside the top 10 under both BM25 and dense. This is now the only labelled item that
  retrieves nothing under dense, and it is the cleanest diagnosis target in the set.
  Deliberately left failing.
- ~~**`max_supersession_violations: 0` is still vacuous.** No label carries a
  `must_not_cite`, so nothing can violate it.~~ **Closed 2026-10-06 by h-14**, which names
  `star-comprehensive-2021` - where `excl.33` excludes sleep-apnea treatment that the 2025
  wording reverses into a bariatric qualifier. The metric now has something to violate.
  The measured numbers above were taken while it was still vacuous and are NOT restated;
  h-01 still returns the superseded 2021 definition at rank 1 under BM25, and h-01 itself
  still carries no `must_not_cite`, so the gate still passes for it. What changed is that
  the metric is no longer vacuous for the SET - not that supersession is defended. Nothing
  produces `AbstainReason.ONLY_SUPERSEDED_EVIDENCE`.
- **The composition shortfall is unchanged:** 7 of 120 items. These remain
  development-signal numbers and must not be quoted as results.
