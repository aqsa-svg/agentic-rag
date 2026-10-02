# v2 — dense + RRF hybrid: a MEASUREMENT, not a promotion

> ## CORRECTED 2026-09-16 — the first version of this document was measured against a broken matcher
>
> Everything below was first published from a run in which **three of seven golden items
> could not score at all**. The chunker recorded a clause's heading number (`3`) while the
> labels used the IRDAI code (`excl.03`), so `SpanMatcher` could not match them: 0 of 1,024
> chunks carried any `excl.NN` id. Silent-wrongness instance 9.
>
> The corrected numbers are below. The superseded ones are struck through **in place** rather
> than deleted, with what was wrong and why, because a corrected record that shows the
> correction is worth more than a clean one.
>
> Nothing about the retrievers changed between the two runs — same corpus, same config, same
> labels, same embeddings. The entire delta is the matcher fix.
>
> | | recall@10 | nDCG@10 | MRR | ctx prec | hit rate |
> |---|---|---|---|---|---|
> | BM25 ~~before~~ | ~~0.300~~ | ~~0.188~~ | ~~0.167~~ | ~~0.040~~ | ~~0.400~~ |
> | BM25 **after** | **0.400** | **0.235** | **0.207** | **0.060** | **0.600** |
> | dense ~~before~~ | ~~0.400~~ | ~~0.309~~ | ~~0.307~~ | ~~0.060~~ | ~~0.600~~ |
> | dense **after** | **0.600** | **0.420** | **0.327** | **0.120** | **0.800** |
> | hybrid ~~before~~ | ~~0.300~~ | ~~0.253~~ | ~~0.250~~ | ~~0.040~~ | ~~0.400~~ |
> | hybrid **after** | **0.400** | **0.383** | **0.350** | **0.080** | **0.600** |
>
> **`docs/BASELINE_V1.md` is invalidated by the same defect** and carries its own notice. The
> `v1_baseline` threshold profile was derived from the understated BM25 figures.

**The gate stays at `v1_baseline`. No v2 threshold profile exists and none should be written
yet.** The configuration is not settled: the hybrid loses to dense alone on every quality
metric, and writing a profile now would bake in a decision nobody has made. `rrf_k` stays at
the documented default of 60 and the weights stay equal, to be swept on day 7 — a delta
measured at a value chosen *after* seeing the numbers is not a delta.

`data/baseline_v1.json` remains the frozen origin. This run is recorded in
`data/v2_measure.json`.

- **Run:** BM25 vs dense vs RRF hybrid, `top_k=10`, `bge-base-en-v1.5` (768-d, CPU),
  `rrf_k=60`, equal weights, exact cosine search, no reranker, no generation.
- **Corpus:** 1,024 markdown-serialised chunks, identical to the v1 baseline.
- **Golden set:** `data/golden/v1.jsonl`, 7 items — 5 with ground-truth spans, 2
  `unanswerable` whose retrieval metrics are undefined rather than zero.
- **Embedding cost:** 1,184s (19.7 min) for the corpus, 1,157 ms/chunk under load. See
  DESIGN §9.

---

## Aggregate

*Corrected run, 2026-09-16, after the instance 9 matcher fix.*

| metric | bm25 (v1) | dense | hybrid | Δ hybrid − bm25 |
|---|---|---|---|---|
| recall@10 | 0.400 | **0.600** | 0.400 | +0.000 |
| nDCG@10 | 0.235 | **0.420** | 0.383 | +0.148 |
| MRR | 0.207 | 0.327 | **0.350** | +0.143 |
| ctx precision@10 | 0.060 | **0.120** | 0.080 | +0.020 |
| hit rate | 0.600 | **0.800** | 0.600 | +0.000 |
| latency mean ms | 2.4 | 123.3 | 103.8 | — |
| tokens at top-10 | 3,744 | 3,348 | 3,779 | — |

Dense still leads on four of five quality metrics. The hybrid now takes MRR, which it did
not before — so the fusion question is genuinely open rather than settled against it, and
that alone is reason the earlier conclusion could not be left standing.

## Per item

| item | bm25 | dense | hybrid |
|---|---|---|---|
| h-01 | 1.000, nDCG 0.631, [#2] | 1.000, nDCG **1.000**, [#1] | 1.000, nDCG 1.000, [#1] |
| h-02 | 0.000, [miss, miss] | **0.500**, [**#10**, miss] | 0.000, [miss, miss] |
| h-03 | 0.000, [miss] | **0.000, [miss]** | 0.000, [miss] |
| h-04 | 0.500, [#3, miss] | 0.500, [#5, miss] | 0.500, [#4, miss] |
| h-28 | **0.500**, [#5, miss] | **1.000**, [#8, #3] | 0.500, [#2, miss] |

Superseded per-item figures, for comparison — h-02, h-28 and BM25's h-28 were all scored
against spans the matcher could not see: ~~h-02 dense 0.000 [miss, miss]~~ · ~~h-28 bm25
0.000 [miss, miss]~~ · ~~h-28 dense 0.500 [miss, #3]~~ · ~~h-28 hybrid 0.000~~.

## Expectation categories

```
engine   row                    PASSED   FAILED   FAILED_AS_EXPECTED   FIXED
bm25     as shipped                  2        5                    0       0
bm25     retrieval-only              5        2                    0       0
dense    as shipped                  2        5                    0       0
dense    retrieval-only              6        1                    0       0
hybrid   as shipped                  2        5                    0       0
hybrid   retrieval-only              5        2                    0       0
```

~~Superseded: bm25 and hybrid retrieval-only 4/3, dense 5/2.~~ Each engine gained a pass
from items the matcher previously could not score. Under dense, **h-03 is now the only
labelled item that retrieves nothing.**

---

## 1. The pre-registered prediction: FAILED on h-03, UNTESTED on h-02

DESIGN §11 committed to this before the corpus was embedded, when v1 was changed from
dense-only to BM25-only:

> The cost is real and worth stating up front: **BM25 alone should fail the
> vocabulary-mismatch stratum almost completely.** A user asking about "a nurse at home"
> shares no tokens with *domiciliary hospitalisation*, so recall on stratum 5 is expected
> near zero at v1. That is the expected result, not a defect — and **it is precisely the gap
> dense retrieval exists to close**, which makes v1 a clean measurement of what the embedder
> is worth.

The verdict splits, and the first version of this document ran the two items together.

### h-03 — the prediction genuinely failed

*"I bought the policy five days ago and now I have a fever. Will the hospital bill be paid?"*
against `star-comprehensive-2025 p32 excl.03`, *"30-day waiting period — Code Excl 03"*.

**recall 0.000 under BM25, 0.000 under dense, after the matcher fix.** The chunk carrying
excl.03 is not in either engine's top 10; dense's top hit merely moved from one wrong
document to another. The correct chunk is reachable — the agreement test proves a span on
`excl.03 p32` matches `star-comprehensive-2025:p32:s146` — so this is a true retrieval miss,
not a scoring artefact.

The prediction was specific, written in advance, and **wrong for this item**: the gap dense
retrieval was predicted to close is still open. h-03 is now the only labelled item that
retrieves nothing under dense.

### h-02 — the prediction was never tested

*"I need surgery for a hernia. How long do I have to wait after buying the policy?"*

The reported 0.000 was a **matcher artefact, not a retrieval result**. The correct chunk,
`star-comprehensive-2025:p31:s142`, carrying excl.02's text, **was retrieved at rank #10 by
the dense retriever**. `SpanMatcher` could not match it because the chunk carried the id `2`
and the label said `excl.02`. Corrected, **h-02 scores recall 0.500 under dense.**

So h-02 never bore on the prediction either way. It could not score.

### What was wrong in the first version, and why

> ~~"**Measured: h-02 and h-03 score 0.000 under dense, exactly as they did under BM25.**
> … The remaining failures are therefore **not lexical-versus-semantic**, which rules out
> the whole class of fix v2 represents."~~

Struck. That conclusion generalised from two items to a class, and **one of the two was not
a measurement**. The honest version is narrower and less satisfying: dense did not close
h-03, and h-02 provides no evidence in either direction. One item is not a basis for ruling
out a class of fix.

The error was not arithmetic. It was treating a zero as a result without asking whether the
item was capable of scoring — the same failure the `undefined ≠ zero` rule exists to prevent
for unanswerable items, reappearing in a place that rule did not cover.

## 2. The hybrid loses to dense on four of five quality metrics — no longer all five

| | recall | nDCG | MRR | ctx precision | hit rate |
|---|---|---|---|---|---|
| dense | **0.600** | **0.420** | 0.327 | **0.120** | **0.800** |
| hybrid | 0.400 | 0.383 | **0.350** | 0.080 | 0.600 |

~~"Not a wash — worse on all five."~~ Corrected: the hybrid now **wins MRR** (0.350 vs
0.327). It finds fewer of the labelled spans but ranks what it finds higher, which is what
fusion is supposed to do and what the first run did not show. The fusion question is
therefore open, not settled against it.

**The mechanism still holds, and h-28 still shows it.** Dense retrieves both labelled spans
(ranks #8 and #3, recall 1.000); the hybrid retrieves one (rank #2, recall 0.500). Fusing
BM25's ranking pushed a span dense had found out of the top 10 — while simultaneously
promoting the other from #8 to #2. Both effects come from the same property:

```
score(chunk) = Σ  weight_r / (k + rank_r(chunk))
```

Magnitude is discarded deliberately, because BM25 scores and cosine similarities are not on
comparable scales. The price is that **a confident wrong ranking is weighted exactly as
heavily as a confident correct one** — so a chunk only one retriever found can be displaced
by ten the other found, regardless of how weak they are. That costs recall and buys
precision at the top. Which trade is right is exactly what the day-7 sweep of `k` and the
weights is for.

Left at `k=60`, equal weights, unswept.

## 3. The h-01 supersession improvement is INCIDENTAL and NOT load-bearing

BM25's rank-1 result for h-01 was `star-comprehensive-2021 p3 def.hospital` — the superseded
wording. Dense puts the correct 2025 definition at rank 1 and scores nDCG 1.000.

**Do not read this as the supersession problem being addressed.** Nothing in this slice
implements version handling:

- **No `as_of` filtering is wired.** `RetrievalFilters.as_of` exists and both retrievers
  honour it through the shared `arag.retrieval.filtering` module, but no caller sets it. The
  measurement ran unfiltered.
- **h-01 still carries no `must_not_cite`.** `max_supersession_violations: 0` therefore
  still reports a vacuous 0, exactly as in the v1 baseline.
- **The gate is blind to a regression here.** If a future change put the 2021 wording back
  at rank 1, every metric in this table would be unchanged — recall stays 1.000 either way,
  because the correct span is in the top 10 regardless — and no threshold would fire.

The improvement is an accident of what the embedder happened to prefer. It could reverse on
a different model, a different chunking, or a re-embed, and nothing would report it. It is
recorded here as an observation, not as progress against the supersession requirement.

## 4. The "as shipped" 2/7 is not a quality signal

Every engine shows `as shipped: PASSED 2`. Those two are **h-23 and h-24**, the
`unanswerable` items, and they pass because `RetrievalOnlyEngine` abstains on every input —
the same fact that makes all five `answer` items fail. An abstain item is satisfied by an
abstention, so a system that abstained unconditionally would score 2/7 here, and so does
this one.

The number will only mean something once a generator exists and abstention becomes a
decision rather than a default. Until then the `retrieval-only` row is the one carrying
information, and it is the one that separates the engines: dense 5 PASSED, BM25 and hybrid 4.
