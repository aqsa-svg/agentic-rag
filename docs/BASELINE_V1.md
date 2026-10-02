# v1 baseline — FROZEN, and **INVALIDATED 2026-09-16**

> ## These numbers were measured against a broken matcher
>
> Silent-wrongness instance 9: the chunker recorded a clause's heading number (`3`) while the
> golden set used the IRDAI code (`excl.03`), so **0 of 1,024 chunks carried any `excl.NN`
> id** and 3 of the labelled items could not score at all. Every figure below understates
> retrieval quality.
>
> Re-measured on identical corpus, config and labels, with only the matcher fixed:
>
> | BM25 | recall@10 | nDCG@10 | MRR | ctx prec | hit rate |
> |---|---|---|---|---|---|
> | as published below | 0.300 | 0.188 | 0.167 | 0.040 | 0.400 |
> | **actual** | **0.400** | **0.235** | **0.207** | **0.060** | **0.600** |
>
> Per item, h-28 was reported 0.000 and is actually **0.500** under BM25.
>
> **Consequences, stated rather than quietly fixed:**
>
> * The `v1_baseline` threshold profile in `src/arag/eval/thresholds.yaml` was derived from
>   the understated figures, so its floors (`min_recall_at_k: 0.30`) sit **below** the true
>   v1 floor. It has not been changed: re-deriving a gate is a deliberate act and the
>   corrected baseline needs to be re-run and re-frozen first.
> * Any delta quoted against this document before 2026-09-16 is wrong in an unknown
>   direction, because both sides were mismeasured by different amounts.
>
> The document is kept unedited below, because a frozen baseline that is silently corrected
> is no longer a baseline. The correction lives here, dated, at the top.

---

# v1 baseline — FROZEN (as originally published)

**`data/baseline_v1.json` is the reference point for every subsequent measurement and must
not be regenerated.** Every later claim in this project — "the reranker improved context
precision", "dense retrieval closed the vocabulary gap" — is a delta against these numbers.
Regenerating the file would silently move the origin, and a delta measured against a moved
origin is not a measurement.

If the baseline ever needs to be re-measured, it gets a **new** file (`baseline_v2.json`)
and a new section here, with the reason for the re-measurement written down. This file is
appended to, not edited.

> **Why this document exists rather than a section in `EVAL_LOG.md`.** `EVAL_LOG.md` is a
> generated view: `arag.eval.report.write_report` calls `write_text` on the whole file from
> `data/eval_runs.jsonl` every time `arag-eval report` runs. Anything hand-written there is
> destroyed on the next run. The `EVAL_LOG.md` History table carries the machine-recorded
> v1 row; this file is the authored record of what the run means, and survives.

- **Run:** BM25-only, `top_k=10`, `k1=1.5`, `b=0.75`, no stemming, no rerank, no query
  expansion, no generation.
- **Golden set:** `data/golden/v1.jsonl`, 5 items, all `authored_by: human`.
- **Profile:** `v1_baseline` (thresholds derived from this run — see the PROVISIONAL note in
  `src/arag/eval/thresholds.yaml`: at n=5 one item flipping moves recall by 0.2).
- **Recorded:** `data/baseline_v1.json`.

---

## Aggregate

| | markdown | row_NL |
|---|---|---|
| recall@10 | 0.300 | 0.300 |
| nDCG@10 | 0.188 | 0.179 |
| MRR | 0.167 | 0.150 |
| ctx precision@10 | 0.040 | 0.040 |
| hit rate | 0.400 | 0.400 |

By stratum: `definitional_carveout` recall 0.333 / nDCG 0.210 (3 items) ·
`conditional_override` recall 0.250 / nDCG 0.153 (2 items).

Expectation categories:

| | PASSED | FAILED | FAILED_AS_EXPECTED | FIXED |
|---|---|---|---|---|
| as shipped | 0 | 5 | 0 | 0 |
| retrieval-only | 2 | 3 | 0 | 0 |

`as shipped` is the gate verdict: `RetrievalOnlyEngine` abstains on every item and an
`answer` item requires a non-abstaining response, so all five fail for a reason unrelated to
retrieval. `retrieval-only` is the same rule with the abstention clause lifted — the number
that will move when v2 lands a generator.

## Per item

| item | span retrieved? | rank of correct span |
|---|---|---|
| h-01 | yes (1/1) | `star-2025 p4 def.hospital` = rank 2 |
| h-02 | no (0/2) | `p31 excl.02` NOT RETRIEVED · `p32` NOT RETRIEVED |
| h-03 | no (0/1) | `p32 excl.03` NOT RETRIEVED |
| h-04 | partial (1/2) | `p42 cl.12` = rank 3 (row_NL: rank 4) · `p41 cl.9` NOT RETRIEVED |
| h-28 | no (0/2) | `p31 excl.01` NOT RETRIEVED · `p30 cl.25` NOT RETRIEVED |

## Per serialisation

| | markdown | row_NL |
|---|---|---|
| chunk count | 1,024 (703 prose / 99 def / 222 table) | 2,537 (703 prose / 99 def / 1,735 table) |
| index bytes | 2,390,950 | 3,259,401 |
| build seconds | 75.89 | 54.35 |
| latency mean ms | 2.69 | 4.20 |
| latency p50 ms | 2.68 | 4.31 |
| latency p95 ms | 3.22 | 4.93 |
| tokens/query at top-10 | 3,456 | 2,162 |

Both: 2 tables refused, 0 pages uncovered across all 197 pages.

**Build seconds and latency are ranges, not points.** Three runs of identical code on the
same corpus: build 73.30/73.23, 113.14/119.08, 75.89/54.35; mean latency 3.86/6.10,
6.58/11.23, 2.69/4.20. Row_NL built slower than markdown twice and faster once. Machine
load, not code. The retrieval metrics reproduced to the digit in all three runs, which is
the distinction that matters: quality is frozen, timing is indicative.

---

## Finding 1 (highest priority) — the supersession failure, measured

**h-01's rank-1 result is `star-comprehensive-2021 p3 def.hospital`: the superseded
document, ranked above the current one.** The correct 2025 span is at rank 2.

This is the project's central failure mode, and until now it was hypothesised. It is now
measured — on the only item in the golden set that retrieved anything at all.

Three things make it worse than the raw fact:

1. **The metric looks healthy.** h-01 scores recall 1.000 and is the single best-performing
   item in the baseline, because the correct span is inside the top 10. Recall@k cannot see
   that the *best* candidate is from a wording replaced in 2025.
2. **The gate is silent.** `max_supersession_violations: 0` passes, because h-01 carries no
   `must_not_cite` entry. The threshold exists, the metric runs, and it returns 0 — a
   vacuous 0, measuring nothing. This is exactly the "mechanism that runs is not a result"
   failure the eval design was written to avoid, and it is present in the baseline.
3. **A generator would have used it.** With generation wired, rank 1 is the passage most
   likely to be quoted. The answer would cite a definition of "Hospital" that no longer
   governs any policy sold today, and it would cite it confidently, with a page number.

**What this does not license:** a supersession filter added now would improve the number
without anyone having established what it costs. The fix belongs in v3 with a delta against
this baseline, and the label needs a `must_not_cite` entry first so the gate can actually
see the violation. Recorded as the top priority for the next slice, not fixed here.

## Finding 2 — the pre-registered prediction held

DESIGN §11 committed to this outcome **before the corpus was indexed or the labels written**
(the section was revised when v1 was changed from dense-only to BM25-only, and the cost was
stated up front rather than discovered afterwards):

> The cost is real and worth stating up front: **BM25 alone should fail the
> vocabulary-mismatch stratum almost completely.** A user asking about "a nurse at home"
> shares no tokens with *domiciliary hospitalisation*, so recall on stratum 5 is expected
> near zero at v1. That is the expected result, not a defect — and it is precisely the gap
> dense retrieval exists to close, which makes v1 a clean measurement of what the embedder
> is worth.

**Measured: recall@10 = 0.300 over the 5 items, and h-03 = 0.000 with zero chunks retrieved
from the correct document.** h-03's question is *"I bought the policy five days ago and now
I have a fever"*; the governing clause reads *"30-day waiting period — Code Excl 03…
treatment of any illness within 30 days"*. Shared terms: none. All ten results came from
Niva Bupa — a different insurer's wording entirely.

The prediction was made first and is quoted above verbatim; the measurement followed. That
ordering is the point. A baseline that confirms a prediction nobody wrote down in advance
is indistinguishable from a baseline explained after the fact.

Note the two zero-scoring failures are **not the same failure**, and the aggregate hides it:

- **h-03** retrieved nothing from the right document — vocabulary mismatch, a *matching*
  failure. Dense retrieval is the fix.
- **h-28** retrieved the right document and the wrong pages — a *ranking* failure. A
  reranker is the fix.

Both read 0.000. Only the per-item ranks separate them.

## Finding 3 — row-NL sends 37% less context, and that is counter-intuitive

| | markdown | row_NL |
|---|---|---|
| chunk count | 1,024 | 2,537 (**2.5×**) |
| index bytes | 2,390,950 | 3,259,401 (**+36%**) |
| tokens/query at top-10 | 3,456 | **2,162 (−37%)** |

More chunks, a larger index, and **less context per query**. The intuition that a
finer-grained corpus means more retrieved text is wrong here, and the reason is that top-k
is a count, not a budget: ten row-sentences are less text than ten markdown tables. Splitting
tables into rows shrinks the average chunk, and `k=10` then buys less of it.

Which direction that is *good* is not settled by this table. Less context is cheaper and
leaves more room under a context limit; it is also less surrounding material for a generator
to ground a qualified answer in, and a table row torn from its neighbours can be read as
unconditional when it is not.

**The quality columns are not usable yet.** recall@10 is identical (0.300) and nDCG differs
only slightly (0.188 vs 0.179) — but **no labelled item targets a table**, so both
serialisations produce identical prose and definition chunks, and every scored item was
answered from those. The recall tie is an artefact of the golden set, not a finding about
serialisation. The cost and latency columns are valid regardless of what gets labelled.

Blocked on: h-06–h-09 (the four table-targeting labels, h-09 marked `expected_to_fail`).
Until those land, T3 has a cost answer and no quality answer, and the write-up says so.
