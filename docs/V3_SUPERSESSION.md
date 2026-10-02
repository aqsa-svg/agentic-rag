# v3 — the supersession filter, and what it cost

Measured 2026-10-02. Same corpus, same config, same labels, same embeddings as
`docs/V2_MEASUREMENT.md`. **The only variable is whether `as_of` is passed.**

That isolation is the point. The filter was deferred deliberately — *"a supersession filter
added now would improve the number without anyone having established what it costs"* — so
it is measured as a delta against a frozen baseline rather than switched on and admired.

Raw: `data/v3_supersession.json`.

---

## The delta

| engine | metric | `as_of` OFF | `as_of` ON | delta |
|---|---|---|---|---|
| **bm25** | recall@10 | 0.400 | 0.400 | +0.000 |
| | nDCG@10 | 0.235 | **0.314** | **+0.079** |
| | MRR | 0.207 | **0.317** | **+0.110** |
| | stale chunks in top-k | 11 | **0** | −11 |
| **dense** | recall@10 | 0.600 | **0.700** | **+0.100** |
| | nDCG@10 | 0.420 | **0.526** | **+0.106** |
| | MRR | 0.327 | **0.375** | +0.048 |
| | ctx precision@10 | 0.120 | **0.160** | +0.040 |
| | stale chunks in top-k | 14 | **0** | −14 |
| **hybrid** | nDCG@10 | 0.383 | 0.386 | +0.003 |
| | MRR | 0.350 | 0.367 | +0.017 |
| | stale chunks in top-k | 8 | **0** | −8 |

**No metric regressed on any engine.** Stale chunks reach zero everywhere.

## h-01 — the finding this project opened with, closed

```
h-01  off: star-comprehensive-2021  stale=3   →   on: star-comprehensive-2025  stale=0
```

h-01 is *"My local nursing home has 8 beds. Does it count as a hospital for my claim?"* — the
first label written. Its rank-1 result under BM25 had been the **superseded 2021 definition
of "Hospital"** since the first baseline, recorded in `docs/BASELINE_V1.md` as the
highest-priority finding: a metric reading recall 1.000 while the system's best candidate
came from a wording replaced in 2025, and the gate silent because the label carries no
`must_not_cite`.

BM25's +0.110 MRR is almost entirely this one item moving the correct definition from rank 2
to rank 1.

## Three things the numbers say that are less comfortable

**Dense gained the most, because it was the worst offender.** It pulled 14 stale chunks
against BM25's 11. An embedder has no notion of document version, and the 2021 and 2025
definitions of "Hospital" are near-identical text — so semantic similarity actively *favours*
the superseded wording. Better retrieval meant more exposure to this failure mode, which is
worth knowing before anyone treats dense as the safer leg.

**The hybrid barely moved** (+0.003 nDCG). Its RRF fusion was already suppressing some stale
chunks — by accident, for no principled reason, and therefore not something to rely on.

**recall@10 did not move for BM25 or hybrid.** Removing stale chunks improved *ordering*,
not *coverage*, which is what a correct version filter should do. A filter that was removing
real evidence would have shown the opposite, so this is the reassuring half of the result.

## What was actually broken, which was not "no filter was wired"

The filter chain was **complete and correct end to end** and did nothing:

- `arag.eval.runner` passes `as_of=item.as_of_date` — and every label carries a date
- `GeneratingEngine` builds `RetrievalFilters(as_of=...)`
- both retrievers honour it through `arag.retrieval.filtering`
- `in_force()` implements the date logic correctly

**Nothing ever wrote the metadata it reads.** Both `ChunkMeta` construction sites set only
`is_table`, `section_path` and `clause_ids`. Measured across 1,024 chunks: `insurer`,
`product`, `effective_from`, `effective_to`, `superseded_by` — all empty, always. `Source`
did not even have an `effective_to` field, so the manifest could not express the date the
filter depends on.

Proof, before the fix: retrieving h-01 with and without `as_of=2026-09-04` returned the
**identical** top 10, three chunks of it from the replaced wording.

### The mirror image of a pattern already in this repository

`docs/SILENT_WRONGNESS.md` records *a signal produced and never consumed* — `retryable`,
`retry_after_s`, `truncated`, all populated correctly and read by nothing.

This is the inverse: **a signal consumed and never produced.** Read by production code,
written by no code. Both look correct from either end, and `tools/audit_signals.py` scored
these fields **healthy**, because it only tracks reads. The audit has a blind spot exactly
symmetrical to the defect it was built to find.

## The fix

- `Source.effective_to` added; the manifest carries it.
- **The invariant is enforced at manifest load.** `filtering.in_force()` documents the
  obligation — *"the ingest invariant is that supersession sets `effective_to`"* — and
  nothing checked it. A `superseded_by` without a date is now a load error, because such an
  edge *disables* the version filter rather than enabling it, silently, while every
  component behaves correctly. It caught a test fixture immediately.
- `meta_from_source()` — one place that turns a manifest `Source` into the filterable
  surface, threaded through prose and table chunks.

## Caveats, stated rather than buried

**The dates are derived, not read.** Neither Star wording states an effective date. The UIN
`SHAHLIP26044V092526` encodes IRDAI financial year 2025-26, which begins 1 April 2025, so
`star-comprehensive-2025.effective_from = 2025-04-01` and
`star-comprehensive-2021.effective_to = 2025-03-31`. Recorded as derived in the manifest
notes. **Every gain above rests on that inference being right**; a wrong boundary means the
filter is excluding on a wrong date and these numbers move.

**Seven items, one superseded pair.** Development signal, not a result.

**Two of the three supersession holes remain open:**

1. ~~No version filter is wired.~~ **Closed by this work.**
2. h-01 still carries no `must_not_cite`, so `max_supersession_violations: 0` is still
   vacuous and the gate still cannot catch a regression here. A labelling act.
3. Nothing produces `AbstainReason.ONLY_SUPERSEDED_EVIDENCE`.

The honest claim is now narrower and true: **the retriever no longer returns the superseded
wording when asked for a date it does not cover. The eval still cannot prove it didn't.**
