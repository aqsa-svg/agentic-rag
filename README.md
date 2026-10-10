# agentic-rag — retrieval over Indian health insurance policy wordings

A question-answering system over six real insurance documents: two versions of a Star
Health policy wording, two Niva Bupa wordings, and two IRDAI regulatory documents. 197
pages, 2,485 chunks (row-NL, the index T3 selected; 1,024 under the markdown serialisation it
replaced), no synthetic corpus.

**The thing being demonstrated is not the pipeline. It is the evaluation.** The harness was
built before the retriever, every number below came out of it, and every claim in this
README links to the run that produced it.

---

## Why this domain

A non-expert can verify ground truth from the document itself. That is rarer than it
sounds, and it is the whole reason the golden set is checkable rather than plausible.

It also punishes the failure mode that matters. "Is my surgery covered?" has a rupee
answer, and a fluent wrong one is worse than a refusal — so the system is built to abstain
rather than guess, and the eval is built to tell the difference.

## Measured results

Every figure is from a recorded run. Nothing here is estimated.

### Retrieval, `top_k=10` — the 7-item v1 baseline era

These runs were measured on the **7-item v1 set**, before the golden set reached its 30-item
target (below). A re-run on all 30 is owed and is v2 work. The **nDCG column was re-scored
after the 2026-10-08 metric fix** — the old metric over-credited a span covered by more than
one retrieved chunk (`docs/SILENT_WRONGNESS.md`, instance 4), so struck figures are the
inflated originals and the bold ones are corrected from the stored per-span ranks:

| | recall@10 | nDCG@10 | MRR | ctx precision | hit rate |
|---|---|---|---|---|---|
| **v1r** BM25 only | 0.400 | 0.235 | 0.207 | 0.060 | 0.600 |
| **v2** dense (`bge-base-en-v1.5`) | **0.600** | ~~0.420~~ **0.383** | 0.327 | **0.120** | **0.800** |
| **v2** RRF hybrid (`k=60`) | 0.400 | ~~0.383~~ **0.330** | 0.350 | 0.080 | 0.600 |
| **v3** dense + supersession filter | **0.700** | ~~0.526~~ **0.383** | 0.375 | **0.160** | 0.800 |
| **v4** + cross-encoder rerank | **0.700** | ~~0.595~~ **0.395** | **0.440** | **0.160** | 0.800 |

BM25's 0.235 is unchanged — lexical retrieval returned no duplicate per-span chunks, so there
was nothing to over-credit. The ordering is untouched by the fix: no conclusion moved, which
is itself recorded (`docs/V2_MEASUREMENT.md`, `docs/V4_RERANK.md`).

Full records: [`docs/BASELINE_V1R.md`](docs/BASELINE_V1R.md), [`docs/V2_MEASUREMENT.md`](docs/V2_MEASUREMENT.md),
[`docs/V3_SUPERSESSION.md`](docs/V3_SUPERSESSION.md), [`docs/V4_RERANK.md`](docs/V4_RERANK.md).
Raw: `data/baseline_v1r.json`, `data/v2_measure.json`, `data/v3_supersession.json`,
`data/v4_rerank.json`.

**v4's reranker is built, measured and deliberately NOT SHIPPED.** It is still the best ranking quality here — nDCG ~~0.595~~ **0.395** vs dense's ~~0.526~~ **0.383** (corrected 2026-10-08: the real gain is **+0.012**, not the +0.070 the over-credited metric implied), MRR 0.440 — and it costs **4,108 ms against DESIGN §5.5's 60 ms budget**: 68x over, more than the entire 2.5s flat-lookup p95 allowance in a single stage. The correction *reinforces* the decision — a smaller gain for the same cost — rather than changing it, because it rested on latency and never on the nDCG value. The online configuration remains **v3 — dense + supersession filter, 208 ms** — and the reranker is an offline quality ceiling until the ONNX-INT8 path closes the gap.

The script that measured it printed `EARNS IT`, because it tested `nDCG > 0` rather than DESIGN's `nDCG > 0 within 60ms`. That is the **third** guard in this project found green while the thing it existed for walked past — see [`docs/SILENT_WRONGNESS.md`](docs/SILENT_WRONGNESS.md), "A THIRD pattern".

**These remain development-signal numbers, not results** — but no longer because the set is
unfinished. ~~Seven labelled items is far below the composition gate, `arag-eval validate`
says `NOT YET PUBLISHABLE`.~~ The golden set reached its target on 2026-10-08 — **30 of 30,
all human-authored, every stratum floor met, `arag-eval validate` reports no shortfall** — and
these retrieval runs predate it on the 7-item set. They stay directional for a different
reason: at the per-stratum sizes one item moves a stratum 20–50 points, so per-stratum figures
are not quotable, and the 30-item retrieval re-run is still owed. See
[Honest scope](#honest-scope).

### Three findings worth more than the numbers

**The hybrid loses to dense alone on four of five metrics.** Mechanism, from h-28: dense
finds a labelled span at rank 3 and fusion pushes it out of the top 10. RRF discards score
magnitude by design, so BM25's confident *wrong* ranking is weighted exactly as heavily as
dense's correct one. Left unswept at the documented default rather than tuned after seeing
the numbers.

**A pre-registered prediction failed.** DESIGN §11 stated in advance that BM25 would fail
vocabulary mismatch and that dense retrieval was "precisely the gap" that would close it.
h-03 — *"I bought the policy five days ago and now I have a fever"* against *"30-day
waiting period — Code Excl 03"* — still scores **0.000 under dense**. The chunk is
reachable and simply is not retrieved. Recorded as a failed prediction, not quietly
dropped.

**A feature can be wired end to end and still be inert.** The `as_of` version filter was
complete and correct at every layer — runner, engine, both retrievers, the date logic — and
did nothing, because **nothing ever wrote the metadata it reads**. Across all 1,024
markdown-serialised chunks, `effective_to`, `superseded_by`, `insurer` and `product` were
empty, always; `Source` did
not even have an `effective_to` field. Retrieving h-01 with and without `as_of` returned the
identical top 10.

Fixed, and measured as a delta rather than switched on and admired: **dense recall
0.600 → 0.700, nDCG 0.420 → 0.526, stale chunks in the top-k 14 → 0**, no metric regressing
on any engine. h-01's rank-1 result — the superseded 2021 definition of "Hospital", the
highest-priority finding since the first baseline — is now the current one.

This is the mirror of the audited pattern: *a signal consumed and never produced.*
`tools/audit_signals.py` scored those fields healthy because it only tracks reads.

Two of the three supersession holes remain. The honest claim is narrow and true: **the
retriever no longer returns the superseded wording. The eval still cannot prove it didn't**,
because no label carries a `must_not_cite`.

## What this project is actually about

### Eleven silent-wrongness instances, found and recorded

[`docs/SILENT_WRONGNESS.md`](docs/SILENT_WRONGNESS.md) — bugs sharing one shape:

> **The code runs. Nothing errors. The output looks right. It is wrong.**

The two that cost the most:

**Instance 9** — the chunker recorded a clause's heading number (`3`) while the golden set
used the IRDAI code (`excl.03`). **0 of 1,024 markdown-serialised chunks carried any `excl.NN`
id**, so three of
seven items could not score however well retrieval performed. Every metric published before
the fix was measured through a matcher blind to them — and the regression gate derived from
those numbers *could not catch the regression it existed for*: a change destroying one
item's retrieval entirely would have scored 0.300 against a 0.30 floor and **passed**.

**Instance 10** — the prompt says "set `answer` to null"; models write the **string**
`"null"`. A four-character string is truthy, so a model's correct refusal was recorded as an
answer. A false confident response, introduced by the layer whose only purpose is preventing
them. Found by the first run against a real model, after 60 passing tests.

### A second pattern: signals produced and never consumed

`LLMError.retryable`, `LLMError.retry_after_s`, `LLMResponse.truncated` — all populated
correctly, all covered by passing tests, all read by nothing. The third failed 5 of 7
questions in the first live run by reprompting a truncation with the same token budget,
which cannot succeed.

[`tools/audit_signals.py`](tools/audit_signals.py) walks the AST for attribute *loads* and
found **12 signals with no production consumer**, split by risk — `ONLY_SUPERSEDED_EVIDENCE`
and `injection_suspected` read as implemented defences and are not. Ratcheted by
`tests/test_signal_consumption.py`: a new unread signal fails the build, and so does one
that becomes consumed without being removed from the list.

### Guards that have been observed to fail

Every architecture test added after instance 7 was **mutation-tested** — the defect
reintroduced, the test confirmed to fail, the defect removed. A guard never observed failing
is not known to be a guard.

### A design assumption, measured against the labels and discarded before any agent code existed

The abstention design put part of the decision on a **pre-generation sufficiency gate**: dense
top-1 cosine plus the top1-top2 gap, below a calibrated threshold → abstain. Before writing
it, the signal was measured against all 30 labelled items — the 5 `unanswerable` as the
should-decline set, the 25 others as should-answer, because a threshold tuned only on items
that should abstain over-abstains, and over-abstention is the failure that looks like safety.

**It does not separate them.** The should-abstain top-1 range `[0.614, 0.715]` sits *entirely
inside* the should-answer range `[0.615, 0.867]`; h-01 — answerable, retrieved at recall 1.0 —
scores 0.615, below every one of the five items that should decline. The best trade-off point
catches 3 of 5 unanswerables while wrongly declining 5 of 25 answers. The cause is one
sentence: **a bi-encoder scores *aboutness*, and the decoys are about the topic** — Zone A-E
is about geography, Section 80-D is about tax; they are on-subject and simply do not answer the
question, so the geometry cannot rank them below a real answer.

The corollary is the part worth keeping: **the absence of a usable operating point is evidence
the unanswerable items are good.** A decoy a bi-encoder could separate on topicality alone
would be a weak decoy — an abstention item that easy probes nothing. A weaker decoy set would
have produced a clean threshold and a gate that passed this measurement and then **failed in
production** on the first real question whose near-misses were on-topic.

So the gate was **deleted before it was built**, not switched off after. Abstention moved
post-generation (the verifier requires every claim grounded and the answer responsive); the
pre-generation entry shrank to the degenerate case — empty candidate set → abstain. Knock-on:
the day-7 threshold sweep no longer exists; **day 7 is the cost-ratio sweep alone.** Full
numbers and the trade-off curve in [`docs/DESIGN.md`](docs/DESIGN.md) §5.7.

## Architecture

```
ingest → chunk → index ──┬── BM25 (lexical)      ──┐
                         └── dense (bge, exact)  ──┴── RRF fusion → generate → VERIFY → answer | abstain
```

Three boundaries are enforced by AST-walking tests, not convention:

- only `arag.agent` may import LangGraph — makes "framework choice is reversible" checkable
- `arag.eval` may not import a concrete engine — what made harness-before-pipeline possible
- the online request path may not reach offline-only dependencies (torch, PyMuPDF),
  **over the full import closure**, not per file

That last one was added after instance 7, where `arag.retrieval.lexical` → two dataclasses →
a package `__init__` → PyMuPDF reached the serverless path while every per-file check passed.

### Generation is three stages, and the middle one is adversarial

Retrieve → generate → **verify**. The model cites passage *integers*, never chunk ids: asked
to reproduce `star-comprehensive-2025:p32:s146` a model will occasionally emit a plausible
variant, which is a hallucinated citation that passes a regex and points at a real, wrong
chunk. An integer is either in range or rejected. An answer with any ungrounded citation is
**discarded**, not annotated.

Its limit is documented rather than oversold: the check verifies a citation points at a
*retrieved* passage, **not that the passage supports the claim**. See
[`LIMITATIONS.md`](LIMITATIONS.md).

## Honest scope

**Built and measured:** ingest, chunking, clause index, BM25, dense retrieval, RRF fusion,
the eval harness, the regression gate, generation with citation verification.

**Built, not measured against the production model:** generation. Three attempts died on
account state — a retired model, a 20/day free quota, an un-prefixed environment variable,
and a suspended Google project. Verified end to end against a local 1.5B model instead,
which found two defects a scripted test could not. Those are pipeline-verification numbers,
never quality numbers.

**Not built:** LangGraph agent and its injection envelope, PII redaction, the cost sweep,
chaos engineering, deployment. The injection stratum (5 items) and its adversarial fixture
*are* built; the agent-side defence that would consume them is not.

**Measured, then deliberately not built:** ~~abstain calibration~~ the pre-generation abstain
threshold. The sufficiency signal did not separate the labelled populations, so the threshold
was dropped before any agent code existed and abstention moved post-generation — see the
subsection above. The day-7 sweep is now the cost-ratio one alone.

**The golden set reached its target on 2026-10-08: ~~7 items of a 40 target~~ 30 of 30, all
human-authored, every stratum floor met** (`arag-eval validate` reports no shortfall). The
total was cut 120 → 40 → 30 as the labelling rate was measured — see `SetTargets`, which
records what each cut costs — and raising it again is a v2 item, not a v1 one.

The adversarial floors were themselves measured down to what the corpus actually contains,
and the sharpest case is **contradictory, which fell 5 → 3 → 2 → 1.** IRDAI's 2024
standardisation mandates verbatim-identical wording for grace period, free look, moratorium,
cancellation, portability and claim settlement, which collapsed most cross-insurer
contradictions (5 → 3 → 2 across a systematic survey, 2026-10-07); the final candidate then
dropped when its text turned out to be a **list heading on a claim form, not an obligation
clause** (2 → 1, 2026-10-08). The one
survivor is h-18 (grace-period cover); the other grace-period item, h-25, is filed under
`cross_doc_comparison` because its test is provenance, not conflict. **That the floor is 1 is a
finding about the corpus, not a shortfall in labelling** — stated unsoftened in
[`LIMITATIONS.md`](LIMITATIONS.md), because the number reads as a gap if it appears without it.

Everything downstream is gated on the set, and it is hand-labelled by a human reading the PDFs,
deliberately — labels written by a model and marked `authored_by: human` would falsify the
claim the whole set rests on. Two of fifteen strata exist *because* hand-labelling found
failure modes no scan had proposed.

## Running it

```bash
make install            # or: pip install -e ".[dev,offline]"
arag-ingest fetch       # downloads the 4 insurer wordings (never committed — copyrighted); the
                        # 2 public IRDAI regulatory PDFs ARE committed, for the public deploy (§10.1)
arag-ingest clause-index
arag-eval validate      # schema + corpus resolution + composition
arag-eval run --engine bm25
make test               # 606 tests
```

Generation needs `ARAG_GEMINI_API_KEY` in `.env` — note the `ARAG_` prefix; the bare vendor
name is read by nothing, and `arag.config.misprefixed_credentials()` exists because that
mistake cost a full verification run.

## Documents

| | |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | architecture, 15 strata, trade-offs with the rejected alternative named |
| [`docs/SILENT_WRONGNESS.md`](docs/SILENT_WRONGNESS.md) | eleven instances, four patterns, sixteen practices |
| [`LIMITATIONS.md`](LIMITATIONS.md) | what does not work, measured |
| [`docs/BASELINE_V1R.md`](docs/BASELINE_V1R.md) | the frozen baseline, and why it was re-frozen |
| [`docs/V2_MEASUREMENT.md`](docs/V2_MEASUREMENT.md) | dense + hybrid, corrected in place with the original struck |
| [`docs/V3_SUPERSESSION.md`](docs/V3_SUPERSESSION.md) | the version filter, and the delta it was deferred for |
| [`docs/V4_RERANK.md`](docs/V4_RERANK.md) | the reranker: quality gain accepted, latency rejected |
| [`docs/EVAL_EXPLAINED.md`](docs/EVAL_EXPLAINED.md) | the metrics from first principles, hand-worked |
| [`docs/BUILD_RECORD.md`](docs/BUILD_RECORD.md) | which code was reviewed and which was not |
