---
title: Agentic RAG IRDAI
emoji: 📄
colorFrom: blue
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
---

<!-- The YAML above is the Hugging Face Space config (Gradio SDK, free CPU), and it also lets
     this repo be pushed to a Space as-is; app.py is the entrypoint. GitHub renders it as a
     small table. Everything below is the project write-up. -->

# agentic-rag — retrieval-augmented QA over Indian health-insurance policy wordings

A question-answering system over six real insurance documents — two versions of a Star Health
wording, two Niva Bupa wordings, and the two IRDAI 2024 regulatory documents. 197 pages, 2,485
row-NL chunks, no synthetic corpus.

**The artifact is the evaluation, not the pipeline.** The harness was built before the
retriever; every number here came out of it; every claim links to the run that produced it. And
the most useful results are the negative ones — three different abstention signals were each
measured against the labelled set and each failed, and four features were built or specified and
then deliberately dropped on measurement.

> **All generation and judge numbers below are from a local Qwen2.5-1.5B instrument on CPU.** It
> was chosen to exercise the pipeline, not to be good: a 1.5B model's answers are not a hosted
> model's. **Production-model results are unmeasured** — three attempts to measure them died on
> API account state (a retired model, a 20/day quota, an un-prefixed env var, a suspended
> project). Read every figure as "what the pipeline does on this instrument," never as a quality
> claim about the system a good model would make.

**Live instance:** **https://agentic-rag-irdai.onrender.com** — the IRDAI-only service on Render
(FastAPI + Docker, `gemini-2.0-flash`). Scope is deliberately narrow and stated on the landing
page; generation is the production model, which is **unmeasured** (see the banner below). A Gradio
Hugging Face Space (`app.py`) is also prepared for a free-CPU deploy. Render's free instance
sleeps when idle, so the first request after a pause cold-starts (~30–60s).

---

## What was measured, and what it showed

### The headline: the full pipeline over 30 items, end to end

The real engine (retrieve → pre_gate → generate → parse → verify → finalise) run over all 30
golden items on the 1.5B instrument, row-NL BM25-dense, `top_k=5` ([`docs/AGENT_EVAL.md`](docs/AGENT_EVAL.md),
`data/eval_30.json`):

| | value |
|---|---|
| answered | **26 / 30** |
| abstained | 4 |
| unanswerables correctly refused | **2 / 5** |
| answers grounded in the labelled span, when it was retrieved | **7 / 12 (58%)** |

Stated plainly, including where it is bad: **the instrument guesses more than it grounds or
refuses** — the opposite of the posture the project sets out ("abstain rather than guess"). The
11 answers that missed their span split into **6 retrieval misses** (the span was never
retrieved) and **5 true grounding failures** (span retrieved, wrong chunk cited). Best stratum:
`table_formula`, 5/6 grounded — evidence for the corpus thesis that *the tables are the answers*
and for the row-NL serialisation choice. Worst: the prose strata (`conditional_override` 0/2) and
abstention.

### Abstention detection failed three different ways — the strongest result here

"Abstain rather than guess" needs a signal that separates the 5 unanswerable items from the 25
answerable ones. **Three independent signals were measured against the labelled set; none
separates them on this corpus.** Each was measured *before* the code that would depend on it was
written.

**1. Retrieval geometry (the sufficiency gate).** Dense top-1 cosine + the top1-top2 gap, below a
threshold → abstain. Measured on all 30 (`data/suff_signal.json`): the should-abstain top-1 range
`[0.614, 0.715]` sits **entirely inside** the should-answer range `[0.615, 0.867]`. An answerable
item retrieved at recall 1.0 scores 0.615, below every item that should decline. **Cause, one
sentence: a bi-encoder scores *aboutness*, and the decoys are about the topic** (Zone A-E is about
geography; Section 80-D is about tax). The corollary is worth more than the gate: the *absence* of
an operating point is evidence the unanswerable items are good — a decoy a bi-encoder could
separate on topicality alone would be too easy to probe anything. Gate deleted before it was
built.

**2. The model's self-report.** Add `can_answer` + `confidence` to the schema and believe it.
Measured over all 30 (`data/selfreport.json`): of the 5 unanswerables it catches **2**; the other
3 it answers at **confidence 1.0** (one replies "Yes" to overseas hospitalisation the corpus does
not cover). Confidence is near-binary — 28 of 30 values are 0.0 or 1.0 — and **no threshold on it
beats the binary flag**. The model is most certain on exactly the questions it should refuse.

**3. An independent grounded-and-answers judge.** A separate call scoring whether each cited
passage supports the claim and whether the answer is responsive — justified by (2), since the
generator cannot grade itself. Validated on the same 30 answers against the same instrument
(`data/judge_validate.json`): it catches **2/5** abstains and **wrongly declines 9/25** answers —
**strictly worse than self-report** (+7 wrong declines, 0 extra catches), 6 of those 9 on items
whose span was retrieved and answered correctly. Same aboutness confusion as the sufficiency
gate. Pre-registered rule: gate only if it strictly beats self-report on both numbers; it beats
neither, so node 5(b) is wired but **advisory — logged, never gating** — and a stronger judge is
the one untested lever that could change this.

The honest consequence: on the available instrument, nothing reliably tells answerable from
unanswerable, so the hard unanswerables are answered. That is a measured property of a weak judge
on a good decoy set, not a bug to patch.

### Retrieval, `top_k=10` — the 7-item v1 baseline era

Measured on the 7-item v1 set, before the golden set reached 30 (a 30-item re-run is owed). The
nDCG column was re-scored after the 2026-10-08 metric fix (struck = the over-crediting original,
bold = corrected from stored per-span ranks):

| | recall@10 | nDCG@10 | MRR | ctx precision | hit rate |
|---|---|---|---|---|---|
| **v1r** BM25 only | 0.400 | 0.235 | 0.207 | 0.060 | 0.600 |
| **v2** dense (`bge-base-en-v1.5`) | **0.600** | ~~0.420~~ **0.383** | 0.327 | **0.120** | **0.800** |
| **v2** RRF hybrid (`k=60`) | 0.400 | ~~0.383~~ **0.330** | 0.350 | 0.080 | 0.600 |
| **v3** dense + supersession filter | **0.700** | ~~0.526~~ **0.383** | 0.375 | **0.160** | 0.800 |
| **v4** + cross-encoder rerank | **0.700** | ~~0.595~~ **0.395** | **0.440** | **0.160** | 0.800 |

Three findings from these runs outlast the numbers: the **hybrid loses to dense** (RRF discards
score magnitude, so BM25's confident wrong ranking weighs as much as dense's right one); a
**pre-registered prediction failed** (h-03, *"I bought the policy five days ago and now I have a
fever"* vs *"30-day waiting period — Code Excl 03"*, still scores 0.000 under dense — recorded,
not dropped); and a **feature wired end to end was inert** — the `as_of` version filter was
correct at every layer and did nothing because nothing wrote the metadata it read (fixed, then
measured as a delta: dense recall 0.600 → 0.700, stale chunks in top-k 14 → 0).

Full records: [`docs/BASELINE_V1R.md`](docs/BASELINE_V1R.md), [`docs/V2_MEASUREMENT.md`](docs/V2_MEASUREMENT.md),
[`docs/V3_SUPERSESSION.md`](docs/V3_SUPERSESSION.md), [`docs/V4_RERANK.md`](docs/V4_RERANK.md).

## Four deliberate non-adoptions, each killed by a measurement

The discipline the project is built to demonstrate: a thing is left out because a number says so,
and the number is recorded so a later reader does not add it back as an oversight.

| Not adopted | The measurement that killed it |
|---|---|
| **Cross-encoder reranker** | Best ranking quality here — but **+0.012 nDCG for 68× the DESIGN §5.5 latency budget** (4,108 ms vs 60 ms). The gain is real; the latency is disqualifying. Online config stays dense + supersession filter at 208 ms. ([`docs/V4_RERANK.md`](docs/V4_RERANK.md)) |
| **Pre-generation sufficiency gate** | Dense top-1 + gap do not separate the 5 unanswerables from the 25 answerables (above). No threshold worth setting, so the geometric abstain-entry shrank to the degenerate empty-set case. |
| **LangGraph** | The control flow is **linear with failure edges**; no labelled item needs conditional routing (no ReAct loop, no multi-hop planner, no reflect node — none has an item to justify it). A procedural `GeneratingEngine` already did the whole flow, so the dependency + checkpointer + graph API bought nothing. Specified in DESIGN §5.7, then dropped; the import-boundary test flipped from "only `arag.agent` may import it" to "**nothing imports it**." |
| **A datastore / checkpointer** | With LangGraph gone there is no graph state to checkpoint, and a database for a demo audit log buys nothing the structured stdout trace does not — every request mints a `trace_id` carried on the `structlog` JSON line. The deployed service is stateless; the rate limit is in-memory. |

## The golden set as a discovery method

30 human-authored items, every span checked against a wording, zero synthetic. It is hand-labelled
on purpose — labels written by a model and marked `authored_by: human` would falsify the one claim
the whole set rests on — and the labelling is where the sharpest findings came from:

- **Two of the fifteen strata were found while labelling, not during design.** `conditional_override`
  (stratum 14) surfaced from reading two clauses on adjacent pages — a figure real but gated on
  something the policyholder lacks by default. `supersession` (stratum 15) surfaced when a labelled
  span pointed at a clause code the index had *silently dropped*, which led to three deleted
  exclusions, one of which was a polarity reversal, not a deletion. Neither was visible to any scan
  written beforehand. That is hand-labelling as discovery, not data entry.
- **The `contradictory` floor fell 5 → 3 → 2 → 1, and the floor is the finding.** IRDAI's 2024
  standardisation mandates verbatim-identical wording for grace period, free look, moratorium,
  cancellation, portability and claim settlement, which collapsed most cross-insurer
  contradictions; the last candidate dropped when its text turned out to be a list heading on a
  claim form, not an obligation clause. One genuine cross-insurer conflict survives. **A floor of 1
  is a fact about the corpus, not a shortfall in effort** — stated unsoftened in LIMITATIONS,
  because the number reads as a gap without it.

`arag-eval validate` reports no shortfall: 30/30, all human-authored, every stratum floor met. The
total was cut 120 → 40 → 30 as the labelling rate was measured; raising it is a v2 item.

## Silent wrongness, and the meta-finding

[`docs/SILENT_WRONGNESS.md`](docs/SILENT_WRONGNESS.md) records every bug of one shape — **the code
runs, nothing errors, the output looks right, and it is wrong** — across four patterns and
seventeen now-structural practices. The costliest: a chunker that stored a clause's heading number
while the golden set used the IRDAI code, so **0 of 1,024 chunks carried any `excl.NN` id** and the
regression gate derived from those numbers could not catch the regression it existed for; and a
model writing the **string** `"null"` where the prompt asked for JSON `null`, so a correct refusal
was recorded as a confident answer — introduced by the layer meant to prevent exactly that.

**The meta-finding, stated plainly: across this work the instruments were wrong more often than
the system they measured.** A probe that stringified a tuple and reported "0 missing"; an nDCG
check that printed `EARNS IT` against `nDCG > 0` instead of DESIGN's `nDCG > 0 within 60 ms`; a
production nDCG that over-credited multi-chunk spans and read >1; a measurement wrapper that let a
syntax error exit 0; a self-report run that stored its verdicts but not the answers they judged, so
a second instrument could not be run against them without re-paying generation. Each is recorded
where it happened. The practices that caught them — assert every metric against a hand-computed
value, trust no diagnostic until checked against the artefact it measures, a candidate in prose is
a claim until both sides are read verbatim — are the actual product.

## Architecture

```
ingest → chunk → index ──┬── BM25 (lexical)      ──┐
                         └── dense (bge, exact)  ──┴── RRF fusion → generate → VERIFY → answer | abstain
```

Three boundaries are enforced by AST-walking tests, not convention:

- **nothing imports LangGraph** — it is non-adopted (§5.7), and the test is the trip-wire that keeps
  re-adoption a decision rather than a drift
- `arag.eval` may not import a concrete engine — what made harness-before-pipeline possible
- the online request path may not reach offline-only dependencies (torch, PyMuPDF), **over the full
  import closure**, not per file — added after instance 7, where the online path reached PyMuPDF
  transitively while every per-file check passed

**Generation is three stages and the middle one is adversarial.** Retrieve → generate → **verify**.
The model cites passage *integers*, never chunk ids (a reproduced id can be plausibly wrong; an
integer is in range or rejected), and any answer with an ungrounded citation is **discarded**, not
annotated. Its documented limit: the check verifies a citation points at a *retrieved* passage, not
that the passage *supports* the claim — which is the gap node 5(b) exists to close and, on the 1.5B
instrument, does not (above).

## The public instance — IRDAI-only

The deployed service answers over the **two public IRDAI 2024 documents only** (Master Circular +
Annexure). **The four insurer policy wordings, the full six-source corpus, and every measured
result stay local** — the wordings are copyrighted and the project never ships them, so a public
instance that answered over them would publish them. The IRDAI documents are government regulatory
instruments, freely redistributable, and are the only sources baked into the image; an allowlist
`.dockerignore` keeps the wordings out of the build context. The landing page and the engine name
(`agent-irdai`) state this scope so a live URL cannot be mistaken for the measured system.

Retrieval on the deployed path is BM25 (no embedder) over an IRDAI index built at startup;
generation is the hosted Gemini API. With no `ARAG_GEMINI_API_KEY` the service abstains
`generation_not_configured` honestly; with a key it answers. Endpoint discipline: `POST /ask`
(question + optional `as_of`) returns the full `AnswerResult`; `GET /health`; **no endpoint returns
raw chunks or full documents**; quotes are capped; a per-client rate limit applies.

```bash
# verified reachable locally under uvicorn; one authenticated command takes it public
fly launch --no-deploy --copy-config          # edit app name in fly.toml
fly secrets set ARAG_GEMINI_API_KEY=<key>     # without it, the service abstains, honestly
fly deploy                                     # -> https://<app>.fly.dev   (or: Render Blueprint via render.yaml)

curl -s -X POST "$URL/ask" -H 'content-type: application/json' \
  -d '{"question":"What is the grace period for paying premium in instalments?"}'
```

See [DESIGN §10.1](docs/DESIGN.md) for the deployment record.

## Honest scope

**Built and measured:** ingest, chunking, clause index, BM25, dense retrieval, RRF fusion, the eval
harness, the regression gate, generation with citation verification, the full agent pipeline, the
IRDAI-only deployable service.

**Measured on the 1.5B instrument only; production-model results unmeasured:** every generation,
abstention and judge number. The local model found two defects a scripted test could not, but its
quality figures are never the system's.

**Measured, then deliberately not built:** the four non-adoptions above.

**Not built:** PII redaction, the cost-ratio sweep (day 7), chaos engineering, the ONNX online dense
path, a stronger-judge re-validation. The injection stratum and its adversarial fixture are built;
the agent-side injection envelope ships as *unmeasured hardening* (all five injection items already
pass without it).

## Running it

```bash
make install            # or: pip install -e ".[dev,offline]"
arag-ingest fetch       # the 4 insurer wordings (never committed — copyrighted); the 2 public
                        # IRDAI PDFs are committed, for the deploy (DESIGN §10.1)
arag-ingest clause-index
arag-eval validate      # schema + corpus resolution + composition
arag-eval run --engine bm25
make test               # 660 tests
```

Generation needs `ARAG_GEMINI_API_KEY` (note the `ARAG_` prefix; the bare vendor name is read by
nothing — `arag.config.misprefixed_credentials()` exists because that mistake cost a verification run).

## Documents

| | |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | architecture, 15 strata, every trade-off with the rejected alternative named, the deployment record |
| [`docs/AGENT_EVAL.md`](docs/AGENT_EVAL.md) | the headline 30-item pipeline run, by stratum |
| [`docs/SILENT_WRONGNESS.md`](docs/SILENT_WRONGNESS.md) | the instances, four patterns, seventeen practices, the instruments-wrong meta-finding |
| [`LIMITATIONS.md`](LIMITATIONS.md) | what does not work, measured |
| [`docs/BASELINE_V1R.md`](docs/BASELINE_V1R.md) · [`docs/V2_MEASUREMENT.md`](docs/V2_MEASUREMENT.md) · [`docs/V3_SUPERSESSION.md`](docs/V3_SUPERSESSION.md) · [`docs/V4_RERANK.md`](docs/V4_RERANK.md) | the retrieval runs, corrected in place with originals struck |
| [`docs/EVAL_EXPLAINED.md`](docs/EVAL_EXPLAINED.md) | the metrics from first principles, hand-worked |
