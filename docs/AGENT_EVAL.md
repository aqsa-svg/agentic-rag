# Agent eval — the full pipeline, 30 items, end to end

**This is the project's headline generation number, and it is an INSTRUMENT result, not a
production one.** Generation and the advisory judge both ran on the local **Qwen2.5-1.5B on
CPU** (`data/eval_30.json`, 2026-10-10), driving the real `GeneratingEngine` built per DESIGN
§5.7: retrieve → pre_gate (empty-set only) → generate (injection envelope) → parse (one repair)
→ verify (5a membership + 5b judge, **advisory**) → finalize. Row-NL dense retrieval, `top_k=5`.
**Production-model numbers are unmeasured** — a 1.5B model's answers are not the hosted model's,
and this run says what the *pipeline* does, not what a good generator would.

## Headline, by stratum

| stratum | n | answered | abstained | grounded* | grounding failures* |
|---|---:|---:|---:|---:|---:|
| table_formula | 6 | 6 | 0 | **5** | 1 |
| definitional_carveout | 4 | 4 | 0 | 1 | 3 |
| supersession | 5 | 3 | 2 | 1 | 2 |
| conditional_override | 2 | 2 | 0 | 0 | 2 |
| clause_tension | 1 | 1 | 0 | 0 | 1 |
| contradictory | 1 | 1 | 0 | 0 | 1 |
| cross_doc_comparison | 1 | 1 | 0 | 0 | 1 |
| injection | 5 | 5 | 0 | n/a | n/a |
| unanswerable | 5 | 3 | 2 | n/a | n/a |
| **total** | **30** | **26** | **4** | **7** | **11** |

\* grounded / grounding failures count only answered items that have a labelled span (the 18
answerable, non-injection items). Injection items (`ignore_injection`) and unanswerable items
have no labelled span, so grounding is not defined for them.

## What the numbers say, plainly — including where it is bad

**The system answers almost everything: 26 of 30.** The project's stated posture is "abstain
rather than guess," and the 1.5B instrument does the opposite — it guesses. All 4 abstentions
came from the model's own decline (`low_confidence`); the advisory judge drove **zero**
abstentions, exactly as designed (it is logged, not gating, because it failed its bar).

**Grounding, decomposed honestly.** Of the 11 answered items that did not cite their labelled
span:
- **6 are retrieval misses** — the span was never retrieved (dense recall 0: h-02, h-03, h-14,
  h-18, h-25, h-32), so the generator could not have cited it. These are retrieval failures, not
  generation failures, and they cap what any generator could do here.
- **5 are true grounding failures** — the span WAS retrieved and the model cited a different
  chunk anyway (h-04, h-28, h-06, h-21, h-22; three of them at recall 1.0). This is the
  membership-passes-support-absent failure from `LIMITATIONS.md`, now measured at the pipeline
  level: the citation is a real retrieved passage, just not the right one.
- **When retrieval works, the generator grounds correctly 7 of 12 times (58%).** That is the
  cleanest single statement of the 1.5B generator's quality on this corpus, with the retrieval
  bottleneck factored out.

**Abstention is the weakest stratum: 2 of 5 unanswerables caught.** h-24 and h-39 were correctly
declined; **h-23, h-38 and h-40 were answered** — confident false answers on questions the corpus
does not address (h-23/h-40 are the overseas-treatment and government-hospital questions). This is
the exact failure the project exists to prevent, and it reaches the output because all three
abstention signals measured this arc — the sufficiency gate (geometry), self-report (confidence),
and the 1.5B judge (grounding) — failed to separate the populations, so nothing gates it. The
envelope shifted which unanswerables the model declines (h-39 now declines, h-23 no longer does)
but did not improve the count.

**Where it does well:** `table_formula`, 5 of 6 grounded — the strongest stratum, and the
evidence for the corpus thesis that *the tables are the answers* (the row-NL serialisation the
T3 decision selected puts the answer cells in reach). **Where it does badly:** every prose
stratum (definitional 1/4, conditional_override 0/2, the three singletons 0/1 each) and
abstention (2/5). The singletons are one item each, so they are directional, not rates.

## Operational notes (not results)

- One item (h-36) **hung for 3+ hours with no output** and was killed; a fresh process completed
  it in normal time, so the 25 already-done items were preserved by the resumable/incremental
  runner and only the hung item was re-run. Several other items ran far over the ~67s median with
  no prompt-size explanation (h-34 1,353s, h-38 560s, and in earlier runs h-04/h-35/h-31). These
  are **unexplained** — prompt size is ruled out (DESIGN §5.7, r=0.22), machine load is an
  uninstrumented hypothesis, not asserted as the cause.
- The run is **not gated in CI**: it needs the offline 1.5B model and is hours long. It is a
  recorded measurement, not a merge gate. The PR gate remains offline/deterministic/`--engine null`.

## The one-line headline

On the 1.5B instrument, the pipeline **answers 26/30 and grounds 7 of the 12 answers whose
evidence was retrieved; it correctly abstains on only 2 of 5 unanswerable questions.** The two
things the project set out to guarantee — ground the answer or refuse — are **not achieved by
this instrument**, and whether a production generator achieves them is unmeasured.

## Production-model probe (n=1, not a rate) — 2026-10-13

The headline above is the 1.5B instrument. The live Render instance generates with
`gemini-2.0-flash`, and two questions were run against it directly — enough to say something, not
enough to measure a rate. This is **n=1 per item on the production model**, recorded as a measured
observation, not a validated figure.

| probe | question | result |
|---|---|---|
| (a) answerable | "grace period when premium is paid in instalments?" | **Answered, grounded** — "fifteen days (monthly) / thirty days (other)", cited to `irdai-master-circular-2024` p7 cl.8. `abstained=false`, generate ~3.2s. |
| (b) out-of-scope | "room rent sub-limit under the Star Comprehensive policy?" | **Abstained** (`low_confidence`, the model declined) — the IRDAI corpus has no product sub-limits, so this is the honest outcome. generate ~1.7s. |

**The abstention failure measured on the 1.5B did NOT reproduce on the stronger model for (b):**
where the 1.5B answered 3 of 5 unanswerables at confidence 1.0, Gemini declined the one
out-of-scope question it was shown. That is one data point, not a rate — it does not establish
that Gemini abstains reliably, only that it did here. A validated production number would need the
full 30-item set run through the hosted model, which is still unmeasured. Until then the landing
page and the README state plainly that **every measured number is from the 1.5B instrument and the
deployed model's behaviour is unmeasured beyond these two probes.**
