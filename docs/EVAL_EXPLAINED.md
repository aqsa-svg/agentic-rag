# What an evaluation harness is, and why this one looks like this

Written for someone who has shipped backend services but never built an eval harness. The
goal is that you can explain every design decision in `src/arag/eval/` from first
principles, because possessing code you cannot defend is worse than not having it.

---

## 1. The problem an eval harness solves

In ordinary backend work, correctness is mostly **binary and local**. `POST /users`
either creates a user or it doesn't. You write a test, it passes, and the test keeps
passing forever unless someone breaks it.

An LLM system has neither property:

- **Correctness is graded, not binary.** "Bariatric surgery is covered subject to limits"
  and "Bariatric surgery is covered up to Rs 2,50,000" are both plausible; one is right for
  the 2021 wording and one for the 2025 wording. There is no `assertEqual` for this.
- **Every change is global.** Change the chunk size and *every* answer changes. Change the
  prompt and every answer changes. Swap the embedding model and every answer changes. There
  is no such thing as a local change.

So the unit of confidence cannot be "this test passes". It has to be **a distribution of
scores over a fixed set of questions, compared against the same measurement taken before
the change.** That is what an eval harness is: a fixed dataset, a set of scoring functions,
a runner, and a stored history so two runs can be compared.

The consequence people underestimate: **if you build the pipeline first, you will choose
thresholds that make your pipeline look good.** Not dishonestly — you'll just have no
independent reference for what "good" is. That's why DESIGN.md builds the harness first
and runs it against a stub that returns nothing (`NullEngine`). Every metric reads exactly
0.0, which proves the harness can register total failure *before* there is anything to
flatter.

---

## 2. The vocabulary, mapped to this repo

| Term | What it means | Where it lives |
|---|---|---|
| **Golden set** | A fixed, human-labelled dataset of questions plus what a correct response looks like. The reference the system is measured against. | `data/golden/v1.jsonl` (empty — awaiting your 30 labels) |
| **Stratum** | A category of question. Metrics are reported per stratum, not just overall. | `Strata` in `schema.py` (15 of them, 4 adversarial) |
| **Ground truth span** | The location in the corpus where the answer lives — document + clause. Not the answer *text*. | `DocSpan` in `retrieval/types.py` |
| **Retrieval metrics** | Did the retriever find the right passage? Computed from spans, no LLM needed. | `metrics/retrieval.py` |
| **RAGAS metrics** | Four LLM-scored measures of generation quality. | `metrics/ragas_runner.py` (v1) |
| **LLM-as-judge** | Using a model to score answers a human labelled. | `judge` config, `metrics/agreement.py` |
| **Regression gate** | A CI check that fails the build when a metric drops. | `thresholds.py` + `.github/workflows/ci.yml` |
| **Cassette** | A recorded provider response, replayed so CI is deterministic and free. | `cassettes.py` |

---

## 3. Retrieval metrics, from first principles

These are the important ones because in document QA **most wrong answers are retrieval
failures, not generation failures.** If the right clause never reaches the model, no amount
of prompt engineering saves you.

All four are computed by comparing the ranked list of retrieved chunks against the
ground-truth spans a human chose. No LLM, no cost, fully deterministic — which is exactly
why they carry the CI gate.

Say the golden item names **2** correct clauses (A and B), and retrieval returns 5 chunks
in this order: `X, A, Y, B, Z`. Relevance vector: `[0, 1, 0, 1, 0]`.

**recall@k — "did we find the clauses at all?"**
```
recall@5 = (distinct ground-truth spans found) / (total ground-truth spans) = 2/2 = 1.0
```
*Distinct* matters. Three chunks all covering clause A is **one** span found, not three —
otherwise a retriever returning near-duplicates would outscore a precise one. This is the
metric that catches multi-hop failure: the system reliably finds the inclusion clause and
never the waiting-period table, giving 0.5 forever.

**precision@k — "how much junk did we send the model?"**
```
precision@5 = (relevant chunks in top-k) / (chunks actually returned) = 2/5 = 0.4
```
The denominator is what was *returned*, not `k`. The question is "how polluted is the
context window", and an unfilled slot pollutes nothing.

**MRR (Mean Reciprocal Rank) — "how far down was the first hit?"**
```
MRR = 1 / (rank of first relevant chunk) = 1/2 = 0.5
```

**nDCG@k — "is the ranking good, not just the set?"** The headline metric.
```
DCG  = sum over positions of  relevance / log2(rank + 1)
     = 1/log2(3) + 1/log2(5)            (hits at ranks 2 and 4)
     = 0.6309 + 0.4307 = 1.0616

IDCG = the same, if both hits were at ranks 1 and 2 (the ideal ordering)
     = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309

nDCG = 1.0616 / 1.6309 = 0.651
```
The `log2(rank+1)` divisor means a hit at rank 1 is worth more than a hit at rank 5. nDCG
is the headline because **it is the only one of the four that moves when a reranker
reorders a fixed candidate set** — and reordering is the entire purpose of a reranker. If
nDCG doesn't improve, the reranker's 60ms is not earning its place.

Every one of these numbers is asserted in `tests/test_metrics_retrieval.py` against the
arithmetic written above. If the metric is subtly wrong, every number the project ever
publishes is wrong, and asserting against "whatever the code returns today" would lock the
bug in rather than catch it.

### The convention worth being able to defend: undefined is not zero

An unanswerable question has no ground-truth span. So retrieval quality on it is not
zero — it is **undefined**. Every metric function returns `None`, and the aggregator skips
it and reports the denominator it actually used (`scored=12  retrieval-n/a=4`).

Why this matters: if unanswerable items scored 0.0, then **every adversarial item you add
to the golden set would drag reported recall down.** A more rigorous eval set would produce
worse numbers. That inverts the incentive the entire project rests on.

---

## 4. RAGAS: the four generation metrics

RAGAS is a library that scores generation quality using an LLM. Four metrics, and the
useful thing is that they **split retrieval blame from generation blame**:

| metric | question it answers | blames |
|---|---|---|
| **context recall** | Did retrieval fetch everything needed to answer? | retrieval |
| **context precision** | Was the fetched context mostly relevant, and ranked well? | retrieval |
| **faithfulness** | Is every claim in the answer supported by the retrieved context? | generation |
| **answer relevancy** | Does the answer actually address the question asked? | generation |

The diagnostic value is in the *combination*:

- Low context recall + high faithfulness → **retrieval is the bottleneck.** The model is
  being honest about the inadequate context it was given. Fix retrieval.
- High context recall + low faithfulness → **the model is hallucinating** despite having
  the right passages. Fix the prompt or the model.
- High faithfulness + low answer relevancy → the model is faithfully answering a
  *different* question.

This is why DESIGN §5.6 rejects fine-tuning the generator: if v1 shows high faithfulness
and low context recall, generator fine-tuning is **provably** irrelevant — the generator is
already doing the right thing with bad inputs. Being able to name the metric that would
change your mind is what makes that a decision rather than an opinion.

---

## 5. LLM-as-judge, and why κ is non-negotiable

Some things can't be scored mechanically — "is this answer correct?" needs judgement. The
standard move is to have a model do it.

The trap: **a weak judge fails in the flattering direction.** It tends to accept answers a
careful human would reject, so your metrics look good and mean nothing. And you cannot
detect this by looking at the metrics, because the metrics are the thing being corrupted.

DESIGN §5.8 accepts a free local judge on your own GPU on one condition: measure its
agreement with you. That's **Cohen's kappa**.

Why not simple agreement percentage? Suppose 80% of answers are genuinely correct. A judge
that says "correct" to literally everything agrees with you 80% of the time — while
carrying **zero information**. Raw agreement rewards that. Kappa corrects for agreement
expected by chance:

```
po = observed agreement          (fraction of items you both scored the same)
pe = agreement expected by chance (from each rater's own label frequencies)

kappa = (po - pe) / (1 - pe)
```

For that lazy judge: `po = 0.8`, `pe = 0.8`, so `kappa = 0/0.2 = 0.0`. Correctly worthless.

Bands used here: ≥0.80 strong, 0.60–0.79 moderate, <0.60 **the metrics are not evidence**
and we switch to a paid judge. The 0.60 line is enforced in `thresholds.yaml`, and
`arag-eval judge-agreement` exits 1 below it.

You will hand-score 25 stratified items to produce this number. It is roughly an hour and
it converts your cheapest decision into the strongest artefact in the project, because
almost no portfolio RAG project can tell you whether its judge is any good.

---

## 6. Why the regression gate is shaped the way it is

**Threshold profiles are versioned, not edited.** `thresholds.yaml` has `v0_floor`,
`v1_baseline`, and so on. The failure mode this prevents: a regression gets "fixed" by
lowering the threshold in the same PR, and git history then shows a green build with no
record that quality dropped. Bumping `active_profile` is a reviewable act; quietly editing
a number inside a shipped profile is not.

**A missing metric fails a configured threshold.** If a metric key disappears (someone
renames it), the gate fails rather than passes. Treating absent-as-passing is the classic
way a gate rots into decoration.

**Exit codes are separated**: `0` pass, `1` quality regression, `2` the harness itself is
broken. A CI job that returns the same code for "retrieval got worse" and "the golden set
is malformed" trains people to re-run the build instead of reading it.

**Cassettes record at the provider boundary, not the answer boundary.** This one is worth
understanding properly because the obvious design is useless. If CI replayed stored
*answers*, the engine would never run, so **no change to retrieval or prompting could ever
move a metric.** The gate would stay green forever while quality rotted, detecting only
bugs in the metric code while presenting itself as an end-to-end quality gate — worse than
no gate, because it manufactures confidence.

So cassettes record individual LLM/embedding/judge HTTP calls. The real planner runs, real
fusion runs, the real prompt is assembled; only the network is replayed. And the cassette
key is a hash of the **entire request payload including the prompt text**, so editing a
prompt invalidates the recording and fails loudly rather than silently scoring the new
prompt against the old response.

---

## 7. The three things an interviewer is most likely to probe

1. **"How do you know your golden set is any good?"** Answer with the provenance field:
   every item carries `authored_by`, every metric slices by it, and the runner prints a
   warning when unverified labels are present. Aggregate metrics over mixed-provenance
   labels are close to meaningless, and this project can show the human-authored slice
   separately.

2. **"How do you know your judge is any good?"** Answer with κ, published alongside the
   metrics. Most projects cannot answer this at all.

3. **"How do you know your eval isn't circular?"** Answer: questions are drafted from
   page/section windows, never from the retrieval chunks, so context recall isn't measuring
   the chunker against itself; ground truth is a span a human chose, not prose a model
   wrote; and the adversarial strata are entirely hand-authored.

---

## 8. What is still missing

Stated plainly because the harness's own limitations are part of understanding it:

- **No labelled data.** The gate is vacuous until 30 human-labelled items exist.
- **No κ measurement.** Impossible until there are labels to agree about.
- **RAGAS not yet wired.** The runner scores retrieval and behaviour; the four RAGAS
  metrics land in v1 when there is generation to score.
- **The PR gate never touches a real database.** See LIMITATIONS.md.
- **One `concept_id` per item, but `clause_tension` items span two kinds of clause.**
  Open; not being fixed yet. The schema carries a single `concept_id` and the resolver
  checks it against **every** span, so an item whose spans are deliberately of different
  kinds has no correct value to put there.

  Worked example, h-21. Its two spans are star-comprehensive-2025 p35 `excl.32` ("Dental
  treatment or surgery ... except to the extent covered under Section II.17") and p16
  clause 17, the out-patient dental **benefit** that the exclusion cross-references. The
  tension *is* the pair; dropping either span removes the item's reason to exist.
  `exclusion.permanent` is wrong about p16 and `benefit.dental_opd` is, strictly, wrong
  about p35.

  h-21 resolves today only because the exclusion happens to use the benefit's words, so
  the `"dental treatment"` anchor reaches both pages. That is luck, not mechanism: an
  insurer wording the same exclusion as "dental surgery" would break it, and the next
  `clause_tension` item is as likely to fall the other way.

  The fix would be to move `concept_id` onto the span rather than the item. It is not being
  made now, on purpose - with one item in the stratum, a schema change would be designed
  against a single example. The decision waits until the remaining four land and show
  whether per-span concepts are the general shape or whether h-21 is the odd one.
