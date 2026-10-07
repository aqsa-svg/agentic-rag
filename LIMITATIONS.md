# Limitations

Failure cases and scope decisions that are known, deliberate, and unfixed. Anything here
is a thing a reviewer should be able to ask about and get a straight answer on.

---

## The evaluation gate currently has no data

**Status: blocking. The gate is vacuous.**

The harness runs, the metrics are unit-tested against hand-computed values, and the CI gate
enforces thresholds. But `data/golden/` contains only `TEMPLATE.jsonl` — there are **zero
human-labelled items**. Every number the suite currently produces describes the harness,
not the system.

The 16 seed items written to develop the harness were moved to
`tests/fixtures/golden_seed.jsonl` specifically so they cannot be mistaken for data. They
are `authored_by=seed_unverified` and none of their labels was ever checked against a
policy wording.

Until 30 human-labelled items land, the correct phrasing in any status report is
**"eval gate has no data"** — not "eval gate green".

---


## Citation grounding verifies membership, not support

**Measured 2026-10-02**, first run of the generator against a real model (Qwen2.5-1.5B on
CPU, 7 labelled items):

```
answered 7/7    abstained 0    answers citing a LABELLED span: 1 of 7
```

Every item cited passage `[1]`. `[1]` is always in range, so the verification layer passed
**all seven** — while six cited a passage that is not the labelled evidence.

### What the check actually asserts

`GeneratingEngine._verify` resolves each cited integer to a chunk the retriever returned,
and discards the whole answer if any citation falls outside that set. That is a real guard
and it catches a real failure: a model inventing `star-comprehensive-2025:p32:s146`, or
citing passage `[7]` when six were shown, is refused.

What it cannot see is whether the cited passage **supports the claim**. A model that cites
`[1]` for everything is grounded by the letter of the check and meaningless in fact. The
property being enforced is *membership in the retrieved set*, and the property a reader
assumes from the word "grounded" is *evidential support*. Those are not the same, and the
gap is the entire space in which a confident wrong answer lives.

### The sharpest case from that run

**h-23** — *"I am travelling to the United States next month. If I am hospitalised there,
will the policy pay the bill?"* — is an `unanswerable` item. The corpus does not address
overseas treatment; the Zone A–E definitions a retriever surfaces are Indian co-payment
tiers answering a different question.

The system answered **"Yes."** With a citation. No abstention.

That is the failure this project was built to prevent, reaching the output unimpeded, with
the verification layer reporting success.

### What would close it, and why neither is done

* **The LLM judge** (DESIGN §5.8) scores answer-support directly. Not built, and its
  Cohen's κ is unmeasured — until κ clears the 0.60 floor its verdicts are not evidence,
  so building it does not by itself close this.
* **A lexical support check** — does the cited passage share the answer's key terms? Cheap
  and deterministic, but it would have passed h-23 too: the Zone passages do contain
  geography words. A weak check here is worse than none, because it would make the word
  "verified" mean even less.

Recorded unfixed. The honest claim is that the system **checks its citations are real, not
that they are relevant.**

### Caveats on the numbers above

A 1.5B model is not the production generator and these are not quality figures — see
`arag/local/__init__.py`. But the defect is in the engine, not the model: a stronger model
would cite more relevantly on average and the check would still not be testing relevance.

Two abstention paths were also untested by that run: `abstain_threshold` is 0.0 so the
low-confidence route never fired, and no provider failure occurred.

## RESOLVED — `contradictory` named two incompatible requirements

**Fixed 2026-09-09 by splitting `Strata.SUPERSESSION` out as stratum 15**, before any
supersession item was labelled. Kept here rather than deleted, because the reasoning is the
record of how a taxonomy defect gets decided and the same shape will recur.

The decision turned on cost, not on taste: the code cost of a split is flat, the relabelling
cost grows linearly with items already filed under the ambiguous name, and the set stood at
7 items — so splitting first was strictly cheaper than filing four supersession items under
a name meaning two opposite things and splitting afterwards.

What shipped: `SUPERSESSION` is adversarial, `needs_agent=False` (version filtering is a
retrieval-side metadata filter, not conditional retrieval, so it must not justify the
agent's latency), implies a non-empty `must_not_cite`, and implies
`expected_behaviour=answer`. `CONTRADICTORY` keeps the >=2-document requirement and means
one thing again: cross-source disagreement where **both positions are current**. The
composition floor is 5 for supersession against 10 for the rate-gated strata — see
`SetTargets.min_per_stratum` for why a count-gated stratum takes a coverage floor instead of
a statistical one.

One thing the doc-cost estimate below got wrong, worth keeping visible: it predicted `"14
strata"` in three places. All three actually said **13** — the count was never updated when
stratum 14 was added, so that stratum *did* renumber the documentation silently, in exactly
the way this split was careful not to. Corrected in `schema.py`, `docs/EVAL_EXPLAINED.md`
and DESIGN §3, with a note in the `Strata` docstring to grep the count on any future
addition.

The original analysis follows, unedited.

`Strata.CONTRADICTORY` is currently doing two jobs whose required behaviours are opposites:

| | **cross-source disagreement** | **supersession** |
|---|---|---|
| DESIGN §3 description | row 12: *"What is the room rent limit?" across three insurers — must surface both, attributed* | not in the taxonomy at all; carried by `must_not_cite` |
| status of each position | **both current.** Two insurers, or two products, genuinely differ | **one is stale.** The 2021 wording was replaced in 2025 |
| correct behaviour | **surface both**, attributed to their sources. Suppressing either is the failure | **suppress the stale one.** Surfacing it is the failure |
| `expected_behaviour` | `surface_conflict` | `answer` |
| what a hard failure looks like | answering with one insurer's limit as though it were universal | citing `star-comprehensive-2021` at all |
| resolvable by machinery? | no — the disagreement is a fact about the market | yes — version metadata and `as_of_date` decide it |
| metric that catches it | `behaviour.conflict_surfaced` | `max_supersession_violations` via `must_not_cite` |

So an item filed as `contradictory` could require either "surface both" or "surface exactly
one", and the stratum name does not say which. Per-stratum recall and nDCG over a mixed
population of the two measure nothing in particular — which is the same objection raised
against adding an `intra_document_conflict` stratum, and it applies with more force here,
because this ambiguity already exists in the shipped enum rather than being a proposal.

**Why it has not bitten yet.** No supersession item is labelled. h-14 will be the first.
The measured supersession failure in the v1 baseline — h-01 retrieving
`star-comprehensive-2021 p3 def.hospital` at rank 1 — is invisible to the gate for a
different reason: h-01 carries no `must_not_cite`. See `docs/BASELINE_V1.md`.

### What splitting it would cost

Costed against the current tree, so the decision is not made blind.

**Code — small, ~1 hour.**

- `Strata`: add `SUPERSESSION = "supersession"`. One line, plus the `is_adversarial`
  property (it belongs there) and `needs_agent` (it does not — version filtering is
  retrieval-side, not conditional retrieval).
- `GoldenItem.validate`: the new stratum implies `must_not_cite` is non-empty, and should
  probably imply `expected_behaviour=answer`. Two `problems.append` branches, same shape
  as the existing `strata=injection implies expected_behaviour=ignore_injection` rule.
- `thresholds.yaml`: nothing. `max_supersession_violations` is already keyed on
  `must_not_cite`, not on the stratum.
- Report and sampling: nothing. Both iterate `Strata` dynamically.

**Tests — small, ~30 minutes.** Six or seven cases mirroring
`TestConflictRequiresTwoThingsInConflict`. Two existing tests enumerate strata coverage
and will need the new member added.

**Documentation — the real cost, ~1 hour.** DESIGN §3's table is quoted in the README plan,
in `docs/EVAL_EXPLAINED.md`, and in the golden-set template header. A stratum count of 14
appears in prose in at least three places, and §3 already carries a note that stratum 14
was added during labelling. Adding stratum 15 the same way is honest and cheap; silently
renumbering is not.

**The cost that is not hours — the golden set.** The composition gate in
`arag-eval validate` is written per stratum: `contradictory 0/10`. Splitting it means
deciding how the 10 divides, and every label already filed under the old name has to be
re-examined. At 7 items that is free. At 30 it is an afternoon. At 120 it is not worth
doing, and the taxonomy would be frozen by its own data.

**Recommendation on timing, not on the decision:** decide before the count passes ~30
labelled items. The code cost is flat; the relabelling cost is the only thing that grows,
and it grows linearly with items already filed under an ambiguous name.

---

## The `contradictory` floor is corpus-limited, not effort-limited

Lowered from 5 to 3, then to 2, on 2026-10-07 — first after a systematic survey, then
after reading the third candidate's actual text rather than its summary (a list heading on
a claim form, not an obligation clause). Reasoning in `docs/DESIGN.md`. Stated here because it is a limitation of what can be measured, not a
design choice.

IRDAI's 2024 standardisation mandates the wording for grace period, free look, moratorium,
cancellation, portability and claim settlement, so those clauses are verbatim-identical
across all four insurer documents. Only two genuine cross-source contradictions survive in this corpus, and both are places
where an insurer retained a pre-standardisation term. One of the two — the claim-document
conflict — has a contested premise of its own: its counterparty is the REGULATOR rather
than another insurer, and nothing in this system encodes that a regulator outranks an
insurer. See the authority-override note in `docs/DESIGN.md`.

**What this costs:** any cross-insurer contradiction rate computed from two items is a
development signal and nothing more — one item moves it by 50 percentage points. The
stratum demonstrates that the system can surface a conflict; it cannot support a claim
about how often it does.

**What would change it:** a fifth insurer, or a pre-2024 wording from a second insurer to
pair against `star-comprehensive-2021`. Neither is a labelling task.

## Corpus scope

### Hindi IRDAI versions exist and were not ingested

The IRDAI landing page for the 2024 Master Circular on Health Insurance Business publishes
**both English and Hindi** versions of the circular *and* its annexure. Only the English
ones are in `data/manifest/sources.jsonl`.

This is a **v1 scope decision, not an oversight.** The reasoning:

- Spike S5 measured zero Devanagari codepoints across all 197 ingested pages, which is
  what licenses the monolingual `bge-base-en-v1.5` embedding choice (DESIGN §5.3).
- Adding the Hindi documents would invalidate that measurement and force a multilingual
  embedding model, changing every latency number in DESIGN §9.
- A bilingual corpus also raises a question this project has not answered: when the English
  and Hindi texts of the same regulatory instrument differ, which governs? That is a real
  legal question, not a retrieval one, and answering it wrongly would be worse than not
  supporting Hindi.

The Hindi URLs are recorded in the S5 findings so re-adding them is a manifest edit, not a
research task.

### Care Health excluded

Care Health policy wordings sit behind product-page gating; only brochures are reachable.
Brochures assert cover that the policy wording does not grant, so ingesting one would let
the system answer a coverage question from marketing copy. `SourceKind.BROCHURE` is
rejected by the manifest validator to make this structural rather than a matter of care.

### One deliberately superseded document

`star-comprehensive-2021` is ingested on purpose so that retrieving stale terms is a
*measurable* failure rather than invisible noise. It is a known-bad document in the index.

---

## Known extraction defects, unfixed at time of writing

Measured in spike S5 (`docs/INGEST_FINDINGS.md`), fixes scheduled for v1 ingest:

| defect | scale | consequence if unfixed |
|---|---|---|
| Ligature contamination (U+FB00–06) | 269 codepoints; 254 in `star-comprehensive-2021` on all 18 pages | `beneﬁt != benefit`, so BM25 cannot match the most-queried word in the domain, and the embedder sees an out-of-vocabulary token. Silent — no error, and the text looks correct in a PDF viewer. |
| Interleaved prose columns | 4 pages, all in `star-comprehensive-2021` | Clauses from different columns are spliced. Reads fluently, is wrong. |
| Soft hyphens / NBSP | 2 / 58 | Split words break exact matching. |

## `def.<term>` extraction is typographic, so the key is Star-only in practice

Measured 2026-10-07 while preparing h-40, and it qualifies a principle rather than noting a
bug. Full account and the restated principle in `docs/DESIGN.md`.

The index extracts defined terms with a line-anchored `Term: ` pattern. Star typesets
definitions that way; **Niva Bupa numbers them** (`2.1.36. Room Rent means…`), so the
pattern never fires on a real Niva definition. What it does fire on is page furniture:

| document | `def.*` ids | what they mostly are |
|---|---|---|
| `star-comprehensive-2021` / `-2025` | 82 / 81 | real defined terms |
| `nivabupa-rise` / `-reassure2` | 27 / 30 | `def.product_name` on **34 of 34 pages**, `def.note`, `def.fax`, `def.email` |

Two consequences, and the second is the one that bites:

1. A labeller writing `nivabupa-rise#def.room_rent` gets a load error — annoying, loud,
   harmless. Writing `nivabupa-rise#def.product_name` gets an id that validates and
   resolves to every page in the document. Partly guarded: a span on an id appearing on
   more than five pages draws the AMBIGUOUS warning.
2. It is not clean inside Star either. `def.reasonable_and_customary_charges` resolves for
   `star-comprehensive-2021` and for **no other document**, though all four insurer
   wordings define the term — in the 2025 wording the term shares a line with a page
   header, and the pattern is line-anchored. The same term, two versions of one product,
   one key.

Not fixed. Fixing it means teaching the extractor Niva's numbered-definition convention,
which is a per-insurer rule in a component whose value is being insurer-agnostic. The
cheaper guard is the one now stated in DESIGN: **check that the id exists in both documents
before any cross-insurer item relies on it.** `excl.NN` needs no such check, because the
regulator mandates the literal string rather than the layout.

## Superscript-marker numeric corruption: guarded, not observed

A footnote marker fused onto a monetary value (`5,00,000/-¹` → `5,00,0001`) would be a
silent numeric corruption strictly worse than the ligature problem, because in this domain
the tables *are* the answers.

**Measured across all 197 pages: zero occurrences**, by five independent detectors —
Unicode superscript/subscript codepoints, NFKC numeric-token delta, `/-N` adjacency,
Indian-format-plus-extra-digit, and small-font digit spans adjacent to numerics. The two
detectors that fired at all (293 `/-N`, 89 long digit runs) were verified as false
positives: a currency value ending `/-` followed by the next table cell, and phone numbers.

The invariance guard is implemented anyway, because the hazard is real in principle and
would arrive silently with any new insurer document. It is insurance against a future
corpus, not a fix for an observed defect — an important distinction, since the guard
passing today proves nothing about the guard working.

---

## Table extraction: four named gaps

Tables are the load-bearing content in this corpus — 165 candidates on 131 of 197 pages —
so the gaps in table handling matter more than anything else on this page.

### Cross-version table comparison was done on flattened text, not on a column grid

The supersession survey for items h-34 and h-35 compared the Modern Treatments sub-limit
tables between `star-comprehensive-2021` p10 and `star-comprehensive-2025` p10 and reported
**"every figure matches, in every sum insured band"**. That claim is true of the *number
sequence* the two pages extract, and it was checked that way: the page text was flattened
to a single line and the rupee figures read off in document order.

What it does NOT establish is that each figure sits under the same column heading in both
versions. The two pages are laid out differently - 2021 prints twelve treatments as one
block, 2025 splits the same twelve across two tables of six - and at least one cell reads
`Up to Sum Insured` spanning several columns, so the number count per row is lower than the
heading count. A reader who takes "every figure matches" to mean "the cap on robotic
surgery is unchanged" is relying on a column alignment that was never reconstructed.

The conclusion drawn from it is weaker than it looks, and is stated here at its true
strength: **no rupee figure was added, removed or altered between the two versions, and the
sum insured bands are identical**. Whether any figure moved between columns is unknown.

Closing it means parsing both pages into a cell grid - `arag.ingest.tables` can do this -
and comparing cell by cell. Not done, because the survey's purpose was finding supersession
candidates and it found two without needing the grid; a labelled item that turns on a
specific cap would need the grid first.

### The T2 header guarantee is weaker than it sounds

**T2 guarantees that a header exists. It does not guarantee that the header is
informative.** Three stitched tables in `star-comprehensive-2025` extract a *caption*
where their column names should be, and a caption satisfies the non-empty check:

| document | pages | extracted "header" | rows |
|---|---|---|---|
| star-comprehensive-2025 | 14-15 | `Limits for Vaccination` | 6 |
| star-comprehensive-2025 | 15-16 | `Out-Patient Consultation` | 14 |
| star-comprehensive-2025 | 20-21 | `Table - B2` | 27 |

The consequence is concrete. A row-level sentence built from the third one reads
**"For Table - B2 27%: ..."** — which passes every check the ingest pipeline applies, is
embedded, is indexed, and is unretrievable by any question a user would actually ask. It
is the same class of silent failure as the ligature problem: no error, plausible-looking
output, useless in practice.

This is not a rounding error and it is not fixed. `star-comprehensive-2025` pages 14-15
(vaccination limits) is deliberately labelled as a **known-failing golden item**
(`expected_to_fail: true`), because a stratum containing only cases the system is expected
to pass measures nothing. The run in which that item flips to passing is the intended
artefact.

### Six of sixteen page-spanning "tables" are prose

`find_tables` detects table-shaped regions, and a densely-laid-out prose page is
table-shaped. Six stitched regions are paragraphs, not grids — their "header" cells run
313 to 2286 characters:

| document | pages | longest header cell |
|---|---|---|
| irdai-annexure-2024 | 2-3 | 765 chars |
| irdai-master-circular-2024 | 7-9 | 313 chars |
| nivabupa-reassure2 | 19-24 | 1921 chars |
| nivabupa-reassure2 | 26-28 | 2242 chars |
| nivabupa-rise | 17-22 | 1450 chars |
| nivabupa-rise | 23-25 | 2286 chars |

So the honest headline is **7 trustworthy data grids out of 16 detected stitches**, not
16. Reporting the raw stitch count would overstate the result by 2.3x. The split is
asserted in `tests/test_ingest_tables.py` with the current counts as expected values, so
it cannot drift silently in either direction.

These are not deleted, and that is deliberate: the prose inside a spurious stitch is real
content that the prose chunker still indexes. The classification exists so the number is
not overstated and so a labeller knows which stitched tables are worth targeting.

### Two tables are refused outright

`irdai-master-circular-2024` pages 10 and 11 produce no detectable header and are refused
rather than indexed with an invented one:

- **p10** — a table of contents. Three rows of clause titles (`18) Claims in respect of
  multiple Policies`, `19) Redressal of Grievances`, `20) Implementation of Ombudsman
  Award`), no data.
- **p11** — a numbered principles list (`A. General Principles`, `All Insurers shall
  ensure the following...`) laid out in columns.

Both are boilerplate. Neither is a benefit grid or a turnaround-time table, so no header
is hand-supplied. **Refusing them does not make those pages unanswerable**: only the
table-shaped extraction is refused, and the prose chunker covers the same regions — a
property with its own test, because without it the refusal would create a genuine blind
spot in the one document that governs the other five.

---

## The conditional-override detector finds one shape of gate and misses the rest

A conditional override is a figure that is real but gated on something the policyholder
does not have by default. Two confirmed instances in `star-comprehensive-2025`:

| # | clauses | the gate | consequence of missing it |
|---|---|---|---|
| 1 | `excl.01` p30 vs p31 | an optional buy-back rider bought at first purchase | 12 months reported to someone who has 36 |
| 2 | clause 9 p41 vs clause 12 p42 | annual renewal vs instalment payment | "not covered during the grace period" reported to an instalment payer who **is** covered |

**The detector found #1 and scored #2 as noise.**

The scan works by proximity: a conditional marker (`Optional Cover`, `on payment of
additional premium`, `where the Insured Person has opted`, `buy back`, `rider`, `add-on`)
within 400 characters of a duration or money amount. That shape is exactly what a rider
looks like, so instance #1 lights up with five co-occurring markers.

Instance #2 has no such shape. The gate is a **payment mode** — the difference between
paying annually and paying by instalment — and the clauses that carry it read:

- p41 cl.9: *"While coverage is not available during the Grace Period…"*
- p42 cl.12 viii: *"For premium paid in instalments during the Policy Period, coverage is
  available during the Grace Period also."*

Nothing there is marker-shaped. The only phrase the scanner could latch onto is
`opted for`, which it correctly rates as weak because `opted for` appears in ordinary
definitions ("the Sum Insured **opted for** and for which the premium is paid"). So the
sharpest conditional override in the document was filed under "weak / likely false
positive" alongside seven genuine false positives.

**The ratio, plainly: 2 confirmed instances. Reading found 2. Scanning found 1.**

The classes of gate the detector structurally cannot find:

- **payment modes** — annual vs instalment vs monthly
- **eligibility states** — entry age, ported policy, continuous-coverage duration
- **benefit contexts** — a limit that applies only within one section's cover

All three are expressed in ordinary policy prose with no lexical marker. The detector is
kept because it is cheap and it did find a real instance, but its count is a **floor, not
an estimate**, and it must not be quoted as corpus-wide incidence.

### The enhancement rule is NOT a third instance

Worth recording because it was nearly counted as one. The rule that waiting periods
restart on an enhanced sum insured appears in two places:

- as a **sub-clause of each exclusion**: `excl.01` **B**, `excl.02` **B**, `excl.03` **C** —
  *"In case of enhancement of Sum Insured the exclusion shall apply afresh to the extent of
  Sum Insured increase"*
- restated in **clause 26, "Revision of Sum Insured"** (p44), which cross-references all
  three codes

Those two statements **agree**. There is no gate producing a different answer, so it is a
restatement, not a conditional override, and calling it one would have inflated the tally
by 50%.

It is still a risk, but a different one at a different layer: the caveat lives in
sub-clause B of a clause whose answer is in sub-clause A. **If chunking splits a clause
from its sub-clauses, the caveat is lost.** That is a chunking requirement — keep a clause
and its lettered sub-clauses in one chunk — not a stratum, and not a retrieval failure.

---

## Architectural limitations accepted

### The PR gate does not exercise retrieval against a live database

Cassettes replay provider calls (LLM, embedding, judge) but not Postgres. Until a committed
fixture corpus exists, the PR gate covers metric code, schema validation, behavioural
checks and the engine contract; the nightly live run covers the rest. A cassette that
replayed finished answers instead of provider calls would make the gate incapable of
detecting any engine regression at all — see `src/arag/eval/cassettes.py` for why that
design was rejected.

### `clause_id` cannot join across document versions

Measured in S5: bariatric surgery is `Section 6 m.` in `star-comprehensive-2021` and item
`15` under `Section II` in `star-comprehensive-2025`, with no arithmetic relationship
between the schemes. `concept_id` (a controlled vocabulary, human-assigned at labelling
time) is the cross-document join key. The cost is that supersession and cross-insurer items
require a human to assign a concept, and a concept missing from the vocabulary is a load
error rather than something inferred.

### The local judge is unvalidated

DESIGN §5.8 accepts a free local judge on the condition that Cohen's κ against human labels
is measured and published. That measurement **has not been made** — it cannot be, until
labelled data exists. Any judge-derived metric (faithfulness, answer correctness) is
untrustworthy until `arag-eval judge-agreement` reports κ ≥ 0.60.

### Gemini free tier and outbound data

Spike S1 is unresolved: the Gemini API free tier is believed to use submitted data for
product improvement. Until verified, outbound PII redaction is treated as a functional
requirement rather than a demo, and no real personal data should be sent through the
deployed service.
