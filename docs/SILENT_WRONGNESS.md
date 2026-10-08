# The silent-wrongness pattern

*Destined for the README. Kept as its own document while the project is in flight so it
can be extended each time another instance turns up.*

---

Eleven bugs found in this project so far share a single shape, and it is not the shape
most testing is built to catch:

> **The code runs. Nothing errors. The output looks right. It is wrong.**

A crash tells you where to look. A wrong number that renders beautifully in a terminal,
passes every type check, satisfies every schema, and produces no log line will be believed
— and in a system that answers insurance-coverage questions, believed and wrong is the
worst available outcome.

Three further patterns are named at the end of this document, and the fourth has no code
in it at all: a *finding* that existed only in the prose that reported it. It is listed with
the others because the consequence is identical — a label would have validated, scored and
reported a conflict that does not exist — and because it is the only one the harness cannot
be taught to catch.

Instances 1-5 were found by **checking output against the artefact it described**; none
would have been caught by more unit tests of the code as written. Instances 6 and 7 are
a second variety: not a wrong number but a wrong *module graph*, where the artefact to
check against is what the interpreter actually loaded. Instance 7 is the first whose
symptom would have been a production failure rather than a plausible-looking answer; instance 8 is the first that would have corrupted the golden set
by telling a labeller their correct span was wrong.

---

## Instance 1 — ligature contamination

**What it looked like:** BM25 retrieval quality was mediocre on `star-comprehensive-2021`.
Nothing failed. The extracted text rendered perfectly in a PDF viewer and in a terminal.

**What was actually true:** the document contains **254 ligature codepoints** across all 18
of its pages. `benefit` typeset with an fi-ligature extracts as a string containing
**U+FB01** — one character that is *not* the letters `f` and `i`:

```python
"beneﬁt" != "benefit"      # BM25 cannot match a user's query
"fi" not in "beneﬁt"        # substring search fails too
```

The word `benefit` — the single most-queried term in the domain — appeared to occur
**once** in an 18-page policy wording. The other 50 occurrences were hidden behind a
codepoint that looks identical.

**Detected by:** a debug `print()` crashing on a cp1252 console. The crash was incidental;
the character it choked on was the finding.

**Fixed by:** Unicode NFKC normalisation at ingest *and* at query time — because a user who
copy-pastes a phrase out of the PDF submits U+FB01 too, so index-side-only normalisation
leaves the query broken in exactly the case where the user is quoting the document.

**Measured effect:** `benefit` 1 → 51 occurrences. `office` 0 → 24.

---

## Instance 2 — concept anchors that did not match the document's wording

**What it looked like:** a *correct* golden-set label was rejected. The span pointed at
`excl.03` on page 32, which is exactly where the 30-day waiting period lives, and
validation reported it as almost certainly on the wrong page.

**What was actually true:** the concept `waiting_period.initial` listed the anchor phrase
`"30 days waiting"`. The document says **"30-day waiting period"**. A missing anchor and a
wrong page are indistinguishable to the checker, so the checker blamed the label.

Worse in kind than instance 1: this bug *rejected correct work*. Left in place it would
have trained the labeller to move good spans to bad pages to satisfy the tool.

**Detected by:** writing a worked example against the real corpus instead of a synthetic
fixture. A synthetic fixture would have used my phrasing and passed.

**Fixed by:** broadening the anchors, and — more importantly — making every concept
rejection **print the anchors it tried**, so the next occurrence is diagnosable without
opening the module:

```
ERROR - concept 'waiting_period.initial' does not appear on page 14 ...
  Anchors tried: ['initial waiting period', '30-day waiting', '30 day waiting', ...].
  If the clause genuinely discusses this concept in different words, extend
  CONCEPT_ANCHORS rather than moving the span.
```

---

## Instance 3 — case-sensitive prefix strip in `def.<term>` parsing

**What it looked like:** `validate_clause_id("Definition Hospital")` returned
`def.definition_hospital` and was **accepted**.

**What was actually true:** the prefix-stripping regex was case-sensitive, so `Definition `
was not recognised as a prefix and got folded into the term slug. The result is a
well-formed clause id that matches nothing in the corpus. It would have loaded, counted in
the denominator, and scored zero forever.

The identical input `def.hospital` worked. So the bug only fired on one of several
equivalent spellings — the kind of defect that survives a test suite written by the same
person who wrote the parser.

**Detected by:** printing the output for every spelling a labeller might plausibly type,
rather than only the documented one.

**Fixed by:** `re.IGNORECASE`, and a test asserting all four spellings collapse to one id.

---

## Instance 4 — my own diagnostics, wrong four times

Not a bug in the system: a bug in the instruments used to measure it. Worth its own entry
because it is the same shape one level up, and because the measurements were briefly
reported as findings before being checked.

| heuristic | claimed | actual | what caught it |
|---|---|---|---|
| space ratio ⇒ fused text | a discount-rate table was corrupt | tables extract one short cell per line; text was clean | dumping the page |
| two-column layout ⇒ broken reading order | 32 pages damaged | the content stream emits one full column then the other; **0** damaged | printing block x-ranges and emission order |
| column transitions ⇒ prose splicing | 22 pages spliced | 18 of them were tables, whose cells alternate by construction; **4** real | checking whether flagged pages contained tables |
| header cell > 45 chars ⇒ prose | 4 genuine stitched grids | verbose column names are legitimate; **7** genuine | reading the flagged headers |

The uncorrected probe over-reported broken pages by roughly **15x** (61 → 4).

And the conditional-override detector, built after all of the above, **still** missed the
sharpest instance in the corpus: the grace-period contradiction, whose gate is a payment
mode with no marker-shaped text to latch onto. It scored the document's most dangerous
clause pair as "weak / likely false positive". Reading found 2 of 2 instances; scanning
found 1.

---

## Instance 5 — C0 control characters surviving into indexed text

**Identified, not yet fixed, and the counts are not yet measured.** Recorded here so the
numbering is not invented later: `normalise_text` strips the characters a PDF extractor is
expected to emit, but C0 controls other than tab/newline — BEL (`\x07`) among them — pass
through into chunk text and therefore into the BM25 term stream. A control character does
not error, does not display, and does not obviously change a token count, so the only
symptom is a term boundary in the wrong place.

What is still missing is the number: how many chunks contain one, and whether any of them
sit inside a labelled span. Until that is measured this entry is a suspicion with a
mechanism, not a finding, and it is deliberately not counted in any claim about corpus
quality. Fixing it before measuring would also destroy the evidence — the before/after
token counts are the whole proof that the fix changed anything.

---

## Instance 6 — a circular import whose `ImportError` was swallowed

`arag.index.build` imported its collaborators inside `try/except ImportError`, recording
failures in `_MISSING_MODULES`. That was correct while three modules were being written in
parallel by different agents: the module stayed importable before its collaborators
existed. It also built a trap.

`arag/index/__init__.py` imported `arag.index.build` at module scope, and `build` imported
`arag.retrieval.lexical` at module scope. Any import that re-entered `arag.index` on that
chain would get the partially-initialised module back, raise `ImportError`, and have it
**caught and discarded**. The symptom is not a crash. It is `build_corpus` reporting that
the lexical retriever is unavailable — pointing at the wrong module entirely — and doing so
only for some import orders.

Import-order dependence is what makes this worse than a plain bug: it can pass every test
and fail in the deployed service, or the reverse, with no code change in between.

Guarded by `tests/test_import_cycle.py`, which runs five import orders in **separate
interpreters** (`sys.modules` caching means one process can only ever test one order) and
asserts `_MISSING_MODULES == []`. It also asserts the attribute still exists, so it cannot
pass vacuously if the guards are renamed away. The eager edge itself is gone as of instance
7's fix; the test stays, because the `try/except` remains and the cycle can be rebuilt from
the other side.

---

## Instance 7 — an import closure that dragged PyMuPDF into the request path

The first instance whose symptom is a **production failure rather than a wrong number**,
and the only one no test caught.

```
arag.retrieval.lexical              # online package
  imports arag.ingest.chunk_types   # a file containing two dataclasses
    -> executes arag/ingest/__init__.py    # importing a submodule runs its package first
      -> imports arag.ingest.probe
        -> imports pymupdf                 # ~40MB native, offline-only
```

Every module on that chain is a reasonable dependency, and every one of them passes a
per-file import check. `tests/test_import_boundaries.py` walked *direct* imports, so it was
green throughout. The thing that was broken was the import **closure**, which nothing
looked at.

Measured, not assumed — importing each module in a fresh interpreter and inspecting
`sys.modules`:

| module | before | after |
|---|---|---|
| `arag.retrieval.lexical` | `pymupdf` loaded | clean |
| `arag.api.app` | `pymupdf` loaded | clean |
| `arag.index.lexical` | `pymupdf` loaded | clean |
| `arag.retrieval.stub` | clean | clean |
| `arag.agent.retrieval_only` | clean | clean |

Consequence, had it shipped: DESIGN §9 splits offline compute (PyMuPDF, a GPU) from online
(CPU, serverless) specifically to keep a native PDF library out of a function that never
opens a PDF. The deployed bundle would have carried it anyway — a size-limit failure or
seconds of cold start, discovered in a deploy.

**Fix.** Lazy re-exports (PEP 562 `__getattr__`) in `arag/ingest/__init__.py` and
`arag/index/__init__.py`, so importing a submodule no longer pays for its siblings. The
flat convenience surface is preserved; a `TYPE_CHECKING` block keeps the re-exports typed,
because mypy does not follow `__getattr__` and without it every one of them would silently
become `Any`.

**Guard.** `tests/test_import_closure.py`, two overlapping tests:

- an AST closure walk that models the implicit parent-package edge — importing `arag.a.b`
  executes `arag.a` — which is the exact edge whose absence made the old test useless. It
  is fast and names the offending chain.
- a subprocess test that imports each request-path module in a fresh interpreter and asserts
  no offline root reached `sys.modules`. This is ground truth: it cannot be fooled by an AST
  blind spot, an `importlib` call, or a `__getattr__` chain.

The first says where to look, the second says whether you are wrong.

Both were **mutation-tested**: reintroducing a single eager `from arag.ingest.probe import
probe_page` into `arag/ingest/__init__.py` fails 4 of the 8 tests, including the AST walk
and three of the interpreter probes. A guard that has never been observed to fail is not
known to be a guard. There is also a test asserting the graph still records the
parent-package edge, so the closure test cannot go green by modelling nothing.

---

## Instance 8 — an identifier pattern applied to the wrong unit of text

The clause index is the only artefact standing between a labeller and a wrong span, and it
is trusted absolutely: `arag-eval validate` reports a clause it cannot find as a problem
with the **label**. So an incomplete index does not degrade — it actively misdirects.

`build_clause_index` split each page into lines and ran every identifier pattern over each
line. The regex for the IRDAI codes, `\bCode\s+Excl\s*(\d{2})\b`, is correct, and `\s`
spans a newline perfectly well — but only when the newline is inside the string being
matched, and after `splitlines()` it never is. PyMuPDF emits a newline wherever the PDF
wraps, and `star-comprehensive-2025`'s two-column exclusions layout wraps **inside the
token**:

```
p33: 'Code Excl \n06'      p33: 'Code \nExcl 07'
p33: 'Code \nExcl 08'      p34: 'Code Excl \n16'      p44: 'Code \nExcl 03'
```

`star-comprehensive-2021`'s exclusions are single-column and wrap nowhere, so it lost
nothing. **That is what made the bug invisible**: the gap presented as a difference between
the two wordings — plausible, since insurers do renumber between versions — rather than as
a defect in the reader. It was found only because a labelled span pointed at `excl.08`,
validate rejected it, and the rejection happened to be checked against the PDF.

### The two halves, and the second is worse

The four fully-absent codes (`excl.06`, `07`, `08`, `16`) are the loud half. A label naming
one of them blocks: validate refuses it, nothing enters the golden set, and the labeller is
stopped — misdirected as to the cause, but stopped.

**`excl.03` losing page 44 is the harder half, and it is a different failure.** The code
*was* in the index, from page 32, so:

* `has_clause()` returns True. Nothing blocks.
* A label written as `excl.03` **p32** resolves, passes, and is correct.
* A label written as `excl.03` **p44** — equally correct, a genuine second occurrence —
  gets `clause 'excl.03' exists but NOT on page 44 (found on pages [32])`, which is a
  **hard ERROR asserting a true label is false**, in a message with no hedge in it.

An absent code produces a labeller who is stuck. An incomplete page list produces a
labeller who moves a correct span to a wrong page because the tool told them to, and every
metric computed afterwards is quietly measured against the wrong location. The first wastes
an hour. The second corrupts the golden set, and it does it through the exact mechanism the
resolver exists to prevent.

Generalised: **a partially-correct index is more dangerous than an absent one**, because
partial correctness is what earns the trust that the wrong part then spends.

### Blast radius, counted properly

Counting distinct codes understates it. Counted as `(code, page)` pairs:

| document | pairs in text | pairs indexed | lost |
|---|---|---|---|
| star-comprehensive-2025 | 40 | 35 | **5** |
| star-comprehensive-2021 | 43 | 43 | 0 |

And the same per-line application was silently costing `star-comprehensive-2021` **11
compound ids** (`II.2.i` … `II.12.s`) — which are precisely the cross-version join keys for
its summary table. Nobody was looking for those; they turned up in the diff after the fix.
The lesson is not "check the codes", it is that a defect in a shared mechanism does not
confine itself to the symptom that exposed it.

### Fix and guard

Patterns are now split by the unit of text they are valid against — `LINE_ANCHORED`
(`section`, `numbered`, `definition`) matched per line, `PAGE_SEARCHED` (`compound`,
`excl_code`) matched over the whole page. Rebuild added 4 codes and 1 page to star-2025 and
11 compound ids to star-2021, and removed nothing anywhere.

`tests/test_clause_index_completeness.py` asserts the property directly: for every
document, every `Code Excl NN` in the extracted text appears in `clause_index.json` **on
that page**. Two design choices in it are load-bearing:

* It **does not import the builder's regex.** A test that reuses the pattern under test
  cannot detect that the pattern is applied to the wrong unit of text — it would have been
  green throughout. It carries its own, and a second test asserts that its own regex still
  spans a newline, so the guard cannot quietly stop guarding.
* It asserts a **property against the source PDFs**, not a stored count. "star-2025 has 35
  codes" would have passed happily with the bug present, because the wrong number would
  have been the expected one.

Mutation-tested: deleting `excl.08` from the index fails it with `[('excl.08', 33)]`.

### The message that caused it, changed

A clause id in one of the two **portable** forms — `excl.NN` or `def.<term>` — is
regulator-imposed rather than insurer-chosen, so a labeller did not invent it. When one is
absent from the index there are two candidate explanations and the harness cannot tell them
apart. It now says so, as UNVERIFIABLE rather than ERROR:

> code `'excl.23'` is not in clause_index.json for star-comprehensive-2025 — the index may
> be incomplete; verify against the PDF before moving your span.

Insurer numbering (`4.2.1`, `99.9`) still returns ERROR, because for those forms absence
really is evidence about the label.

---

## Instance 9 — the chunker and the labels used different names for the same clause

The worst instance so far, measured by consequence: **every metric this project reported
before 2026-09-16 was computed against a matcher that could not see three of seven items'
spans**, and the resulting failures were attributed to the retriever.

### What happened

DESIGN establishes that the corpus has exactly two portable join keys, `excl.NN` and
`def.<term>`, and that a labeller should prefer them because they survive a version bump
while insurer numbering does not. That principle was applied to the golden set, and to
`clause_index.json`, and **never to the chunker**.

`star-comprehensive-2025` p32 opens a block with the list number `3.` and the same block
carries `Code Excl 03`. The chunker recorded only the heading id:

```
chunk span  : star-comprehensive-2025 p32 clause='3'
h-03 label  : star-comprehensive-2025 p32 clause='excl.03'
SpanMatcher : False
```

Measured across the whole corpus before the fix:

| | |
|---|---|
| `excl.NN` ids in `clause_index.json` | 83 |
| indexed chunks carrying **any** `excl.NN` id | **0 of 1,024** |
| golden items keyed on `excl.NN` | 3 of 7 (h-02, h-03, h-28) |

### Why it survived every existing check

Nothing was wrong with any component on its own, which is why every test passed:

* The chunker's ids were correct — `3` *is* the heading.
* The clause index's ids were correct — `excl.03` *is* in the text.
* `SpanMatcher` compared them correctly and correctly returned False.
* `arag-eval validate` resolves labels against `clause_index.json`, where `excl.03` exists
  on p32. It reported **0 errors**. The label was valid; it was merely unmatchable.

The defect lived in the **agreement between two components**, and no test asserted an
agreement. Each side was tested against its own intent.

### The consequence, stated plainly

Every retrieval number published before the fix was measured against a matcher blind to
three of seven items. Corrected on identical corpus, config and labels — the only change is
that chunks now carry the ids their text contains:

| | recall@10 | nDCG@10 | MRR | ctx precision | hit rate |
|---|---|---|---|---|---|
| BM25 before | 0.300 | 0.188 | 0.167 | 0.040 | 0.400 |
| BM25 after | **0.400** | **0.235** | **0.207** | **0.060** | **0.600** |
| dense before | 0.400 | 0.309 | 0.307 | 0.060 | 0.600 |
| dense after | **0.600** | **0.420** | **0.327** | **0.120** | **0.800** |

The sharpest single case: **h-02's correct chunk was retrieved at rank #10 by the dense
retriever while the item was reported as recall 0.000.** Retrieval worked. The scoring could
not see it. On that evidence the v2 write-up concluded "the remaining failures are not
lexical-versus-semantic" — a conclusion about the retriever drawn from a defect in the
matcher.

`docs/BASELINE_V1.md` is invalidated by this and says so; the `v1_baseline` threshold
profile was derived from the understated numbers.

### Fix

The chunker attaches **every** citable identifier its text carries to `ChunkMeta.clause_ids`,
using `clause_index.ids_on_page` — the same function that builds the index, deliberately not
a second copy, because two implementations of "what identifiers does this text contain?"
would drift back into exactly this bug. `SpanMatcher.matches_chunk` tries the primary span
and every alternate. `span.clause_id` is unchanged, so citations still name what the chunk
*is* rather than every name it answers to.

### Guard

`tests/test_chunk_clause_agreement.py` asserts the agreement that had no test:

* every `excl.NN` in `clause_index.json` is carried by a chunk on its page — **0 tolerated**,
  because this is the family the golden set prefers and the one instance 9 erased;
* every span in the golden set matches at least one chunk in the corpus — the end of the
  chain, catching an unmatchable label the moment it is written rather than after it has
  contributed a zero to a published number;
* the `def.<term>` and bare-numeric divergences are **ratcheted at their measured size**
  (351 and 271 of 1,238 pairs) rather than asserted to zero. Those two are the index being
  deliberately more permissive than the chunker — the index applies the definition pattern
  on every page while the chunker gates definitions to the definitions section, and the
  index records every numbered line while R2 forbids splitting a clause from its limbs. They
  are tolerated at that size and cannot silently widen.

### The consequence that matters most: it disarmed the guard

Understating the numbers was not the worst of it. **The regression gate was derived from
those numbers**, so the defect propagated into the one mechanism built to catch a
regression in exactly them.

`v1_baseline` set `min_recall_at_k: 0.30`, taken from the measured 0.300. The true figure
was 0.400. The arithmetic of what that permitted is exact, not illustrative:

> Corrected per-item recall under BM25: h-01 `1.0`, h-02 `0.0`, h-03 `0.0`, h-04 `0.5`,
> h-28 `0.5` → mean **0.400**.
>
> Lose h-28 entirely to a regression: `(1.0 + 0 + 0 + 0.5 + 0) / 5` = **0.300**.
>
> `0.300 >= 0.30` → **GATE PASS.**

A change that destroyed one item's retrieval completely would have been reported green. And
nothing about the gate looked wrong from outside — it was passing throughout, which is the
only state anyone ever observed it in.

This is why the chunker gap mattered beyond its arithmetic. A mismeasurement that stays in
a report is embarrassing. A mismeasurement that becomes a *threshold* is load-bearing: every
later change is checked against it, and the check is weaker than anyone reading it believes.
The guard was not merely wrong about the past; it was **disarmed for the future**.

Re-derived as `v1r_baseline` from `data/baseline_v1r.json` — `min_recall_at_k: 0.40` at the
measured floor, no tolerance. The same regression now fails. The superseded profile is kept
in `thresholds.yaml`, marked unusable, rather than deleted. See `docs/BASELINE_V1R.md`.

### What it changes about how this project is built

`arag-eval validate` checked labels against the index and pronounced them sound. That check
was necessary and **insufficient**, and its confidence was the problem: it told a labeller
their span was fine when nothing in the retrieval path could ever match it. A label is not
sound because it is well-formed, nor because the index agrees it exists. It is sound when
**something a retriever can return actually matches it** — which is now what the test
asserts.

---

## Instance 10 — a refusal recorded as an answer

Found by the first run against a real model, not by any of the 60 tests covering this
engine. The direction of the error is what makes it the worst one in the list so far.

The prompt tells the model: *"If you cannot answer from the passages, set `answer` to
null."* Models comply by writing the **string** `"null"` — the word, in quotes — and the
engine checked emptiness with:

```python
if not answer_text or not str(answer_text).strip():
```

A four-character string is truthy. So h-24 (*"How much premium would a 70-year-old pay?"* —
structurally unanswerable, premium tables are not in policy wordings at all) was
**correctly declined by the model** and recorded by the engine as an answer of `"null"`.

### Why this is worse than the nine before it

Every earlier instance produced a wrong *number*. This one produces a **false confident
response** — the system reporting that it answered when it refused — and it was introduced
by the layer whose entire purpose is to prevent exactly that. The verify-or-abstain design
exists so a fluent wrong answer never reaches someone asking whether their surgery is
covered; the bug turned the abstention path off for any model that spells null with quotes.

### Why no test caught it

`tests/test_generate.py` already had a case for this: `test_a_model_declining_to_answer_is_
low_confidence_not_malformed`, passing `reply(None, [])` — JSON `null`, which the prompt
asks for and which the check handles correctly. The test asserted the behaviour the prompt
*specifies*. The defect lives in the gap between what a prompt specifies and what a model
actually emits, and only a real model walks into that gap. A scripted provider returns
exactly what the test author imagined.

### Fix, and the trap inside the fix

`_is_declined()` matches a set of refusal spellings — `null`, `none`, `n/a`, `nil`, `-` —
case-insensitively, with a trailing full stop tolerated.

It matches the **whole stripped field, never a substring**, and that restraint is
load-bearing. `"No"` is a complete and correct answer to a coverage question, and *"None of
the listed exclusions apply to this claim"* opens with a declined-looking word. A substring
match would have converted a fix for false-answers into a cause of false-refusals — trading
one direction of the same failure for the other. Seventeen tests: twelve declined forms
abstain, five genuine answers survive.

---

## Instance 11 — a command that wrote first and validated second

`arag-eval import-xlsx` reads the labelling spreadsheet into `data/golden/v1.jsonl`. Its
docstring has always said:

> Exit 2 on any bad row - nothing is written when one row is unusable, because a partial
> import leaves the golden set in a state nobody chose.

The implementation wrote the JSONL, *then* called `validate()` on it. So an import of five
merged rows wrote 10 items to disk, then printed three of them as UNUSABLE and exited 2.
The golden set was left holding exactly the items the validator had just rejected.

### Why this one is not a near-miss

Everything downstream of the golden set reads the file, not the exit code. `arag-eval run`
loads `v1.jsonl` and scores against whatever spans are in it. A span on the wrong page does
not error - it scores zero forever, in the denominator, for every future run. The window
between a failed import and someone noticing is a window in which every metric is wrong in
a direction nobody can see, and the only thing holding it shut was a human reading an exit
code in a terminal they had already scrolled past.

Found by the user, not by me, and not by any test: *"You described this as all-or-nothing.
The behaviour doesn't match the claim."* The claim and the code had sat next to each other
in the same function since the command was written.

### Fix

Write to `v1.jsonl.staged`, validate **that**, and `Path.replace()` it onto the target only
on success - an atomic rename on both POSIX and Windows. The staged file is removed in a
`finally`, so a crash mid-validation leaves neither a corrupt golden set nor a second
unvalidated copy of it beside the first.

### The test, and what it had to be able to fail

`tests/test_import_atomicity.py`. The bad row is bad in a way **only corpus resolution can
see** - well-formed schema, clause on the wrong page - because a schema-level failure was
already refused before any write happened, so a test built on one would have passed against
the broken ordering. Mutation-checked: restoring `staged = target` makes two of the six
tests fail, including the positive control that proves the command still writes.

The general lesson is the one this document keeps arriving at from different directions: a
docstring describing a guarantee is not the guarantee. This one was a *correct* description
of intended behaviour, written by the same person, in the same file, four lines above the
code that contradicted it.

---

## A SECOND pattern — the signal produced and never consumed

Instances 1-9 share one shape: code that was **wrong**. This is a different shape and it
deserves its own name, because the tests that catch the first kind cannot catch this one.

> **The code is right. Nothing reads it.**

A field is added to a failure taxonomy, populated correctly at the producing site, covered
by a test asserting it is populated correctly — and no branch anywhere acts on it. Every
test passes. The producer is provably correct. The signal does nothing.

It is not dead code, which is at least visibly unused. It is worse: the field's presence is
read by a human as evidence that the case is handled, so it actively *suppresses* the
question "what happens when this is true?"

### Three in one session, all mine, all in the generation layer

| signal | produced | consumed | measured cost of the gap |
|---|---|---|---|
| `LLMError.retryable` | from the first commit | nothing | no retries at all; a 503 blip that the next call would have survived became an abstention |
| `LLMError.retry_after_s` | added with the 429 handler | nothing, briefly | the server's own `retryDelay` was parsed and discarded in favour of a client guess an order of magnitude too short |
| `LLMResponse.truncated` | from the first commit, set correctly | nothing | **the expensive one — see below** |

### The truncation case, with the arithmetic

`gemini-3.6-flash` is a thinking model: internal reasoning is billed against
`maxOutputTokens`. A 1024-token budget left roughly 39 tokens of visible output, so the
JSON reply was cut mid-string:

```
{"answer": "Pre-existing Diseases (PED) are excluded for 36 months of continuous
```

`finish_reason=MAX_TOKENS`, `truncated=True`, correctly set, and the engine never looked.
So it classed the failure as generic `MALFORMED` and took the repair path — **reprompting
with the same budget**, which cannot succeed by construction. The result:

* 5 of 7 questions in the first live run failed, all from this one cause;
* each burned **two** requests instead of one, against a quota of **20 per day**;
* and the failure presented as "the model cannot produce JSON", which is a prompt problem,
  pointing every diagnosis at the wrong layer.

The information needed to skip the pointless retry and print the actual fix was already in
the response object. It had been there the whole time.

### Why the existing tests could not catch this

`tests/test_generate.py` asserted the taxonomy thoroughly — every `LLMErrorKind` mapping to
its own `AbstainReason`, `retryable` set correctly per kind. Those tests check the
**producer**. A signal nothing consumes has a perfect producer. Coverage was no help either:
the line that sets `truncated=True` was covered, by a test asserting it gets set.

This is the difference between "is the value right?" and "does the value change what
happens?" Only the second is a behaviour.

### The audit, run once the pattern was named

`tools/audit_signals.py` (committed, so it can be re-run) walks the AST for attribute
*loads* outside the defining module — not grep, because a field's own assignment matches
its name. Every field of `LLMResponse`, `LLMError`, `LLMRequest`, `AnswerResult`,
`RetrievedChunk` and `ChunkMeta`, plus every member of the three enums.

**12 signals have no production consumer, and they split into two groups that carry very
different risk.** Someone reading the type declarations cannot tell them apart, which is
exactly why the split has to be written down.

#### FALSE ASSURANCE — reads as an implemented defence, is not one

| signal | what its presence implies | what is true |
|---|---|---|
| `AbstainReason.ONLY_SUPERSEDED_EVIDENCE` | the system refuses to answer from a replaced wording | **nothing produces it** |
| `AbstainReason.CONTRADICTORY_UNRESOLVABLE` | irreconcilable clauses are detected and refused | nothing produces it |
| `ChunkMeta.injection_suspected` | ingested text is screened for injection | never set, never read — the defence is a field name |
| `DegradedComponent.DENSE_RETRIEVAL` | a response can report which retrieval leg was missing | never set |
| `DegradedComponent.LEXICAL_RETRIEVAL` | " | never set |
| `DegradedComponent.RERANKER` | " | never set |
| `DegradedComponent.CHECKPOINTER` | " | never set |

Only `DegradedComponent.GENERATION` is ever produced, so `degraded` — a list precisely so
it can name *which* leg failed — currently cannot distinguish a missing retriever from a
missing generator.

#### PLANNED PLACEHOLDER — slice not yet built, recorded as such

| signal | waiting on |
|---|---|
| `RetrievedChunk.fused_score` | set by the fuser today; no consumer until the reranker lands |
| `RetrievedChunk.rerank_score` | v2 reranker, not built |
| `AnswerResult.tool_calls` | v3 agent, not built |
| `AnswerResult.cold_start` | a cold path existing to report |
| `LLMResponse.raw` | nothing — deliberate, a debugging payload never branched on |

**A placeholder someone has written down is a plan. An unread field nobody has noticed is
a false assurance.** That distinction is the whole difference between the two tables, and
it only exists once the audit has been run.

### The supersession defence has three independent holes

Worth stating on its own, because this is the failure mode the project is largely about
and the type declarations make it look addressed:

1. **No version filter is wired.** `RetrievalFilters.as_of` exists and both retrievers
   honour it through `arag.retrieval.filtering` — and no caller sets it. Every measurement
   so far ran unfiltered.
2. **No label guards against the superseded wording.** `max_supersession_violations: 0`
   passes because nothing can violate it. h-23 and h-24 do carry `must_not_cite`, seven
   entries each, but those are *abstention decoys* — day-care and OPD clauses — and name
   no superseded document. h-01 retrieves `star-comprehensive-2021 p3 def.hospital` at
   rank 1 under BM25 and the gate is silent.
3. **Nothing produces `ONLY_SUPERSEDED_EVIDENCE`.** The abstention exists as an enum
   member and has no code path.

Dense retrieval does rank the current 2025 definition first for h-01 where BM25 ranked the
superseded one first. That is **incidental** — an artefact of what the embedder happened to
prefer, reversible by any re-embed, and it sits on top of all three holes rather than
closing any of them.

The true claim is: **we can see supersession failing and have not fixed it.** Anything
stronger is not supported by anything in the repository.

### Ratcheted, so the list cannot rot

`tests/test_signal_consumption.py` asserts the unread set is **exactly** these 12. A new
unread signal fails the build; a signal that becomes consumed also fails, until someone
deletes it from the allowed set. Both directions are deliberate — an inventory that
silently goes stale is one nobody trusts. It also carries a guard-the-guard test asserting
that known-consumed signals are absent from the unread set, so the ratchet cannot pass by
detecting nothing, and a test pinning hole 3 above.

---

## A THIRD pattern — the check structurally unable to catch what it exists for

Three times now, and that is enough to name it. Distinct from both earlier patterns:
instances 1-11 are code that is wrong, the second pattern is code that is right and unread,
and this is a **guard that is working perfectly on the wrong question.** It reports success,
it is green, it is tested — and the thing it was built to catch walks past it.

It is the most dangerous of the three, because a green guard is read as evidence.

### The three occurrences

| # | the guard | what it checked | what it was for | how long it was green |
|---|---|---|---|---|
| 1 | `test_import_boundaries.py` | **direct** imports, per file | the online bundle not reaching PyMuPDF | the whole life of the project, while the boundary was broken (instance 7) |
| 2 | `v1_baseline` gate, `min_recall_at_k: 0.30` | a floor derived from a measurement taken through a blind matcher | catching a retrieval regression | every run, while a change destroying one item entirely would have scored 0.300 and **passed** (instance 9) |
| 3 | the v4 rerank script's `EARNS IT` verdict | `nDCG gain > 0` | DESIGN §5.5's *"nDCG must justify its 60ms"* | one run — caught by reading the number against the design rather than trusting the label beside it |

### The third, in detail

The v4 measurement printed **`EARNS IT`** for both reranked configurations. Measured cost
was **~3,900ms against a 60ms budget** — 65× over, and more than the entire 2.5s
flat-lookup p95 allowance consumed by one stage. The verdict was not a lie about the data;
every number it printed was correct. It was a lie about the *criterion*, because the script
asserted a threshold the design never set.

A quality gain of +0.070 nDCG is real and worth having. A script that calls it "earned"
against an invented bar converts a split verdict — *good quality, unshippable latency* —
into a single misleading word, and the word is the part a tired reader takes away.

### What the three have in common, and the practice

None of them failed. None of them had a bug in the sense of producing a wrong value. Each
one answered a question correctly, and the question was not the one that mattered:

* *direct* imports, not the import **closure**
* a floor from the measured number, not from a **valid** measured number
* `gain > 0`, not `gain > 0 at a cost under the budget`

The producer was fine in all three. The *specification* of the check was wrong, and nothing
tests a specification.

The only defence that has actually worked here is **mutation testing** — reintroduce the
defect, confirm the guard fails. A guard never observed failing is a guard whose question
has never been verified. That is practice 12 in the list below, and it was written after
occurrence 1; occurrences 2 and 3 both predate its application to the thing they guarded.

---

## A FOURTH pattern — the finding that exists only in its own summary

Named 2026-10-07, and it is the first one here with **no code in it at all.**

Instances 1-11 are code that ran and produced a wrong number. The second pattern is code
that is right and unread. The third is a guard working perfectly on the wrong question.
This one is **narration**: a claim about a document, offered in prose, that the document
does not make.

### What happened

A survey for `contradictory` candidates produced three. The third, C3, was reported as:

> `nivabupa-reassure2` p25 cl.6.2.4 and `nivabupa-rise` p23 cl.8.2.4 list **documents the
> claimant must supply**, against the IRDAI rule that the policyholder *"shall not be
> required to submit the documents"*.

Every checkable part of that was true. The clauses exist. The pages are right. Both ids
resolve cleanly in the index — better provenance than either surviving candidate. What the
clauses actually say is:

```
b. Documents required with claim form: [list]
```

A **list heading on a claim form**. It does not say the claimant must supply them; it does
not address who collects them at all. A form convention does not disagree with a rule about
who collects documents. **The obligation existed only in the paraphrase.**

### The consequence, stated plainly

The labeller would have written the item. It would have validated: two documents, two
resolving clause ids, a concept, a well-formed `surface_conflict` expectation. It would have
been committed, scored, and reported — and it would have tested a conflict that does not
exist. Every downstream number computed from it would have been arithmetically correct and
meaningless.

**Nothing in this project could have caught it.** The schema validates shape. The resolver
checks that a span's clause is on its page — and it was. Corpus resolution confirms the text
exists — and it does. There is no check anywhere that asks *does this clause say what the
summary says it says*, because that is a reading, and the harness reads nothing.

Nor is it like instance 4 ("my own diagnostics, wrong four times"). Those were scripts that
ran and printed a false number; a second script caught them. Here no code ran. The only
thing that caught it was going back to the PDF and reading the sentence again.

### Why it is structurally likely rather than careless

Summarising is the one step in this workflow with no artefact to check against. Every other
step produces something another step can contradict: a span resolves or it does not, a
clause id is in the index or it is not, a metric is computed or it is undefined. A candidate
described in prose produces a sentence, and a sentence is checked by reading the source —
which is the work the summary was supposed to save.

That makes it the hardest pattern to guard and the easiest to repeat. It is also the one
most likely to appear in a system where one party reads the documents and another writes the
labels, which is exactly this project's division of labour.

### Second instance (2026-10-08): "644 passed, ruff clean" while CI was red from day one

The same pattern, a different artefact. For a week I closed almost every batch with some
form of **"644 tests pass, ruff clean, mypy unchanged"** and treated the tree as shippable.
Every one of those statements was true of the commands I ran - `pytest -m 'not live'`,
`ruff check src tests` - and false of the gate CI actually runs.

The PR gate (`.github/workflows/ci.yml`) runs, in one Lint step, `ruff check` **and**
`ruff format --check`; I only ever ran the first. It then runs `mypy`, whose non-zero exit
I had been reporting as "5 pre-existing errors in untouched files" - a failing gate I read
as green. Checked against the actual run history: **main has been red on all 33 runs, since
the first commit (`689ffafe`, 2026-10-04). CI has never once passed.** The failure mode even
shifted under me - early commits failed Typecheck first, my later edits added format drift
so Lint fails first and now masks Typecheck, the coverage floor and the eval gate, none of
which have run on a recent commit at all.

The accurate claim all week would have been: *the subset of checks I run locally passes;
the full gate is red and has been since day one.* What I reported instead was a
self-selected subset, narrated as the whole. The code was fine; the reporting was wrong -
exactly C3's shape, pointed at my own status line instead of at a document.

Why it belongs with C3 and not with the code instances: nothing here produced a wrong
number. The gate was doing its job and saying so, loudly and continuously, in a place I was
not looking. The green came from running a smaller thing and calling it the bigger thing -
a claim about the state of the tree that the tree did not make. The guard against it is the
same as practice 16, moved from documents to process: **"CI is green" is a claim about the
CI run, not about the commands you happened to run locally; it is unverified until the run
is read.** A local `pytest` is to the PR gate what a prose summary is to the clause - a
convenience that is not the thing itself.

---

## What this pattern implies for how the project is built

Not "write more tests". These bugs passed their tests. The specific practices that caught
them, and that are now structural rather than remembered:

1. **Every metric is asserted against a hand-computed value**, with the arithmetic written
   into the test docstring. nDCG@5 of `[0,1,0,1,0]` is `1.0616/1.6309 = 0.651` — asserting
   against "whatever the implementation returns today" would have locked a wrong metric in
   rather than catching it.

2. **A diagnostic is not trusted until checked against the artefact it measures.** The
   corrections above are recorded in `docs/INGEST_FINDINGS.md` §6 and pinned by
   regression tests with the current counts as expected values, so the bad heuristics
   cannot come back silently.

3. **Detector counts are reported as floors, not estimates.** The conditional-override scan
   found 1 of 2 real instances, and LIMITATIONS says so in those words.

4. **Undefined is distinguished from zero.** Every retrieval metric returns `None`, never
   `0.0`, for an item with no ground truth — otherwise each adversarial item added to the
   golden set would drag reported recall down, meaning a *more* rigorous eval set produces
   *worse* numbers.

5. **An unverifiable label blocks.** A span that cannot be checked by anything is treated
   as failing, not passing, because uncheckable is indistinguishable from wrong.

6. **The eval gate is described as having no data** until human-authored labels exist. A
   mechanism that runs is not a result.

7. **A guard is mutation-tested before it is trusted.** Every architecture test added
   after instance 7 is checked by reintroducing the defect and confirming the test fails.
   The direct-import boundary test was green for the whole life of the project while the
   boundary it named was broken; it had never once been observed to fail, so nothing was
   actually known about what it detected.

8. **An architectural rule is checked over the import closure, not the file.** "The online
   path must not import PyMuPDF" is a property of the transitive graph. Checked per file it
   is a different, much weaker claim that happens to use the same words.

9. **A partially-correct artefact is treated as more dangerous than a missing one.** An
   absent clause code blocks and the labeller is stopped. A code present with an incomplete
   page list resolves, and quietly excludes a page somebody may have labelled correctly -
   the tool then tells them to move a correct span. Partial correctness is what earns the
   trust that the wrong part spends.

10. **An agreement between two components is tested explicitly, or it is not tested.**
   Instance 9 lived entirely in the gap between a correct chunker and a correct index. Every
   component test passed; no test asserted that the two used the same names. Where two
   modules must agree on a representation, the agreement gets its own assertion.

11. **A validator that says "fine" must mean the thing downstream needs.** `arag-eval
   validate` reported 0 errors on three labels that could never match a chunk, because it
   checked them against the index rather than against what retrieval can return. A check
   whose confidence exceeds its coverage is worse than no check.

12. **A field added to a failure taxonomy is not done until something branches on it, and
   a test asserts the branch.** Populating a signal correctly is half the work and the half
   that tests naturally cover. `truncated` was set correctly from the first commit, asserted
   by a passing test, and read by nothing - so a truncation was reprompted with the same
   budget, which cannot succeed, at two requests per item against a 20/day quota. The
   producer test proves the value is right; only a consumer test proves the value *does*
   anything.

13. **Audit for unread signals when the pattern is fresh, not when it bites.** One AST pass
   over attribute loads found 12 with no production consumer, including
   `AbstainReason.ONLY_SUPERSEDED_EVIDENCE` - the abstention this project is largely about.
   The audit costs minutes; each unread signal costs a wrong diagnosis at the moment it
   would have mattered most.

14. **A scripted provider tests what you imagined; only a real model tests what happens.**
   The declined-answer case had a passing test - written against JSON `null`, which is what
   the prompt asks for. The model emitted the string `"null"`. Sixty tests covered this
   engine and the first real call found two defects, because a stub returns exactly what
   its author pictured. Run the real thing early, with a cheap model if necessary.

15. **A guard's VERDICT must be checked against the written criterion, not against
   intuition.** Three guards in this project were green while the thing they existed for
   walked past, and all three answered a subtly different question than the one the design
   asked. Before trusting a pass, re-read what the design actually requires and confirm the
   assertion encodes *that* - `nDCG > 0` and `nDCG > 0 within 60ms` differ by one clause and
   by the entire conclusion.

16. **A candidate offered in prose is a claim about the document, not a finding, until the
    exact wording of both sides has been read.** Added after C3 — a contradictory candidate
    whose clauses, pages and ids were all correct and whose *conflict* was invented by the
    summary that reported it. The rule is operational, not aspirational: a candidate is
    quoted verbatim before it is labelled, and where a summary and a quotation disagree, the
    summary is wrong by default. Paraphrase is where the obligation gets added.


The through-line: **prefer a loud failure to a plausible output, at every layer.** Ingest
raises rather than emit a header-less table chunk. Validation refuses rather than accept a
span nothing can confirm. A stale cassette errors rather than replay against a changed
prompt. Each of those is a deliberate choice to be stopped rather than quietly misled.
