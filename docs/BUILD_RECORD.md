# Build record — how the v1 modules were produced and verified

Kept because the provenance of code matters the same way the provenance of a golden-set
label matters. `authored_by: human` on a label written by an assistant falsifies the claim
the golden set rests on; "481 tests passing" on modules that skipped their review stage
falsifies the claim the build process rests on.

---

## v1 baseline modules — 2026-09-04

Four modules were built in parallel by subagents against interfaces fixed in advance, with
an adversarial reviewer designed to follow each build. The reviewer's brief was specific:
*four silent-wrongness bugs have already been found in this project, assume a fifth is
present, report only defects verified by execution.*

**Three of the four agents were killed by a session API limit before returning.**

| module | files | build agent | adversarial review |
|---|---|---|---|
| BM25 index + retriever | `index/lexical.py`, `retrieval/lexical.py`, `tests/test_lexical.py` | **completed**, 49 tests, self-reported ruff clean | **killed** before running |
| prose chunker | `ingest/chunk.py`, `tests/test_ingest_chunk.py` | **killed** after writing files, before verifying | never started |
| build pipeline + T3 harness | `index/build.py`, `tests/test_index_build.py` | **killed** after writing files, before verifying | never started |

### Verification status: verified by the orchestrator, not independently reviewed

This is the honest description and it should not be softened. What actually happened:

- The two killed build agents had written their modules *and* their tests before dying.
  Those tests pass. That is worth something, but it is **self-consistency**, not review —
  the same author wrote the code and the assertions, which is precisely the condition
  under which instance 3 in `docs/SILENT_WRONGNESS.md` survived (a case-sensitive prefix
  strip, tested only with the spelling the author had in mind).
- The orchestrator ran the full suite, ruff, and mypy strict across the integrated result,
  fixed a mypy override for PyMuPDF, and empirically tested the one defect the surviving
  agent reported. That is integration verification.
- **No adversarial pass was made over `chunk.py`, `build.py`, or `lexical.py`.** Nobody
  tried to break them. Nobody checked whether their tests assert hand-computed values or
  merely assert whatever the implementation produces — which is the check that matters
  most, and the one the reviewer stage existed to perform.

So: **481 passing tests, verified by the orchestrator, not independently reviewed.**

Until a review happens, treat the following as unverified by anyone but their authors:

- the prose chunker's R2 guarantees (definitions and sub-clauses never split)
- the BM25 scoring arithmetic beyond the one hand-computed case its own tests assert
- the T3 harness's token estimate and percentile arithmetic

`tests/test_import_cycle.py::TestOrchestratorVerifiedModules` asserts this file still says
so, so the gap cannot quietly disappear from the record.

### The one defect the surviving agent did report

It flagged a circular import — `arag.index.__init__` → `arag.index.build` →
`arag.retrieval.lexical` — whose `ImportError` is swallowed by `build.py` into
`_MISSING_MODULES`, producing a silently absent module rather than a crash. It fixed its
own side with a `TYPE_CHECKING` import and said the fix was fragile.

Verified empirically across five import orders: `_MISSING_MODULES` is empty in all of
them. Now pinned by `tests/test_import_cycle.py` so a future module-scope import cannot
reintroduce it silently. Recorded as instance 6 in `docs/SILENT_WRONGNESS.md`.

### Decisions the surviving agent made that are worth keeping

Recorded because they are the kind of reasoning that would otherwise be lost, and because
each names the alternative it rejected:

- **idf uses the Robertson/Lucene variant** `ln(1 + (N-n+0.5)/(n+0.5))`, strictly positive
  for all n ≤ N. Rejected the classic form, which goes negative for a term appearing in
  more than half the corpus — and in a six-document corpus, "policy" does exactly that —
  then needs an undocumented clamp that silently changes ranking.
- **Filters narrow the candidate set before scoring**, not the result list after. Rejected
  post-filtering: a document-scoped query would come back short because the global top-30
  came from excluded documents, which reads as "no such clause".
- **idf is never recomputed over a filtered subset** — it must stay a corpus property, or
  the same chunk scores differently depending on the filter sent.
- **`to_dict` serialises integer counts only** and `from_dict` recomputes idf/avgdl, so a
  round trip is bit-identical. Rejected serialising the floats, which would let a stale
  idf table load beside counts that no longer justify it.
- **`add()` raises on a duplicate `chunk_id`.** Rejected accumulating — doubled term
  frequencies are indistinguishable from a relevance win.
- **A chunk that tokenises to nothing is kept in N and avgdl, and logged.** Rejected
  excluding it, which would misstate corpus statistics to make avgdl look tidier.
- **Ties break on `chunk_id` ascending.** Rejected insertion order, which would move an
  eval number whenever ingest order changed.

## v2 dense retrieval — 2026-09-16

Four modules added: `retrieval/embedding.py` (the `Embedder` protocol),
`index/dense.py` (`DenseIndex` + the offline sentence-transformers embedder),
`retrieval/dense.py` (`DenseRetriever`), `retrieval/hybrid.py` (RRF fusion). Filtering was
extracted from `retrieval/lexical.py` into `retrieval/filtering.py` so both retrievers
apply identical semantics — RRF over two differently-filtered corpora would be incoherent
in a way no metric reports, and the field that breaks first is `as_of`, the supersession
control.

22 new tests in `tests/test_dense.py`, every expected value hand-computed. 532 pass overall.

### mypy is environment-blocked; these four modules are UNVERIFIED by it

`mypy` will not start on this machine. Exact error, reproduced on two consecutive runs:

```
  File "mypy\build.py", line 99, in <module>
  File "mypy\ipc.py", line 22, in <module>
ImportError: DLL load failed while importing base64: An Application Control policy has
blocked this file.
```

A Windows Application Control policy is blocking a DLL mypy loads at import. It is not a
type error and not a code change — mypy ran clean over 46 source files earlier in the same
session, before any v2 code existed, so the machine changed rather than the project.

Consequence, stated plainly rather than glossed: **the four v2 modules have never been
type-checked.** `ruff check` and `ruff format` pass, the tests pass, but strict typing —
which this project claims in its CI and its README plan — is currently unenforced on them.
Not retried, at the author's instruction. CI on a Linux runner is unaffected and will be
the first real check.

### Open concerns from the same agent, not yet addressed

- **PyMuPDF reaches the online path transitively — silent-wrongness instance 7, and the
  first one that breaks in production rather than corrupting a number.** `arag.index.lexical` must import
  `arag.ingest.normalise` (correct, for the N1 invariant), and `arag/ingest/__init__.py`
  eagerly imports `arag.ingest.probe`, which imports PyMuPDF. The AST-based
  import-boundary test only checks *direct* imports, so it does not catch this. It is a
  real concern for the deployed bundle and needs either a lazy `arag.ingest.__init__` or a
  transitive check in the boundary test. Fix is deferred deliberately: it gates
  the day-7 deployment, not the baseline, and the baseline was the committed deliverable.
- **`src/arag/index/__init__.py` is a last-writer-wins merge** of two agents' export
  surfaces. It currently exports both; it has not been reviewed as a deliberate API.
