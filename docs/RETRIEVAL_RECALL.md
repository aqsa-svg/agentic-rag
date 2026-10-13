# Retrieval recall — what the ranker can and cannot reach (2026-10-13)

This measures **retrieval alone**, no LLM: for the 20 labelled golden items
(`data/golden/v1.jsonl`, `expected_behaviour ∈ {answer, surface_conflict}`), does the gold
evidence span appear in the top-k retrieved chunks? Recall here is the ceiling on grounding —
the generator cannot cite a passage retrieval never returned, so every grounding number in
`AGENT_EVAL.md` is bounded by this one.

## Headline

**Dense retrieval recovers recall@10 = 0.625 of the labelled evidence; the deployed BM25
recovers 0.525. The remaining 0.375 is a ranking gap, not a coverage gap — the ceiling is
1.0.** Every gold span *is* carried by a chunk (verified below), so the misses are passages
ranked outside top-k, concentrated in exclusion/supersession clauses. Two standard fixes —
RRF hybrid and a cross-encoder reranker — were measured and **do not close it**.

## Committed, reproducible: recall@10 by retriever

All three run through the same committed path (`RetrievalOnlyEngine` → `score_retrieval`,
k from `thresholds.yaml` = 10), so these are the harness's own metric, not a side script:

```
arag-eval run --engine bm25         --smoke --no-log
arag-eval run --engine bm25-row-nl  --smoke --no-log
arag-eval run --engine dense        --smoke --no-log      # new; was an uncommitted script
```

| engine | serialisation | recall@10 | nDCG@10 |
|---|---|---:|---:|
| bm25 | markdown | 0.525 | 0.447 |
| bm25-row-nl | row_nl | 0.525 | 0.454 |
| **dense** (bge-base-en-v1.5) | markdown | **0.625** | 0.503 |

Dense wins by +0.10 recall@10. Serialisation (markdown vs row_nl) does not move BM25 recall
at k=10. **The public instance serves BM25** (dense needs the embedding model at runtime,
too heavy for the free tier), so the live demo retrieves at the 0.525 line, not 0.625; this
is why the deployed `top_k` was raised from 5 to 10 (the free lever — see below).

## The ceiling is 1.0 (misses are ranking, not ingestion)

For each of the 20 items, **every** gold span is covered by at least one chunk, in both
serialisations (checked with the same `SpanMatcher` the metric uses, over the whole chunk
store — not just top-k). So no miss is an un-chunked span or a mislabelled page; each one is
a passage the retriever *has* and ranks too low. That is the good case: recall is improvable
by better ranking rather than blocked by missing text.

## Exploratory sweep (score_retrieval directly, k = 5/10/20)

Relative comparison across retrievers, including two not wired into the CLI (hybrid, rerank).
Absolute values run ~0.05 below the committed harness because this path uses a plain matcher;
the **ordering** is what matters here.

| retriever | R@5 | R@10 | R@20 |
|---|---:|---:|---:|
| bm25-row_nl *(deployed)* | 0.425 | 0.500 | 0.625 |
| dense | 0.450 | 0.575 | 0.725 |
| rerank(dense, fetch 30) | 0.450 | 0.600 | — |
| rerank(bm25, fetch 30) | 0.450 | 0.525 | — |
| RRF hybrid(dense+bm25) | 0.425 | 0.475 | 0.575 |

Two negatives, both measured rather than assumed:

* **RRF hybrid ≤ dense everywhere.** Fusing BM25's weaker ranking into dense's dilutes it;
  equal-weight RRF at the default k=60 buys nothing on a corpus that returns tens, not
  thousands, of candidates. (Consistent with the reranker module's note on h-28.)
* **The cross-encoder reranker adds one item at k=10 (0.575 → 0.600) and nothing at k=5.**
  It cannot promote what the first stage ranked outside its fetch window, and the hard misses
  sit there. This is a *quality* number to put beside the *latency* reason the reranker was
  already non-adopted (DESIGN §5.5): it does not earn its 60 ms here either.

## Where the misses are, and why

The spans missed by every retriever at the tested k — **h-03, h-14, h-32, h-33** — are
exclusion and supersession clauses (`Code Excl 03`, `Code Excl 06`, p35 supersession text).
The queries are lay language ("my local nursing home", "treatment taken overseas"); the gold
chunks are legal codes. Neither BM25 term overlap nor a general-purpose bi-encoder bridges
"my overseas treatment" → "Code Excl 03". **Closing this needs a vocabulary bridge** — LLM
query expansion or a domain-adapted embedder — not another fusion of the retrievers we have.
That experiment is tracked separately; this document records the baseline it must beat.

## One-line

On the 20 labelled items, **dense retrieval reaches recall@10 = 0.625 vs 0.525 for the
deployed BM25, against a ceiling of 1.0**; hybrid fusion and cross-encoder reranking were
measured and do not close the gap, which is a lay-language-to-legal-code vocabulary problem
in the exclusion/supersession clauses.
