# v4 — the cross-encoder reranker: built, measured, NOT SHIPPED

> ## Decision, 2026-10-05: the reranker is not in the online path.
>
> This is a **choice made against a measurement**, not a gap in the work. The code exists,
> it is tested, it produces **the best ranking quality this project has measured** —
> nDCG **0.595** against dense's **0.526**, MRR 0.440 — and it is excluded because it costs
> **4,108 ms against DESIGN §5.5's 60 ms budget**: 65x over, more than the entire 2.5 s
> flat-lookup p95 allowance in a single stage.
>
> Shipping it would have bought ~~+0.070~~ nDCG and broken the latency budget by a factor of
> 65. The online configuration therefore remains **v3 — dense + supersession filter,
> 208 ms, nDCG 0.526** — and the reranker stays an offline quality ceiling, the role
> DESIGN §5.5 reserves for exactly this situation.
>
> **nDCG correction, 2026-10-08 — the decision is STRENGTHENED, not changed.** The 0.595 /
> 0.526 figures used the over-crediting nDCG (`docs/SILENT_WRONGNESS.md`: it summed per-chunk
> relevance, so a span covered by several chunks was counted repeatedly). The v4 run stored no
> per-span ranks, so it was re-run on its original items with the fixed metric and the same
> reranker: **dense 0.383, dense + rerank 0.395 — rerank still gives the best ranking quality,
> but the gain is +0.012, not +0.070.** The inflated figures were roughly 6x the real gain.
> The reranker now reads as buying **+0.012 nDCG for 68x the latency budget**, so the
> NOT-SHIPPED decision - which rested on latency and never on the nDCG value - is reinforced:
> a smaller gain for the same cost. (The absolute numbers differ from the 0.526 / 0.595 above
> because those applied the as_of filter and the re-run did not; the RELATIVE rerank effect,
> same config on both sides, is what the correction speaks to.)
>
> It becomes shippable when the ONNX-INT8 path closes the gap between 3,900 ms and 60 ms.
> That work is specified and not done.

Measured 2026-10-05. `cross-encoder/ms-marco-MiniLM-L-6-v2`, PyTorch on CPU, `fetch_k=30 →
top_k=10`, `as_of` filtering on throughout. Same corpus, labels and embeddings as v3; the
only variable is whether a reranker sits after retrieval.

Raw: `data/v4_rerank.json`.

---

## The numbers

| engine | recall@10 | nDCG@10 | MRR | ctx precision | hit rate | latency |
|---|---|---|---|---|---|---|
| dense | 0.700 | ~~0.526~~ 0.383 corrected | 0.375 | 0.160 | 0.800 | **208 ms** |
| dense + rerank | 0.700 | ~~**0.595**~~ 0.395 corrected | **0.440** | 0.160 | 0.800 | **4,108 ms** |
| hybrid | 0.400 | 0.386 | 0.367 | 0.080 | 0.600 | **123 ms** |
| hybrid + rerank | **0.500** | **0.463** | **0.440** | 0.100 | **0.800** | **4,139 ms** |

## The verdict, which is not what the run printed

The measurement script printed **"EARNS IT"** for both. That verdict is **wrong**, and it is
worth recording why rather than quietly fixing the label.

The script tested `nDCG gain > 0`. DESIGN §5.5 sets a different bar: *"nDCG@k is the metric
that must justify its **60ms**."* Measured cost is **~3,900ms** — **65× the budget**, and
more than the entire 2.5s p95 allowance for a flat lookup consumed by one stage.

So the honest verdict is split, and both halves matter:

* **On quality, the reranker earns its place.** +0.070 nDCG on dense, +0.077 on hybrid, and
  MRR to 0.440 on both — the best ranking quality this project has measured.
* **On latency as currently implemented, it is unusable.** Not close to the budget; a
  different order of magnitude.

A test that reports "earns it" against a threshold the design never set is the same defect
class as instance 9's gate: a check whose verdict is structurally unable to catch the thing
it exists to catch. Caught here by reading the number against DESIGN rather than trusting
the label printed beside it.

## Why the gain is real even though the latency is not

The cross-encoder does the one thing neither retriever can. Bi-encoder retrieval embeds
query and passage *separately*, so the passage's representation is fixed before the query
is known; BM25 scores term overlap. A cross-encoder runs the pair through one forward pass,
so attention crosses between them.

That is precisely the gap v2 exposed. The sharpest evidence is **h-02**:

```
first-hit rank      dense=8   dense+rerank=2   hybrid=MISS   hybrid+rerank=2
```

`star-comprehensive-2025:p31:s142` — the chunk carrying `excl.02`'s text — sat at **rank 8**
under dense and was **absent entirely** from hybrid's top 10. The reranker pulls it to
**rank 2** in both. The hybrid case is the stronger one: RRF had pushed a correct span out
of the window, which is the documented cost of discarding score magnitude, and the
cross-encoder recovers it by scoring the pair directly.

That also explains **hybrid's recall 0.400 → 0.500 and hit rate 0.600 → 0.800**: reranking a
30-candidate pool surfaces spans the 10-item window had cut.

## What does not improve, and the one regression

**h-03 is still missed by every configuration.** Its chunk is reachable — the agreement test
proves a span on `excl.03 p32` matches `star-comprehensive-2025:p32:s146` — and it is not in
the top 30 for the reranker to rescue. A reranker reorders what retrieval found; it cannot
find what retrieval missed. h-03 remains the cleanest open diagnosis target in the set.

**h-04 regressed, 4 → 5 on dense and 3 → 5 on hybrid.** Small, one item, and not dismissible:
h-04 is the grace-period conditional override whose answer depends on payment mode, and a
cross-encoder optimising apparent query-passage relevance has no reason to prefer the
clause that *gates* an answer over the one that states it. Worth watching as the
`conditional_override` stratum grows.

## What this makes required rather than optional

DESIGN §5.4 already specified quantised ONNX-INT8 for the online reranker. That was written
as an engineering preference. It is now a **hard requirement with a number attached**: the
PyTorch CPU path cannot ship at 4.1s, and the gap between 3,900ms and the 60ms budget is
what the ONNX work has to close.

Until that is measured, the correct configuration for the online path is **dense retrieval
with the supersession filter and no reranker** — 208ms, nDCG 0.526 — and the reranker stays
an offline quality ceiling, which is exactly the role DESIGN §5.5 reserved for the larger
`bge-reranker-v2-m3`.

## Caveats

Seven labelled items, two of them `unanswerable` and therefore undefined for retrieval
metrics, so every figure rests on five scored items. One item moving changes nDCG by
roughly 0.1. Development signal, not a result.

Latency is PyTorch on 8 CPU threads with nothing else running; it is the honest cost of
*this* implementation, not of cross-encoders generally.
