"""The embedder interface, and the asymmetry that makes it easy to get silently wrong.

## Why a protocol here rather than a concrete class

``arag.retrieval`` is an ONLINE package: DESIGN §9 splits offline compute (PyMuPDF, a GPU,
sentence-transformers, torch) from the online request path, which must stay CPU-small.
``tests/test_import_closure.py`` enforces that over the whole import closure, not per file
— see instance 7 in ``docs/SILENT_WRONGNESS.md`` for what happened the last time only
direct imports were checked.

So the dense retriever cannot import a model. It is handed one. The concrete
``SentenceTransformerEmbedder`` lives in ``arag.index.dense`` (offline), the ONNX one will
live beside it, and neither is reachable from a module the serverless function loads.

## The asymmetry: queries and passages are NOT embedded the same way

BGE models are trained with an instruction prefix on the **query side only**:

    query:    "Represent this sentence for searching relevant passages: " + text
    passage:  text

This is the single easiest thing to get wrong in a dense pipeline, and it fails in the
project's signature shape: **no error, plausible output, quietly worse recall.** Embed the
query without the prefix and every vector is still 768-dimensional, still unit-norm, still
produces a ranking — just a measurably worse one. Nothing anywhere raises.

That is why the protocol has two methods rather than one ``embed``. A single method would
put the decision at every call site, and one call site forgetting it would be invisible.
With two, forgetting is a type-level mistake at worst and a named method at best.

``Embedder.model_id`` exists for the same reason: an index built with one model and queried
with another returns confident nonsense. ``DenseIndex`` refuses that at load time rather
than scoring it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# bge-*-en-v1.5's documented query instruction. Applied to queries, never to passages.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@runtime_checkable
class Embedder(Protocol):
    """Turns text into unit-norm vectors.

    Implementations MUST return L2-normalised vectors, so that a dot product is cosine
    similarity and the index never has to re-normalise. Returning un-normalised vectors
    does not raise; it silently changes what "similarity" means, and ranks long chunks
    above short ones.
    """

    model_id: str
    dim: int

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed corpus text. No instruction prefix."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query. Applies the model's query instruction, if it has one."""
        ...
