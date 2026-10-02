"""Record/replay for deterministic, free, offline evaluation in CI.

## The trap this design avoids

The obvious way to build this is to cache the finished ``AnswerResult`` per question. It
is also useless as a regression gate, and it took a moment's thought to see why: if CI
replays stored answers, then the engine never runs, so **no change to the engine can ever
move a metric**. The gate would go green forever while retrieval quietly rotted. It would
detect bugs in the metric code and nothing else, while presenting itself as an
end-to-end quality gate. That is worse than having no gate, because it manufactures
confidence.

So cassettes here record at the **provider boundary** — the individual LLM, embedding and
judge HTTP calls — and nothing above it. On a PR the real planner runs, the real fusion
runs, the real prompt is assembled; only the network is replayed. A regression in ranking
or prompting therefore shows up as a changed metric, deterministically and for free.

## Why cassette invalidation is automatic, not manual

The cassette key is a hash of the **entire request payload**, prompt text included. Change
a prompt and the key changes, the lookup misses, and the run fails with an instruction to
re-record. It cannot silently serve the old response against the new prompt. This is the
one property that makes the whole scheme trustworthy: a stale cassette is a loud failure
rather than a quiet lie.

## What this still does not cover

Retrieval against a live Postgres is not replayed by this layer. The PR gate runs against
a committed fixture corpus instead (v3). Until that exists, the PR gate covers the metric
code, the schema, the behavioural checks and the engine contract — and the nightly live
run covers the rest. That limitation is stated in LIMITATIONS.md rather than papered over.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

from arag.obs import get_logger

log = get_logger(__name__)


class CassetteMode(StrEnum):
    OFF = "off"  # always call the real provider
    RECORD = "record"  # call the real provider and persist the response
    REPLAY = "replay"  # never touch the network; a miss is an error


class CassetteMiss(RuntimeError):
    """Raised in REPLAY when no recording exists for a request.

    The message names the offending cassette key and the fix, because the person hitting
    this is usually someone who changed a prompt and has no idea why CI failed.
    """

    def __init__(self, key: str, kind: str) -> None:
        super().__init__(
            f"No cassette for {kind} request {key[:12]}. "
            "This normally means a prompt, model id or request parameter changed since the "
            "cassettes were recorded — which is working as intended: stale recordings are "
            "never silently replayed. Re-record with `make record` and commit the diff."
        )
        self.key = key
        self.kind = kind


def _stable_key(kind: str, payload: Any) -> str:
    """Hash a request payload into a filename-safe key.

    ``sort_keys`` makes the hash independent of dict ordering, and ``default=str`` keeps
    dates and enums hashable. Anything non-deterministic (timestamps, request ids, a
    ``trace_id``) must be excluded by the caller *before* it gets here — including one
    would give every run a fresh key and turn every replay into a miss.
    """
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    digest = hashlib.sha256(f"{kind}\x00{blob}".encode()).hexdigest()
    return digest


class CassetteStore:
    """One JSON file per recorded interaction, under ``<root>/<kind>/<key>.json``.

    One file per interaction rather than one big file per suite: a single file would make
    every re-record produce an unreviewable diff, whereas per-interaction files let a
    reviewer see exactly which provider calls changed when a prompt is edited.
    """

    def __init__(self, root: Path, mode: CassetteMode = CassetteMode.OFF) -> None:
        self.root = root
        self.mode = mode
        self.hits = 0
        self.misses = 0
        self.recorded = 0

    def _path(self, kind: str, key: str) -> Path:
        return self.root / kind / f"{key}.json"

    def load(self, kind: str, key: str) -> Any | None:
        path = self._path(kind, key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["response"]

    def save(self, kind: str, key: str, payload: Any, response: Any) -> None:
        path = self._path(kind, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"kind": kind, "key": key, "request": payload, "response": response},
                indent=2,
                sort_keys=True,
                default=str,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        self.recorded += 1

    async def play_or_record(
        self,
        kind: str,
        payload: dict[str, Any],
        call: Callable[[], Awaitable[Any]],
    ) -> Any:
        """The single entry point every provider client routes through."""
        if self.mode is CassetteMode.OFF:
            return await call()

        key = _stable_key(kind, payload)

        if self.mode is CassetteMode.REPLAY:
            cached = self.load(kind, key)
            if cached is None:
                self.misses += 1
                raise CassetteMiss(key, kind)
            self.hits += 1
            return cached

        # RECORD: prefer an existing recording so re-recording is incremental and the
        # committed diff stays small and reviewable.
        cached = self.load(kind, key)
        if cached is not None:
            self.hits += 1
            return cached
        response = await call()
        self.save(kind, key, payload, response)
        log.info("cassette_recorded", kind=kind, key=key[:12])
        return response

    def stats(self) -> dict[str, int | str]:
        return {
            "mode": self.mode.value,
            "hits": self.hits,
            "misses": self.misses,
            "recorded": self.recorded,
        }
