"""Fetch source PDFs from the manifest.

Design stance: **a fetch failure is data, not an exception.** Insurer sites rotate paths,
gate documents behind a User-Agent check, and occasionally serve an HTML error page with a
200 status. If the fetcher raised on the first problem, one dead URL would abort ingest for
the whole corpus and the operator would learn about exactly one failure per run. Instead
every source produces a ``FetchResult`` with a typed outcome, and the run reports all of
them at once.

Three specific hazards this handles, all of them observed on the real corpus:

1. **406 without a browser User-Agent** (nivabupa.com). This is HTTP content negotiation,
   not authentication — the document is published for the public. A document fetcher
   sending a real UA is ordinary behaviour, not evasion.
2. **HTML served with a 200** — a CDN error page or a login interstitial. Trusting the
   status code alone would write an HTML file to ``star-comprehensive-2025.pdf`` and the
   failure would surface much later as bizarre extraction output. So the payload is
   checked for the ``%PDF`` magic bytes and rejected on mismatch.
3. **Mirror drift.** Where a source declares mirrors, all of them are hashed and compared.
   Two copies of a regulatory instrument disagreeing is worth knowing about, and it is
   invisible unless something checks.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import httpx

from arag.ingest.manifest import Manifest, Source
from arag.obs import get_logger, span

log = get_logger(__name__)

PDF_MAGIC = b"%PDF"

# A real browser UA. nivabupa.com returns 406 to httpx's default UA even though the
# document is public; see hazard 1 in the module docstring.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/pdf,application/octet-stream,*/*",
    "Accept-Language": "en-IN,en;q=0.9",
}


class FetchOutcome(StrEnum):
    OK = "ok"
    CACHED = "cached"  # already on disk with a matching hash
    HTTP_ERROR = "http_error"  # non-2xx after mirrors exhausted
    NOT_PDF = "not_pdf"  # 200 but payload is not a PDF
    HASH_MISMATCH = "hash_mismatch"  # differs from expected_sha256 in the manifest
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    EMPTY = "empty"

    @property
    def usable(self) -> bool:
        return self in (FetchOutcome.OK, FetchOutcome.CACHED)


@dataclass
class Attempt:
    url: str
    status: int | None = None
    content_type: str | None = None
    bytes_downloaded: int = 0
    sha256: str | None = None
    error: str | None = None


@dataclass
class FetchResult:
    source_id: str
    outcome: FetchOutcome
    path: Path | None = None
    sha256: str | None = None
    size_bytes: int = 0
    attempts: list[Attempt] = field(default_factory=list)
    mirror_drift: bool = False
    detail: str | None = None

    @property
    def usable(self) -> bool:
        return self.outcome.usable


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _looks_like_pdf(data: bytes) -> bool:
    """Some servers prepend a BOM or whitespace, so scan the first few hundred bytes
    rather than requiring the magic at offset 0."""
    return PDF_MAGIC in data[:1024]


async def _try_url(client: httpx.AsyncClient, url: str) -> tuple[Attempt, bytes | None]:
    attempt = Attempt(url=url)
    try:
        response = await client.get(url)
    except httpx.TimeoutException as exc:
        attempt.error = f"timeout: {exc}"
        return attempt, None
    except httpx.HTTPError as exc:
        attempt.error = f"{type(exc).__name__}: {exc}"
        return attempt, None

    attempt.status = response.status_code
    attempt.content_type = response.headers.get("content-type")
    data = response.content
    attempt.bytes_downloaded = len(data)

    if response.status_code >= 400:
        attempt.error = f"HTTP {response.status_code}"
        return attempt, None
    if not data:
        attempt.error = "empty body"
        return attempt, None
    if not _looks_like_pdf(data):
        # Deliberately strict. Writing an HTML error page to a .pdf would turn a clear
        # network failure into a baffling extraction failure three stages later.
        attempt.error = f"payload is not a PDF (content-type {attempt.content_type})"
        return attempt, None

    attempt.sha256 = _sha256(data)
    return attempt, data


async def fetch_source(
    client: httpx.AsyncClient,
    source: Source,
    raw_dir: Path,
    *,
    force: bool = False,
) -> FetchResult:
    """Fetch one source, trying the primary URL then each mirror in order."""
    result = FetchResult(source_id=source.id, outcome=FetchOutcome.NETWORK_ERROR)
    target = raw_dir / f"{source.id}.pdf"

    if target.exists() and not force:
        # Named distinctly from the loop's `data` below: reusing one name makes the
        # narrowed type bytes here and bytes|None there, which is a real type error.
        cached_bytes = target.read_bytes()
        digest = _sha256(cached_bytes)
        if source.expected_sha256 in (None, digest):
            log.info("fetch_cached", source_id=source.id, sha256=digest[:12])
            return FetchResult(
                source_id=source.id,
                outcome=FetchOutcome.CACHED,
                path=target,
                sha256=digest,
                size_bytes=len(cached_bytes),
            )
        result.detail = (
            f"cached copy hash {digest[:12]} != manifest {source.expected_sha256[:12]}; refetching"
        )

    payload: bytes | None = None
    for url in source.urls:
        attempt, data = await _try_url(client, url)
        result.attempts.append(attempt)
        if data is not None and payload is None:
            payload = data
            result.sha256 = attempt.sha256
        # Compare mirrors against the primary rather than short-circuiting, so drift on a
        # regulatory document is detected instead of hidden by a lucky first hit.
        if (
            data is not None
            and url != source.url
            and result.sha256
            and attempt.sha256 != result.sha256
        ):
            result.mirror_drift = True
        if payload is not None and not source.mirror_urls:
            break

    if payload is None:
        last = result.attempts[-1] if result.attempts else None
        error = (last.error or "") if last else "no attempts"
        result.outcome = (
            FetchOutcome.TIMEOUT
            if "timeout" in error.lower()
            else FetchOutcome.NOT_PDF
            if "not a PDF" in error
            else FetchOutcome.EMPTY
            if "empty" in error
            else FetchOutcome.HTTP_ERROR
        )
        result.detail = error
        log.error("fetch_failed", source_id=source.id, outcome=result.outcome.value, detail=error)
        return result

    digest = _sha256(payload)
    if source.expected_sha256 and digest != source.expected_sha256:
        result.outcome = FetchOutcome.HASH_MISMATCH
        result.detail = f"got {digest[:12]}, manifest pins {source.expected_sha256[:12]}"
        log.error("fetch_hash_mismatch", source_id=source.id, detail=result.detail)
        return result

    raw_dir.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    result.outcome = FetchOutcome.OK
    result.path = target
    result.sha256 = digest
    result.size_bytes = len(payload)
    log.info(
        "fetch_ok",
        source_id=source.id,
        sha256=digest[:12],
        size_kb=round(len(payload) / 1024),
        mirror_drift=result.mirror_drift,
    )
    return result


async def fetch_all(
    manifest: Manifest,
    raw_dir: Path,
    *,
    force: bool = False,
    concurrency: int = 3,
) -> list[FetchResult]:
    """Fetch every source. Never raises for a per-source failure."""
    with span("ingest.fetch"):
        sem = asyncio.Semaphore(max(1, concurrency))
        async with httpx.AsyncClient(
            headers=DEFAULT_HEADERS,
            follow_redirects=True,
            timeout=httpx.Timeout(60.0, connect=20.0),
        ) as client:

            async def guarded(source: Source) -> FetchResult:
                async with sem:
                    return await fetch_source(client, source, raw_dir, force=force)

            return list(await asyncio.gather(*(guarded(s) for s in manifest.sources)))
