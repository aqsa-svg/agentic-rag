"""Cassette tests.

The property under test is the one that makes the whole PR gate trustworthy: **a changed
prompt must invalidate the recording rather than replay a stale one.** If that fails, CI
silently scores the new prompt against the old model output and reports green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arag.eval.cassettes import CassetteMiss, CassetteMode, CassetteStore, _stable_key


@pytest.fixture
def calls() -> dict[str, int]:
    return {"n": 0}


def make_call(counter: dict[str, int], response: str = "ok"):  # type: ignore[no-untyped-def]
    async def _call() -> str:
        counter["n"] += 1
        return response

    return _call


class TestKeying:
    def test_key_is_order_independent(self) -> None:
        """dict ordering must not change the key, or every re-record churns every file."""
        a = _stable_key("llm", {"model": "m", "prompt": "p"})
        b = _stable_key("llm", {"prompt": "p", "model": "m"})
        assert a == b

    def test_changed_prompt_changes_the_key(self) -> None:
        """The invalidation property. This is the assertion the gate's honesty rests on."""
        before = _stable_key("llm", {"model": "m", "prompt": "answer strictly"})
        after = _stable_key("llm", {"model": "m", "prompt": "answer strictly and cite"})
        assert before != after

    def test_changed_model_changes_the_key(self) -> None:
        a = _stable_key("llm", {"model": "gemini-2.0-flash", "prompt": "p"})
        b = _stable_key("llm", {"model": "gemini-2.5-flash", "prompt": "p"})
        assert a != b

    def test_kind_is_part_of_the_key(self) -> None:
        assert _stable_key("llm", {"x": 1}) != _stable_key("judge", {"x": 1})


class TestModes:
    @pytest.mark.asyncio
    async def test_off_always_calls_through(self, tmp_path: Path, calls: dict[str, int]) -> None:
        store = CassetteStore(tmp_path, CassetteMode.OFF)
        for _ in range(3):
            assert await store.play_or_record("llm", {"p": 1}, make_call(calls)) == "ok"
        assert calls["n"] == 3
        assert store.recorded == 0

    @pytest.mark.asyncio
    async def test_record_then_replay(self, tmp_path: Path, calls: dict[str, int]) -> None:
        payload = {"model": "m", "prompt": "p"}

        recorder = CassetteStore(tmp_path, CassetteMode.RECORD)
        assert await recorder.play_or_record("llm", payload, make_call(calls)) == "ok"
        assert calls["n"] == 1
        assert recorder.recorded == 1

        player = CassetteStore(tmp_path, CassetteMode.REPLAY)
        assert await player.play_or_record("llm", payload, make_call(calls)) == "ok"
        assert calls["n"] == 1, "replay must not touch the network"
        assert player.hits == 1

    @pytest.mark.asyncio
    async def test_record_is_incremental(self, tmp_path: Path, calls: dict[str, int]) -> None:
        """Re-recording an unchanged interaction reuses the file so the committed diff
        stays reviewable instead of rewriting every cassette on every run.
        """
        store = CassetteStore(tmp_path, CassetteMode.RECORD)
        payload = {"p": 1}
        await store.play_or_record("llm", payload, make_call(calls))
        await store.play_or_record("llm", payload, make_call(calls))
        assert calls["n"] == 1
        assert store.recorded == 1
        assert store.hits == 1

    @pytest.mark.asyncio
    async def test_replay_miss_raises_with_a_fix_in_the_message(
        self, tmp_path: Path, calls: dict[str, int]
    ) -> None:
        """A stale cassette must be a loud failure, never a quiet lie."""
        store = CassetteStore(tmp_path, CassetteMode.REPLAY)
        with pytest.raises(CassetteMiss) as exc:
            await store.play_or_record("llm", {"prompt": "new"}, make_call(calls))
        assert calls["n"] == 0
        assert "make record" in str(exc.value)
        assert store.misses == 1

    @pytest.mark.asyncio
    async def test_prompt_change_invalidates_the_recording(
        self, tmp_path: Path, calls: dict[str, int]
    ) -> None:
        """End-to-end version of the key test: record with one prompt, replay with
        another, and the harness refuses rather than serving the old response.
        """
        recorder = CassetteStore(tmp_path, CassetteMode.RECORD)
        await recorder.play_or_record("llm", {"prompt": "v1"}, make_call(calls, "old answer"))

        player = CassetteStore(tmp_path, CassetteMode.REPLAY)
        with pytest.raises(CassetteMiss):
            await player.play_or_record("llm", {"prompt": "v2"}, make_call(calls))


class TestPersistence:
    @pytest.mark.asyncio
    async def test_file_layout_and_content(self, tmp_path: Path, calls: dict[str, int]) -> None:
        store = CassetteStore(tmp_path, CassetteMode.RECORD)
        await store.play_or_record("judge", {"prompt": "score this"}, make_call(calls, "5"))

        files = list((tmp_path / "judge").glob("*.json"))
        assert len(files) == 1
        body = files[0].read_text(encoding="utf-8")
        assert '"kind": "judge"' in body
        # The request is stored alongside the response so a reviewer can see what changed.
        assert "score this" in body

    @pytest.mark.asyncio
    async def test_stats_reports_mode(self, tmp_path: Path) -> None:
        store = CassetteStore(tmp_path, CassetteMode.REPLAY)
        assert store.stats()["mode"] == "replay"
