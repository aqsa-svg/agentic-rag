"""Unit tests for node 5(b)'s deterministic logic: parsing the judge's reply and the
pre-registered abstain rule. The MODEL's quality is a measurement (DESIGN §5.7,
`data/judge_validate.json`, where the 1.5B judge failed the bar); this file guards the
plumbing around it - that a verdict is read correctly and that the decision rule does what
it says, so a future stronger judge is scored through code known to be right.
"""

from __future__ import annotations

from arag.agent.judge import GroundedAnswersJudge, JudgeVerdict


def _parse(text: str, n: int = 2) -> JudgeVerdict:
    return GroundedAnswersJudge._parse(text, n_citations=n)


class TestParse:
    def test_well_formed(self) -> None:
        v = _parse('{"support": [true, false], "responsive": true, "reason": "x"}')
        assert v.parsed is True
        assert v.support == (True, False)
        assert v.responsive is True

    def test_fenced(self) -> None:
        v = _parse('```json\n{"support": [true], "responsive": false}\n```', n=1)
        assert v.parsed is True
        assert v.support == (True,)
        assert v.responsive is False

    def test_salvage_embedded_object(self) -> None:
        v = _parse('here is my verdict: {"support": [false], "responsive": true} done', n=1)
        assert v.parsed is True
        assert v.support == (False,)

    def test_string_booleans(self) -> None:
        v = _parse('{"support": ["true", "false"], "responsive": "true"}')
        assert v.support == (True, False)
        assert v.responsive is True

    def test_single_bool_support_is_broadcast(self) -> None:
        # A model sometimes collapses the per-citation list to one bool; broadcast to n.
        v = _parse('{"support": true, "responsive": true}', n=3)
        assert v.support == (True, True, True)

    def test_unparseable_fails_closed_to_parsed_false(self) -> None:
        v = _parse("the answer looks fine to me")
        assert v.parsed is False
        assert v.support == ()
        assert v.responsive is False


class TestAbstainRule:
    """Pre-registered rule: abstain iff NOT responsive OR no cited passage supports."""

    def test_responsive_and_some_support_does_not_abstain(self) -> None:
        v = JudgeVerdict(support=(True, False), responsive=True, reason="", parsed=True)
        assert v.abstains() is False

    def test_not_responsive_abstains(self) -> None:
        v = JudgeVerdict(support=(True, True), responsive=False, reason="", parsed=True)
        assert v.abstains() is True

    def test_no_support_abstains(self) -> None:
        v = JudgeVerdict(support=(False, False), responsive=True, reason="", parsed=True)
        assert v.abstains() is True

    def test_parse_failure_fails_open(self) -> None:
        # A judge that cannot be parsed must not manufacture an abstention - its own
        # unreliability would masquerade as the system refusing well (DESIGN §5.7).
        v = JudgeVerdict(support=(), responsive=False, reason="", parsed=False)
        assert v.abstains() is False

    def test_strict_variant_requires_all_support(self) -> None:
        v = JudgeVerdict(support=(True, False), responsive=True, reason="", parsed=True)
        # lenient: one supporting passage is enough -> answer
        assert v.abstains() is False
        # strict diagnostic: every cited passage must support -> the False forces abstain
        assert v.abstains_strict() is True
