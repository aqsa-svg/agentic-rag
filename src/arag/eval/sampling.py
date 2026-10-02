"""Sampling for the fast subset and for judge-agreement labelling.

Both callers need the same property and it is not what ``head -n`` gives you:
**every stratum must be represented.** A 20-item fast subset taken off the top of the file
would be 20 flat lookups, and the PR gate would then be blind to exactly the multi-hop and
adversarial regressions it exists to catch. Round-robin across strata keeps the subset
diagnostic at a fraction of the cost.

Sampling is deterministic given a seed so that two runs of ``make eval-fast`` on the same
commit score the same items. A gate that samples differently each run produces metric
noise that is indistinguishable from a regression.
"""

from __future__ import annotations

import random
from collections import defaultdict

from arag.eval.schema import GoldenItem, GoldenSet, Strata

FAST_SUBSET_SIZE = 20
FAST_SUBSET_SEED = 20260904


def stratified_subset(
    golden: GoldenSet,
    n: int = FAST_SUBSET_SIZE,
    *,
    seed: int = FAST_SUBSET_SEED,
    always_include_adversarial: bool = True,
) -> GoldenSet:
    """Pick ``n`` items spread across strata, deterministically.

    ``always_include_adversarial`` front-loads the unanswerable, contradictory and
    injection strata. Those three are where a regression is most expensive and least
    likely to be noticed by eyeballing an answer, so they get first claim on the budget
    rather than whatever is left after the ordinary strata have taken their share.
    """
    if n <= 0:
        raise ValueError("subset size must be positive")

    buckets: dict[Strata, list[GoldenItem]] = defaultdict(list)
    for item in golden.items:
        buckets[item.strata].append(item)

    rng = random.Random(seed)
    for items in buckets.values():
        items.sort(key=lambda i: i.id)  # stable base order before shuffling
        rng.shuffle(items)

    adversarial = [s for s in Strata if s.is_adversarial and buckets.get(s)]
    ordinary = [s for s in Strata if not s.is_adversarial and buckets.get(s)]
    order = (adversarial + ordinary) if always_include_adversarial else (ordinary + adversarial)

    picked: list[GoldenItem] = []
    cursor = {s: 0 for s in order}
    while len(picked) < n:
        progressed = False
        for stratum in order:
            idx = cursor[stratum]
            pool = buckets[stratum]
            if idx < len(pool):
                picked.append(pool[idx])
                cursor[stratum] = idx + 1
                progressed = True
                if len(picked) == n:
                    break
        if not progressed:
            break  # exhausted the whole set before reaching n

    return GoldenSet(items=tuple(sorted(picked, key=lambda i: i.id)), source=golden.source)
