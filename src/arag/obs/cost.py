"""Token accounting and cost attribution.

Two numbers are reported for every query, and the distinction matters:

* **marginal_usd** — what this query actually cost. On the Gemini free tier that is
  ``0.0``, which is a true but useless number to put in a README.
* **shadow_usd** — what the same token counts would cost on a named paid model. This is
  the number a reviewer actually wants, because it answers "what does this cost at
  scale?" without pretending the free tier is a business model.

The price table is **versioned and dated**. Every published cost claim must cite a
``PRICE_TABLE_VERSION`` so that a number in the README can be reproduced later even after
providers change their pricing.

!!! VERIFY BEFORE PUBLISHING !!!
The rates below are working values entered on the date shown. They MUST be re-checked
against each provider's current pricing page before any cost figure goes into the README.
`make verify-prices` is the stub for that check; until it exists, treat these as estimates
and say so. An unverified cost claim is exactly the kind of unsourced number this project
exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

PRICE_TABLE_VERSION = "2026-09-04.a"


@dataclass(frozen=True)
class Rate:
    """USD per million tokens."""

    input_per_mtok: Decimal
    output_per_mtok: Decimal
    as_of: str
    verified: bool = False
    note: str = ""


# Keys are internal model ids, not vendor strings, so a provider swap does not ripple.
PRICES: dict[str, Rate] = {
    "gemini-flash-free": Rate(
        Decimal("0"),
        Decimal("0"),
        "2026-09-04",
        verified=False,
        note="Free tier: 0 marginal cost, rate limited. Data-usage policy is spike S1.",
    ),
    "gemini-flash-paid": Rate(
        Decimal("0.10"),
        Decimal("0.40"),
        "2026-09-04",
        verified=False,
        note="Shadow-cost reference for the same model class on the paid tier.",
    ),
    "claude-haiku-4-5": Rate(
        Decimal("1.00"),
        Decimal("5.00"),
        "2026-09-04",
        verified=False,
        note="Shadow-cost reference for a stronger generator.",
    ),
    "local-ollama": Rate(
        Decimal("0"),
        Decimal("0"),
        "2026-09-04",
        verified=True,
        note="Own hardware. Zero marginal cost by definition; electricity not modelled.",
    ),
}

SHADOW_MODEL = "gemini-flash-paid"


@dataclass
class Usage:
    """Accumulated token usage for one traced unit of work (a query, or an eval item)."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    per_model: dict[str, tuple[int, int]] = field(default_factory=dict)

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        prev_in, prev_out = self.per_model.get(model, (0, 0))
        self.per_model[model] = (prev_in + input_tokens, prev_out + output_tokens)

    def marginal_usd(self) -> Decimal:
        total = Decimal("0")
        for model, (tin, tout) in self.per_model.items():
            rate = PRICES.get(model)
            if rate is None:
                # Unknown model must not silently cost zero.
                raise KeyError(f"no price entry for model {model!r}; add it to PRICES")
            total += _cost(rate, tin, tout)
        return total

    def shadow_usd(self, model: str = SHADOW_MODEL) -> Decimal:
        """Cost of the same token counts on a named paid model."""
        return _cost(PRICES[model], self.input_tokens, self.output_tokens)


def _cost(rate: Rate, input_tokens: int, output_tokens: int) -> Decimal:
    million = Decimal("1000000")
    return (
        Decimal(input_tokens) / million * rate.input_per_mtok
        + Decimal(output_tokens) / million * rate.output_per_mtok
    )


def unverified_models() -> list[str]:
    """Used by the report generator to stamp cost tables as estimates rather than facts."""
    return sorted(name for name, rate in PRICES.items() if not rate.verified)
