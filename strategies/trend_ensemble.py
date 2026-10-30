"""Volatility-targeted time-series momentum ensemble.

Once per ``rebalance_hours`` (UTC-aligned), look at the sign of the
return over three horizons (default 10/30/90 days) and average them. The
average is the conviction: +1 when all three agree on up, 1/3 when only
one does. Exposure is conviction x min(1, target_vol / realised_vol), so
the book runs smaller in violent markets and larger in calm trends.

Why it should hold up: trend premia in crypto come from slow-moving
capital and reflexive flows that play out over weeks, not minutes.
Decisions happen on a fixed daily clock using closes sampled at the
boundary, so 5m and 4h data produce nearly the same trades. Several
horizons and continuous sizing mean no single lookback has to be right.
"""

from __future__ import annotations

import math

from strategies.indicators import LogReturnVol
from strategies.signal_base import BarData
from strategies.signal_base import Param
from strategies.signal_base import Signal


class TrendEnsemble(Signal):
    key = "trend_ensemble"
    title = "Trend ensemble (vol-targeted momentum)"
    summary = "Sign of 10/30/90-day returns, averaged, sized to a volatility target. Rebalanced daily."
    PARAMS = (
        Param("fast_days", 10, 2, 60, 1, "Shortest momentum horizon (days)"),
        Param("mid_days", 30, 5, 180, 1, "Middle momentum horizon (days)"),
        Param("slow_days", 90, 20, 365, 5, "Longest momentum horizon (days)"),
        Param("vol_days", 30, 5, 120, 1, "Realised-volatility window (days)"),
        Param("target_vol", 0.60, 0.10, 2.0, 0.05, "Annualised volatility at full conviction"),
        Param("rebalance_hours", 24, 1, 168, 1, "Decision clock (hours, UTC-aligned)", jitter=False),
        Param("allow_short", False, help="Short when trends point down (crypto drifts up: off by default)"),
    )

    def setup(self) -> None:
        p = self.p
        self.vol = LogReturnVol(self.days(p["vol_days"]), self.bar_minutes)
        self.slot_ns = int(p["rebalance_hours"] * 3600 * 1e9)
        self.slot = None
        self.closes: list[float] = []  # one per decision slot
        per_day = 24.0 / p["rebalance_hours"]
        self.lags = [
            max(1, int(round(p[k] * per_day))) for k in ("fast_days", "mid_days", "slow_days")
        ]

    def step(self, bar: BarData) -> float | None:
        self.vol.update(bar.close)
        slot = bar.ts // self.slot_ns
        if slot == self.slot:
            return None
        self.slot = slot
        self.closes.append(bar.close)
        need = max(self.lags) + 1
        if len(self.closes) > need:
            del self.closes[: len(self.closes) - need]
        if len(self.closes) < need or not self.vol.ready:
            return None
        now = self.closes[-1]
        votes = []
        for lag in self.lags:
            past = self.closes[-1 - lag]
            r = math.log(now / past) if past > 0 else 0.0
            vote = 1.0 if r > 0 else -1.0 if r < 0 else 0.0
            if not self.p["allow_short"]:
                vote = max(vote, 0.0)
            votes.append(vote)
        conviction = sum(votes) / len(votes)
        vol = self.vol.value
        scale = 1.0 if not vol or math.isnan(vol) else min(1.0, self.p["target_vol"] / vol)
        return conviction * scale
