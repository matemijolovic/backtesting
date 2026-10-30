"""Blend of the trend, breakout, supertrend (and optionally mean-reversion) signals.

Each child runs with its default parameters and manages its own stops on
bar highs/lows. The ensemble's target is the weighted sum of the
children's current positions. Trend and dip-buying make money in
different conditions, so the blended equity curve is smoother than any
one leg. Smoother also means it can carry more size for the same
drawdown.

Mean reversion defaults to weight 0: in 2021-24 sweeps across four
symbols and three timeframes it had no edge after 12 bps per fill.
"""

from __future__ import annotations

from strategies.breakout import Breakout
from strategies.mean_reversion import MeanReversion
from strategies.signal_base import BarData
from strategies.signal_base import Param
from strategies.signal_base import Signal
from strategies.supertrend import Supertrend
from strategies.trend_ensemble import TrendEnsemble

_LEGS = (
    ("w_trend", TrendEnsemble),
    ("w_breakout", Breakout),
    ("w_supertrend", Supertrend),
    ("w_mean_reversion", MeanReversion),
)


class Ensemble(Signal):
    key = "ensemble"
    title = "Ensemble (trend + breakout + supertrend)"
    summary = "Weighted blend of the trend signals (dip-buying optional). Weights are normalised to sum to 1."
    PARAMS = (
        Param("w_trend", 0.50, 0.0, 1.0, 0.05, "Weight: trend ensemble"),
        Param("w_breakout", 0.25, 0.0, 1.0, 0.05, "Weight: channel breakout"),
        Param("w_supertrend", 0.25, 0.0, 1.0, 0.05, "Weight: supertrend"),
        # 0 by default: no edge net of fees in 2021-24 robustness sweeps.
        Param("w_mean_reversion", 0.0, 0.0, 1.0, 0.05, "Weight: mean reversion"),
        Param("allow_short", False, help="Let every leg trade the short side"),
    )

    def setup(self) -> None:
        total = sum(max(self.p[w], 0.0) for w, _ in _LEGS) or 1.0
        self.legs: list[tuple[float, Signal]] = []
        for weight_key, cls in _LEGS:
            weight = max(self.p[weight_key], 0.0) / total
            if weight <= 0:
                continue
            leg = cls(self.bar_minutes, {"allow_short": self.p["allow_short"]})
            leg.self_stops = True
            self.legs.append((weight, leg))
        self.last = None

    def step(self, bar: BarData) -> float | None:
        for _, leg in self.legs:
            leg.on_bar(bar)
        blended = sum(weight * leg.position for weight, leg in self.legs)
        if self.last is not None and abs(blended - self.last) < 1e-9:
            return None
        self.last = blended
        return blended

    def on_stopped(self) -> None:
        super().on_stopped()
        self.last = None
