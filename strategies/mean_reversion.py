"""Buy-the-dip mean reversion inside an uptrend.

When price is stretched ``entry_z`` standard deviations below its
``lookback_hours`` mean *and* above its long-term trend average, buy.
Exit at the mean (z >= ``exit_z``), after ``max_hold_hours``, or at a
volatility stop. With ``allow_short`` it also fades rips in downtrends.

Panic selling inside an established trend tends to overshoot, and the
trend filter keeps it from catching falling knives in a bear market. It
makes money in different conditions from the trend strategies, which is
the point of pairing them.
"""

from __future__ import annotations

import math

from strategies.indicators import SMA
from strategies.indicators import LogReturnVol
from strategies.indicators import RollingStd
from strategies.signal_base import BarData
from strategies.signal_base import Param
from strategies.signal_base import Signal


class MeanReversion(Signal):
    key = "mean_reversion"
    title = "Mean reversion (dip buying in trend)"
    summary = "Buy 2-sigma dips below the 2-day mean while above the 50-day trend; exit at the mean."
    PARAMS = (
        Param("lookback_hours", 48, 6, 480, 6, "Window for the mean and z-score (hours)"),
        Param("entry_z", 2.0, 0.5, 4.0, 0.1, "Enter when z-score is beyond this"),
        Param("exit_z", 0.0, -1.0, 2.0, 0.1, "Exit when z-score recovers to this"),
        Param("trend_days", 50, 0, 200, 5, "Trend filter SMA (days); 0 disables"),
        Param("max_hold_hours", 96, 6, 720, 6, "Time stop (hours)"),
        Param("stop_vol", 2.5, 0.5, 10.0, 0.25, "Stop distance in daily sigmas"),
        Param("vol_days", 20, 5, 120, 1, "Volatility window (days)"),
        Param("size", 1.0, 0.1, 1.0, 0.05, "Exposure per trade (fraction of max)"),
        Param("allow_short", False, help="Also fade rallies when below the trend"),
    )

    def setup(self) -> None:
        p = self.p
        n = self.bars(p["lookback_hours"])
        self.mean = SMA(n)
        self.std = RollingStd(n)
        self.trend = SMA(self.days(p["trend_days"])) if p["trend_days"] > 0 else None
        self.vol = LogReturnVol(self.days(p["vol_days"]), self.bar_minutes)
        self.max_hold = self.bars(p["max_hold_hours"])
        self.held = 0

    def step(self, bar: BarData) -> float | None:
        close = bar.close
        self.mean.update(close)
        self.std.update(close)
        self.vol.update(close)
        if self.trend is not None:
            self.trend.update(close)
        ready = self.mean.ready and self.std.ready and self.vol.ready
        if self.trend is not None:
            ready = ready and self.trend.ready
        if not ready or self.std.value <= 0:
            return None

        z = (close - self.mean.value) / self.std.value
        p = self.p

        if self.position != 0.0:
            self.held += 1
            done = self.held >= self.max_hold
            if self.position > 0 and (z >= p["exit_z"] or done):
                return 0.0
            if self.position < 0 and (z <= -p["exit_z"] or done):
                return 0.0
            return None

        up = self.trend is None or close > self.trend.value
        down = self.trend is None or close < self.trend.value
        dist = p["stop_vol"] * close * self.vol.value / math.sqrt(365.0)
        if z <= -p["entry_z"] and up:
            self.held = 0
            self.stop_px = close - dist
            return p["size"]
        if p["allow_short"] and z >= p["entry_z"] and down:
            self.held = 0
            self.stop_px = close + dist
            return -p["size"]
        return None
