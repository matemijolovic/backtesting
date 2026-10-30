"""Turtle-style channel breakout with a volatility trailing stop.

Long on a close above the highest high of the last ``entry_days``; exit
on a close below the lowest low of the last ``exit_days``, or when the
trailing stop is hit. The stop trails the best close since entry by
``stop_vol`` daily standard deviations. Size is set at entry from a
volatility target and held, so trades are not churned.

Channels are measured in days over bar highs and lows. The 20-day high is
the same price on 5m or 4h bars, so entries are timeframe-invariant.
Finer bars only move the exit timing.
"""

from __future__ import annotations

import math

from strategies.indicators import LogReturnVol
from strategies.indicators import RollingExtreme
from strategies.signal_base import BarData
from strategies.signal_base import Param
from strategies.signal_base import Signal


class Breakout(Signal):
    key = "breakout"
    title = "Channel breakout (Turtle-style)"
    summary = "Buy 20-day highs, exit on 10-day lows or a 3-sigma trailing stop. Vol-targeted size."
    PARAMS = (
        Param("entry_days", 20, 5, 120, 1, "Breakout channel (days)"),
        Param("exit_days", 10, 2, 60, 1, "Exit channel (days)"),
        Param("vol_days", 20, 5, 120, 1, "Volatility window (days)"),
        Param("stop_vol", 3.0, 0.5, 10.0, 0.25, "Trailing stop distance in daily sigmas"),
        Param("target_vol", 0.60, 0.10, 2.0, 0.05, "Annualised volatility target per trade"),
        Param("allow_short", False, help="Also trade downside breakouts"),
    )

    def setup(self) -> None:
        p = self.p
        self.entry_hi = RollingExtreme(self.days(p["entry_days"]), "max")
        self.entry_lo = RollingExtreme(self.days(p["entry_days"]), "min")
        self.exit_hi = RollingExtreme(self.days(p["exit_days"]), "max")
        self.exit_lo = RollingExtreme(self.days(p["exit_days"]), "min")
        self.vol = LogReturnVol(self.days(p["vol_days"]), self.bar_minutes)
        self.best = math.nan

    def step(self, bar: BarData) -> float | None:
        # Channels as of the previous bar (no peeking at this bar's extreme).
        ready = self.entry_hi.ready and self.exit_hi.ready and self.vol.ready
        up, dn = self.entry_hi.value, self.entry_lo.value
        ex_up, ex_dn = self.exit_hi.value, self.exit_lo.value
        self.entry_hi.update(bar.high)
        self.entry_lo.update(bar.low)
        self.exit_hi.update(bar.high)
        self.exit_lo.update(bar.low)
        self.vol.update(bar.close)
        if not ready:
            return None

        close = bar.close
        sigma_px = close * self.vol.value / math.sqrt(365.0)
        dist = self.p["stop_vol"] * sigma_px

        if self.position > 0:
            if close < ex_dn:
                return 0.0
            self.best = max(self.best, close)
            self.stop_px = max(self.stop_px or 0.0, self.best - dist)
            return None
        if self.position < 0:
            if close > ex_up:
                return 0.0
            self.best = min(self.best, close)
            self.stop_px = min(self.stop_px or math.inf, self.best + dist)
            return None

        size = min(1.0, self.p["target_vol"] / self.vol.value) if self.vol.value > 0 else 0.0
        if close > up:
            self.best = close
            self.stop_px = close - dist
            return size
        if self.p["allow_short"] and close < dn:
            self.best = close
            self.stop_px = close + dist
            return -size
        return None
