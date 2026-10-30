"""Supertrend with volatility-scaled bands.

Classic Supertrend puts bands ``mult x ATR`` around the bar midpoint and
ratchets them. ATR of a 5m bar is far smaller than ATR of a 4h bar, so
the classic version is a different strategy on each timeframe. Here the
band is ``mult`` *daily* standard deviations, measured from bar returns
and rescaled, so the flip distance in price is roughly the same at any
bar size.

Always in the market (long/short) unless ``allow_short`` is off, in
which case the short side is flat. Size is vol-targeted at each flip.
"""

from __future__ import annotations

import math

from strategies.indicators import LogReturnVol
from strategies.signal_base import BarData
from strategies.signal_base import Param
from strategies.signal_base import Signal


class Supertrend(Signal):
    key = "supertrend"
    title = "Supertrend (vol-scaled bands)"
    summary = "Ratcheting bands 2.5 daily sigmas from price; flip on close through the band."
    PARAMS = (
        Param("mult", 2.5, 0.5, 8.0, 0.25, "Band distance in daily sigmas"),
        Param("vol_days", 20, 5, 120, 1, "Volatility window (days)"),
        Param("target_vol", 0.60, 0.10, 2.0, 0.05, "Annualised volatility target per position"),
        Param("allow_short", False, help="Short on down-flips instead of going flat"),
    )

    def setup(self) -> None:
        self.vol = LogReturnVol(self.days(self.p["vol_days"]), self.bar_minutes)
        self.upper = math.nan
        self.lower = math.nan
        self.trend = 0  # +1 up, -1 down
        self.prev_close = math.nan

    def step(self, bar: BarData) -> float | None:
        self.vol.update(bar.close)
        if not self.vol.ready or not self.vol.value > 0:
            self.prev_close = bar.close
            return None
        mid = 0.5 * (bar.high + bar.low)
        band = self.p["mult"] * bar.close * self.vol.value / math.sqrt(365.0)
        up_raw, lo_raw = mid + band, mid - band

        if math.isnan(self.upper):
            self.upper, self.lower = up_raw, lo_raw
            self.trend = 1 if bar.close >= mid else -1
            self.prev_close = bar.close
            return None

        # Ratchet: bands only tighten while price stays inside them.
        upper = up_raw if (up_raw < self.upper or self.prev_close > self.upper) else self.upper
        lower = lo_raw if (lo_raw > self.lower or self.prev_close < self.lower) else self.lower
        self.upper, self.lower = upper, lower
        self.prev_close = bar.close

        flipped = False
        if self.trend <= 0 and bar.close > upper:
            self.trend, flipped = 1, True
        elif self.trend >= 0 and bar.close < lower:
            self.trend, flipped = -1, True

        size = min(1.0, self.p["target_vol"] / self.vol.value)
        if self.trend > 0:
            self.stop_px = lower
            if flipped or self.position <= 0:
                return size
        else:
            if not self.p["allow_short"]:
                self.stop_px = None
                return 0.0 if self.position != 0 else None
            self.stop_px = upper
            if flipped or self.position >= 0:
                return -size
        return None

    def on_stopped(self) -> None:
        # The band *is* the stop: a stop fill means the trend flipped.
        super().on_stopped()
        self.trend = -1 if self.trend > 0 else 1
