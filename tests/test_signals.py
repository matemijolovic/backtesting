"""Signals in isolation: they run, stay in bounds, and the time-based ones
make the same decisions on different bar sizes."""

import pandas as pd
import pytest

from backtester.data import interval_minutes
from backtester.data import resample
from backtester.data import with_close_index
from strategies.catalog import SIGNALS
from strategies.signal_base import BarData


def feed(cls, bars: pd.DataFrame, interval: str, params=None):
    sig = cls(interval_minutes(interval), params or {})
    targets = []
    for ts, row in bars.iterrows():
        bar = BarData(int(ts.value), row.open, row.high, row.low, row.close, row.volume)
        t = sig.on_bar(bar)
        targets.append((ts, t, sig.position))
    return sig, pd.DataFrame(targets, columns=["ts", "target", "position"]).set_index("ts")


@pytest.mark.parametrize("key", sorted(SIGNALS))
@pytest.mark.parametrize("interval", ["1h", "4h"])
def test_signal_runs_and_bounded(key, interval, bars_1m):
    bars = with_close_index(resample(bars_1m, interval), interval)
    _, out = feed(SIGNALS[key], bars, interval, {"allow_short": True})
    pos = out["position"]
    assert pos.between(-1, 1).all()
    assert (pos != 0).any(), f"{key} never traded on {interval}"


def test_unknown_param_rejected():
    with pytest.raises(ValueError):
        SIGNALS["breakout"](60, {"nope": 1})


def test_trend_ensemble_is_timeframe_invariant(bars_1m):
    """Daily decisions from 15m and 4h bars agree (same closes at the boundary)."""
    cls = SIGNALS["trend_ensemble"]
    daily = {}
    for interval in ("15m", "4h"):
        bars = with_close_index(resample(bars_1m, interval), interval)
        _, out = feed(cls, bars, interval, {"slow_days": 40, "mid_days": 20, "vol_days": 10})
        decided = out["target"].dropna()
        daily[interval] = decided.groupby(decided.index.floor("D")).last()
    a, b = daily["15m"].align(daily["4h"], join="inner")
    assert len(a) > 30
    # Conviction (sign agreement) is identical; size differs only via vol estimate.
    assert ((a > 0) == (b > 0)).mean() > 0.97
    assert (a - b).abs().median() < 0.1


def test_breakout_channels_timeframe_invariant(bars_1m):
    cls = SIGNALS["breakout"]
    entries = {}
    for interval in ("15m", "1h"):
        bars = with_close_index(resample(bars_1m, interval), interval)
        _, out = feed(cls, bars, interval, {"entry_days": 10, "exit_days": 5, "vol_days": 10})
        opened = out[(out["target"] > 0)].index
        entries[interval] = set(opened.floor("D"))
    overlap = len(entries["15m"] & entries["1h"]) / max(len(entries["1h"]), 1)
    assert overlap > 0.6


def test_self_stops_flatten(bars_1m):
    cls = SIGNALS["breakout"]
    sig = cls(60, {"stop_vol": 0.5})
    sig.self_stops = True
    bars = with_close_index(resample(bars_1m, "1h"), "1h")
    stopped = 0
    for ts, row in bars.iterrows():
        before = sig.position
        sig.on_bar(BarData(int(ts.value), row.open, row.high, row.low, row.close, row.volume))
        if before > 0 and sig.position == 0:
            stopped += 1
    assert stopped > 0
