"""End-to-end through Nautilus on synthetic data (no network)."""

import numpy as np
import pandas as pd
import pytest

from backtester import registry
from backtester.data import resample
from backtester.data import with_close_index
from backtester.engine import RunConfig
from backtester.engine import run_backtest
from backtester.metrics import _money


def _bars(bars_1m, interval):
    return with_close_index(resample(bars_1m, interval), interval)


def _cfg(strategy, interval="1h", **kw):
    return RunConfig(strategy=strategy, symbol="TESTUSDT", interval=interval, start="2023-03-01", end="2023-05-01", **kw)


SINGLE = [k for k, s in registry.REGISTRY.items() if s.kind != "carry"]


@pytest.mark.parametrize("key", SINGLE)
def test_every_strategy_runs(key, bars_1m):
    params = {"allow_short": True} if "allow_short" in registry.get(key).defaults() else {}
    result = run_backtest(_cfg(key, params=params), bars=_bars(bars_1m, "1h"))
    assert len(result.equity) > 100
    assert result.equity.index[0] >= pd.Timestamp("2023-03-01", tz="UTC")
    assert result.equity.iloc[0] == pytest.approx(100_000)
    assert np.isfinite(result.metrics["max_drawdown"])
    assert result.benchmark.iloc[0] == pytest.approx(100_000)


def test_equity_matches_fill_accounting(bars_1m):
    """Rebuild equity from fills + closes independently and compare with the recorder."""
    bars = _bars(bars_1m, "1h")
    # Start at the first bar so no fills fall in a (filtered-out) warmup window.
    cfg = RunConfig(strategy="supertrend", symbol="TESTUSDT", interval="1h", start="2023-01-01", end="2023-05-01",
                    params={"allow_short": True})
    result = raw = run_backtest(cfg, bars=bars)
    assert len(raw.fills) > 10
    sign = np.where(raw.fills["side"] == "BUY", 1.0, -1.0)
    qty = raw.fills["filled_qty"].astype(float).to_numpy() * sign
    px = raw.fills["avg_px"].astype(float).to_numpy()
    fee = np.array([sum(_money(c) for c in cs) for cs in raw.fills["commissions"]])
    ts = pd.to_datetime(raw.fills["ts_last"], utc=True)
    flow = pd.DataFrame({"qty": qty, "cash": -qty * px - fee}, index=ts).groupby(level=0).sum()
    close = bars["close"]
    flow = flow.reindex(close.index, fill_value=0.0)
    pos = flow["qty"].cumsum()
    cash = 100_000 + flow["cash"].cumsum()
    rebuilt = (cash + pos * close)[result.equity.index]
    rebuilt = rebuilt * (100_000 / rebuilt.iloc[0])
    err = (rebuilt - result.equity).abs() / result.equity
    assert err.max() < 2e-3, err.max()


def test_fees_reduce_returns(bars_1m):
    bars = _bars(bars_1m, "1h")
    cheap = run_backtest(_cfg("supertrend", fee_bps=0, slippage_bps=0), bars=bars)
    dear = run_backtest(_cfg("supertrend", fee_bps=30, slippage_bps=10), bars=bars)
    assert cheap.metrics["trades"] > 0
    assert dear.metrics["final_equity"] < cheap.metrics["final_equity"]
    assert dear.metrics["fees"] > cheap.metrics["fees"]


def test_stops_are_real_orders(bars_1m):
    result = run_backtest(_cfg("breakout", params={"stop_vol": 0.75}), bars=_bars(bars_1m, "1h"))
    assert result.extras["n_stops"] > 0
    assert "STOP_MARKET" in set(result.fills["type"].astype(str))


def test_legacy_lookbacks_scale_with_interval():
    spec = registry.get("dual_ma")
    assert spec.resolve({}, "1d", True) == {"fast": 20, "slow": 50}
    assert spec.resolve({}, "1h", True) == {"fast": 480, "slow": 1200}
    assert spec.resolve({}, "1h", False) == {"fast": 20, "slow": 50}
    with pytest.raises(ValueError):
        spec.resolve({"bogus": 1}, "1h", True)


def test_leverage_scales_exposure(bars_1m):
    bars = _bars(bars_1m, "1h")
    one = run_backtest(_cfg("supertrend", leverage=1.0, params={"target_vol": 2.0}), bars=bars)
    two = run_backtest(_cfg("supertrend", leverage=2.0, params={"target_vol": 2.0}), bars=bars)
    assert two.exposure.abs().max() > 1.5 * one.exposure.abs().max() * 0.9
    assert one.exposure.abs().max() <= 1.05
