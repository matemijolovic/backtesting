import math

import numpy as np
import pandas as pd
import pytest

from backtester import metrics as M
from strategies.indicators import ATR
from strategies.indicators import EMA
from strategies.indicators import KAMA
from strategies.indicators import RSI
from strategies.indicators import SMA
from strategies.indicators import RollingExtreme
from strategies.indicators import RollingStd

RNG = np.random.default_rng(1)
X = list(100 + np.cumsum(RNG.normal(0, 1, 500)))


def test_sma_std_extremes_match_pandas():
    s = pd.Series(X)
    sma, std, hi, lo = SMA(20), RollingStd(20), RollingExtreme(20, "max"), RollingExtreme(20, "min")
    out = [(sma.update(x), std.update(x), hi.update(x), lo.update(x)) for x in X]
    got = pd.DataFrame(out, columns=["sma", "std", "hi", "lo"])
    np.testing.assert_allclose(got["sma"][19:], s.rolling(20).mean()[19:], rtol=1e-9)
    np.testing.assert_allclose(got["std"][19:], s.rolling(20).std()[19:], rtol=1e-6)
    np.testing.assert_allclose(got["hi"][19:], s.rolling(20).max()[19:])
    np.testing.assert_allclose(got["lo"][19:], s.rolling(20).min()[19:])


def test_ema_matches_pandas():
    ema = EMA(10)
    got = [ema.update(x) for x in X]
    want = pd.Series(X).ewm(span=10, adjust=False).mean()
    np.testing.assert_allclose(got, want, rtol=1e-9)


def test_rsi_bounds_and_direction():
    rsi = RSI(14)
    for x in range(1, 50):
        rsi.update(float(x))
    assert rsi.value == pytest.approx(100.0)
    for x in range(50, 0, -1):
        rsi.update(float(x))
    assert rsi.value < 5


def test_atr_and_kama_sane():
    atr = ATR(14)
    for x in X:
        atr.update(x + 1, x - 1, x)
    assert 2.0 <= atr.value < 5.0
    kama = KAMA(10)
    for x in np.linspace(1, 100, 200):
        kama.update(float(x))
    assert kama.er == pytest.approx(1.0)


def test_summary_on_known_curve():
    idx = pd.date_range("2024-01-01", periods=366, freq="1D", tz="UTC")
    eq = pd.Series(100 * 1.001 ** np.arange(366), index=idx)
    eq.iloc[100:110] *= 0.9  # a 10% dip that recovers
    s = M.summary(eq)
    assert s["total_return"] == pytest.approx(1.001**365 - 1, rel=1e-6)
    assert s["max_drawdown"] == pytest.approx(-0.1, abs=0.002)
    assert s["sharpe"] > 1


def test_drawdown_duration_and_period_returns():
    idx = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
    eq = pd.Series([100, 110, 99, 100, 105, 111, 111, 120, 90, 95], index=idx, dtype=float)
    s = M.summary(eq)
    assert s["max_drawdown"] == pytest.approx(90 / 120 - 1)
    assert s["max_dd_days"] == pytest.approx(4.0)  # 110 on day 1 -> new high on day 5
    yearly = M.period_returns(eq, "YE")
    assert yearly.iloc[0] == pytest.approx(95 / 100 - 1)


def test_trade_stats_parses_money_strings():
    pos = pd.DataFrame({"realized_pnl": ["10.5 USDT", "-5.0 USDT", "20 USDT"]})
    stats = M.trade_stats(pos)
    assert stats["trades"] == 3
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["profit_factor"] == pytest.approx(30.5 / 5)
    fills = pd.DataFrame({"commissions": [["1.5 USDT"], ["2.5 USDT"]]})
    assert M.fees_paid(fills) == pytest.approx(4.0)


def test_combine_equity_equal_weight():
    from backtester.robustness import combine_equity

    idx = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    a = pd.Series([1.0, 2.0, 2.0], index=idx)
    b = pd.Series([1.0, 1.0, 0.5], index=idx)
    out = combine_equity([a, b], capital=100)
    assert list(out.round(6)) == [100.0, 150.0, 125.0]
    assert math.isclose(out.iloc[-1], 125.0)
