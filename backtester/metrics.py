"""Performance metrics from an equity curve.

Return/risk ratios use *daily* equity (last value per UTC day), so a 5m
run and a 1d run of the same strategy are directly comparable.
Annualisation uses 365 days (crypto trades every day). Drawdown uses the
full-resolution curve, so intraday dips count.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.0


def daily(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return equity
    return equity.resample("1D").last().ffill().dropna()


def drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def _max_dd_duration(equity: pd.Series) -> pd.Timedelta:
    if equity.empty:
        return pd.Timedelta(0)
    peak = equity.cummax()
    under = equity < peak
    if not under.any():
        return pd.Timedelta(0)
    # Each new high starts a new group; time spent under water per group.
    group = (~under).cumsum()
    idx = equity.index.to_series()
    spans = idx[under].groupby(group[under]).agg(["first", "last"])
    # Recovery happens at the next index after the last underwater point.
    longest = pd.Timedelta(0)
    for first, last in spans.itertuples(index=False):
        pos = equity.index.get_loc(last)
        end = equity.index[pos + 1] if pos + 1 < len(equity) else last
        start_pos = equity.index.get_loc(first)
        start = equity.index[start_pos - 1] if start_pos > 0 else first
        longest = max(longest, end - start)
    return longest


def summary(equity: pd.Series) -> dict:
    """Headline stats for an equity curve (any sampling)."""
    equity = equity.dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return {}
    days_curve = daily(equity)
    rets = days_curve.pct_change().dropna()
    span_days = max((equity.index[-1] - equity.index[0]) / pd.Timedelta(days=1), 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (DAYS_PER_YEAR / span_days) - 1.0 if equity.iloc[-1] > 0 else -1.0
    vol = rets.std(ddof=1) * math.sqrt(DAYS_PER_YEAR) if len(rets) > 1 else math.nan
    mean = rets.mean() * DAYS_PER_YEAR if len(rets) else math.nan
    sharpe = mean / vol if vol and vol > 0 else math.nan
    downside = rets[rets < 0]
    dvol = math.sqrt((downside**2).sum() / max(len(rets), 1)) * math.sqrt(DAYS_PER_YEAR)
    sortino = mean / dvol if dvol > 0 else math.nan
    dd = drawdown(equity)
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else math.nan
    best = rets.max() if len(rets) else math.nan
    worst = rets.min() if len(rets) else math.nan
    return {
        "total_return": float(total),
        "cagr": float(cagr),
        "volatility": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "calmar": float(calmar),
        "max_dd_days": _max_dd_duration(days_curve) / pd.Timedelta(days=1),
        "best_day": float(best),
        "worst_day": float(worst),
        "days": float(span_days),
    }


def trade_stats(positions: pd.DataFrame) -> dict:
    """Stats over closed round trips from a Nautilus positions report."""
    if positions is None or positions.empty or "realized_pnl" not in positions:
        return {"trades": 0}
    pnl = positions["realized_pnl"].map(_money).astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_loss = -losses.sum()
    out = {
        "trades": int(len(pnl)),
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else math.nan,
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else math.inf,
        "avg_trade": float(pnl.mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
    }
    if {"ts_opened", "ts_closed"} <= set(positions.columns):
        held = pd.to_datetime(positions["ts_closed"], utc=True) - pd.to_datetime(positions["ts_opened"], utc=True)
        out["avg_hold_hours"] = float(held.mean() / pd.Timedelta(hours=1)) if len(held) else math.nan
    return out


def fees_paid(fills: pd.DataFrame) -> float:
    if fills is None or fills.empty or "commissions" not in fills:
        return 0.0
    total = 0.0
    for item in fills["commissions"]:
        if isinstance(item, (list, tuple)):
            total += sum(_money(x) for x in item)
        else:
            total += _money(item)
    return total


def _money(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if hasattr(value, "as_double"):
        return value.as_double()
    text = str(value).strip().strip("[]'\"")
    if not text:
        return 0.0
    return float(text.split()[0].replace(",", ""))


def period_returns(equity: pd.Series, freq: str = "YE") -> pd.Series:
    """Compounded return per calendar period ('YE' yearly, 'ME' monthly)."""
    if equity.empty:
        return equity
    last = equity.resample(freq).last()
    first_val = equity.iloc[0]
    prev = last.shift(1)
    prev.iloc[0] = first_val
    return (last / prev - 1.0).dropna()


def monthly_table(equity: pd.Series) -> pd.DataFrame:
    """Year x month grid of returns plus a full-year column."""
    monthly = period_returns(equity, "ME")
    if monthly.empty:
        return pd.DataFrame()
    frame = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values})
    grid = frame.pivot(index="year", columns="month", values="ret")
    grid.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in grid.columns]
    yearly = period_returns(equity, "YE")
    grid["Year"] = pd.Series(yearly.values, index=yearly.index.year)
    return grid


def exposure_stats(exposure: pd.Series) -> dict:
    if exposure is None or exposure.empty:
        return {}
    return {
        "time_in_market": float((exposure.abs() > 1e-6).mean()),
        "avg_gross_exposure": float(exposure.abs().mean()),
    }
