"""Robustness sweeps: one strategy across timeframes, bar offsets,
symbols and perturbed parameters, run in parallel.

An edge that only exists on 4h bars starting at 00:00, on one symbol,
with one parameter set, is a fitted artefact. A sweep shows how wide the
plateau is:

- intervals:   5m ... 1d, including odd sizes like 90m or 3h
- offsets:     same bar size with shifted boundaries (4h bars from 01:00)
- symbols:     the same rules on other assets
- jitter:      every numeric parameter randomly moved +/- N%

``summarize`` reduces the grid to a handful of numbers, and
``combine_equity`` builds an equal-weight multi-symbol portfolio.
"""

from __future__ import annotations

import math
import os
import random
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace
from typing import Any
from typing import Callable

import numpy as np
import pandas as pd

from backtester import metrics as M
from backtester import registry
from backtester.data import load_1m
from backtester.engine import RunConfig
from backtester.engine import _window
from backtester.engine import run_backtest


@dataclass
class SweepRun:
    config: RunConfig
    params: dict[str, Any]
    row: dict[str, Any]
    equity: pd.Series  # daily
    benchmark: pd.Series  # daily
    yearly: pd.Series
    error: str | None = None


def jitter_params(key: str, base: dict[str, Any] | None, n: int, pct: float, seed: int = 7) -> list[dict[str, Any]]:
    """``n`` random variants of ``base``, each numeric param moved +/- ``pct``."""
    spec = registry.get(key)
    base = {**spec.defaults(), **(base or {})}
    rng = random.Random(seed)
    variants = []
    for _ in range(n):
        out = dict(base)
        for p in spec.params:
            if not p.jitter or p.kind not in ("int", "float"):
                continue
            value = float(base[p.name])
            if value == 0:
                continue
            new = value * (1.0 + rng.uniform(-pct, pct))
            if p.lo is not None:
                new = max(new, p.lo)
            if p.hi is not None:
                new = min(new, p.hi)
            out[p.name] = int(round(new)) if p.kind == "int" else round(new, 6)
        variants.append(out)
    return variants


def build_grid(
    base: RunConfig,
    *,
    symbols: list[str] | None = None,
    intervals: list[str] | None = None,
    offsets: list[str | None] | None = None,
    param_sets: list[dict[str, Any]] | None = None,
) -> list[RunConfig]:
    grid = []
    for symbol in symbols or [base.symbol]:
        for interval in intervals or [base.interval]:
            for offset in offsets or [base.offset]:
                for params in param_sets or [base.params]:
                    grid.append(replace(base, symbol=symbol, interval=interval, offset=offset or None, params=dict(params)))
    return grid


def _prefetch(configs: list[RunConfig]) -> None:
    """Download everything once in this process so workers only read cache."""
    seen: dict[tuple[str, str], tuple[pd.Timestamp, pd.Timestamp]] = {}
    for cfg in configs:
        spec = registry.get(cfg.strategy)
        params = spec.resolve(cfg.params, cfg.interval, cfg.scale_lookbacks)
        start, end = _window(cfg)
        lo = start - pd.Timedelta(days=spec.warmup_days(params, cfg.interval))
        markets = ("spot", "perp") if spec.kind == "carry" else ("spot",)
        for market in markets:
            key = (cfg.symbol, market)
            old = seen.get(key)
            seen[key] = (min(lo, old[0]), max(end, old[1])) if old else (lo, end)
    for (symbol, market), (lo, hi) in seen.items():
        load_1m(symbol, lo, hi, market=market)


def _run_one(cfg: RunConfig) -> SweepRun:
    try:
        result = run_backtest(cfg)
    except Exception as exc:  # one bad cell must not kill the sweep
        return SweepRun(cfg, cfg.params, {"error": str(exc)}, pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), error=repr(exc))
    yearly = M.period_returns(result.equity, "YE")
    yearly.index = yearly.index.year
    return SweepRun(
        config=cfg,
        params=result.params,
        row=result.row(),
        equity=M.daily(result.equity),
        benchmark=M.daily(result.benchmark),
        yearly=yearly,
    )


@contextmanager
def _neutral_main():
    """Hide ``__main__`` while worker processes start.

    Spawned workers (macOS/Windows default) re-import the parent's main
    module. Under ``streamlit run`` or an unguarded script that is the code
    that started the sweep, so every worker would start it again.
    Workers only need ``_run_one``, which lives in this module.
    """
    main = sys.modules.get("__main__")
    sys.modules["__main__"] = types.ModuleType("__main__")
    try:
        yield
    finally:
        if main is not None:
            sys.modules["__main__"] = main


def run_sweep(
    configs: list[RunConfig],
    *,
    workers: int | None = None,
    progress: Callable[[int, int, SweepRun], None] | None = None,
) -> list[SweepRun]:
    """Run every config, in parallel processes. Order of results = order of configs."""
    if not configs:
        return []
    _prefetch(configs)
    workers = workers or max(1, min(len(configs), (os.cpu_count() or 2) - 1))
    results: list[SweepRun | None] = [None] * len(configs)
    if workers == 1:
        for i, cfg in enumerate(configs):
            results[i] = _run_one(cfg)
            if progress:
                progress(i + 1, len(configs), results[i])
        return results  # type: ignore[return-value]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        with _neutral_main():
            futures = {pool.submit(_run_one, cfg): i for i, cfg in enumerate(configs)}
        for done, fut in enumerate(as_completed(futures), start=1):
            i = futures[fut]
            results[i] = fut.result()
            if progress:
                progress(done, len(configs), results[i])
    return results  # type: ignore[return-value]


def table(runs: list[SweepRun]) -> pd.DataFrame:
    rows = []
    for i, run in enumerate(runs):
        row = {"run": i, **run.row}
        for k, v in run.config.params.items():
            row[f"p.{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(runs: list[SweepRun]) -> dict[str, float]:
    """How consistent is the edge across the grid?"""
    ok = [r for r in runs if r.error is None and r.row.get("sharpe") is not None]
    if not ok:
        return {"runs": len(runs), "failed": len(runs)}
    sharpe = np.array([r.row["sharpe"] for r in ok], dtype=float)
    cagr = np.array([r.row["cagr"] for r in ok], dtype=float)
    mdd = np.array([r.row["max_drawdown"] for r in ok], dtype=float)
    bh_sharpe = np.array([r.row.get("bh_sharpe") or np.nan for r in ok], dtype=float)
    bh_mdd = np.array([r.row.get("bh_max_drawdown") or np.nan for r in ok], dtype=float)
    years = pd.concat([r.yearly for r in ok], axis=1) if ok else pd.DataFrame()
    return {
        "runs": len(runs),
        "failed": len(runs) - len(ok),
        "profitable": float(np.mean(cagr > 0)),
        "beats_bh_sharpe": float(np.nanmean(sharpe > bh_sharpe)),
        "shallower_dd_than_bh": float(np.nanmean(mdd > bh_mdd)),
        "sharpe_median": float(np.nanmedian(sharpe)),
        "sharpe_p10": float(np.nanpercentile(sharpe, 10)),
        "sharpe_min": float(np.nanmin(sharpe)),
        "sharpe_std": float(np.nanstd(sharpe)),
        "cagr_median": float(np.nanmedian(cagr)),
        "max_dd_median": float(np.nanmedian(mdd)),
        "max_dd_worst": float(np.nanmin(mdd)),
        "positive_years": float((years.median(axis=1) > 0).mean()) if not years.empty else math.nan,
    }


def combine_equity(curves: list[pd.Series], capital: float = 100_000.0) -> pd.Series:
    """Equal-weight portfolio of daily equity curves (each sleeve gets capital/N, never rebalanced)."""
    curves = [c for c in curves if c is not None and len(c) > 1]
    if not curves:
        return pd.Series(dtype=float)
    norm = pd.concat([c / c.iloc[0] for c in curves], axis=1).ffill().bfill()
    return capital * norm.mean(axis=1)
