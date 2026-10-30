"""Command line.

    uv run python -m backtester list
    uv run python -m backtester download BTCUSDT ETHUSDT --start 2021-01-01
    uv run python -m backtester run trend_ensemble --symbol ETHUSDT --interval 4h --start 2022-01-01
    uv run python -m backtester run breakout -p entry_days=55 -p exit_days=20
    uv run python -m backtester sweep ensemble --symbols BTCUSDT,ETHUSDT --intervals 15m,1h,4h,1d \\
        --offsets 0,30m --jitter 8 --jitter-pct 0.3 --out sweep.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import pandas as pd

from backtester import registry
from backtester.data import COMMON_INTERVALS
from backtester.data import load_1m
from backtester.data import quality


def _parse_params(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--param expects name=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            out[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            out[key.strip()] = raw
    return out


def _fmt_metrics(metrics: dict[str, Any]) -> str:
    pct = {"total_return", "cagr", "volatility", "max_drawdown", "best_day", "worst_day", "win_rate", "time_in_market"}
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if key in pct:
                text = f"{value:+.2%}"
            elif value != 0 and abs(value) < 0.01:
                text = f"{value:.6f}"
            else:
                text = f"{value:,.2f}"
        else:
            text = str(value)
        lines.append(f"  {key:<20} {text}")
    return "\n".join(lines)


def cmd_list(_args) -> int:
    for spec in registry.REGISTRY.values():
        print(f"{spec.key:<16} [{spec.kind}] {spec.title}")
        print(f"{'':<16} {spec.summary}")
        for p in spec.params:
            print(f"{'':<18}{p.name}={p.default!r}  {p.help}")
        print()
    print("Intervals: any <n>m / <n>h / <n>d, e.g.", ", ".join(COMMON_INTERVALS))
    return 0


def cmd_download(args) -> int:
    for symbol in args.symbols:
        for market in args.markets.split(","):
            def progress(done, total, msg):
                print(f"\r  {msg}  [{done}/{total}]", end="", flush=True)

            frame = load_1m(symbol, args.start, args.end or pd.Timestamp.now(tz="UTC"), market=market, progress=progress)
            print()
            q = quality(frame)
            print(f"{symbol} {market}: {q.bars:,} bars, coverage {q.coverage:.3%}, largest gap {q.largest_gap}")
    return 0


def _run_config(args):
    from backtester.engine import RunConfig

    return RunConfig(
        strategy=args.strategy,
        symbol=args.symbol.upper(),
        interval=args.interval,
        start=args.start,
        end=args.end,
        params=_parse_params(args.param),
        offset=args.offset,
        capital=args.capital,
        leverage=args.leverage,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        perp_fee_bps=args.perp_fee_bps,
        scale_lookbacks=not args.no_scale,
    )


def cmd_run(args) -> int:
    from backtester.engine import run_backtest

    cfg = _run_config(args)
    print(f"Running {cfg.label()} {cfg.start} -> {cfg.end or 'now'} ...", flush=True)
    result = run_backtest(cfg)
    print(f"\nParams: {result.params}")
    print(f"\n=== {cfg.strategy} ===")
    print(_fmt_metrics(result.metrics))
    print("\n=== Buy & hold ===")
    print(_fmt_metrics({k: result.benchmark_metrics[k] for k in ("cagr", "sharpe", "max_drawdown") if k in result.benchmark_metrics}))
    extras = {k: v for k, v in result.extras.items() if v is not None}
    if extras:
        print("\n=== Extras ===")
        print(_fmt_metrics(extras))
    from backtester.metrics import monthly_table

    grid = monthly_table(result.equity)
    if not grid.empty:
        print("\n=== Returns by month ===")
        print((grid * 100).round(1).to_string())
    print(f"\n{result.extras.get('bars', 0):,} bars in {result.elapsed:.1f}s")
    if args.out:
        result.equity.to_csv(args.out, header=True)
        print(f"Equity curve -> {args.out}")
    return 0


def cmd_sweep(args) -> int:
    from backtester.robustness import build_grid
    from backtester.robustness import jitter_params
    from backtester.robustness import run_sweep
    from backtester.robustness import summarize
    from backtester.robustness import table

    base = _run_config(args)
    symbols = [s.strip().upper() for s in (args.symbols or base.symbol).split(",")]
    intervals = [s.strip() for s in args.intervals.split(",")]
    offsets = [None if o.strip() in ("", "0") else o.strip() for o in args.offsets.split(",")]
    param_sets = [base.params]
    if args.jitter:
        param_sets += jitter_params(base.strategy, base.params, args.jitter, args.jitter_pct)
    grid = build_grid(base, symbols=symbols, intervals=intervals, offsets=offsets, param_sets=param_sets)
    print(f"Sweeping {len(grid)} runs ...", flush=True)

    def progress(done, total, run):
        status = "ERR" if run.error else f"sharpe {run.row.get('sharpe', float('nan')):+.2f}"
        print(f"  [{done}/{total}] {run.config.label():<40} {status}", flush=True)

    runs = run_sweep(grid, workers=args.workers, progress=progress)
    frame = table(runs)
    pd.set_option("display.width", 200)
    if len(intervals) > 1 or len(symbols) > 1:
        print("\nMedian Sharpe (symbol x interval):")
        print(frame.pivot_table(index="symbol", columns="interval", values="sharpe", aggfunc="median").round(2).to_string())
    print("\nSummary:")
    print(_fmt_metrics(summarize(runs)))
    if args.out:
        frame.to_csv(args.out, index=False)
        print(f"\nAll runs -> {args.out}")
    return 0


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("strategy", choices=list(registry.REGISTRY))
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--offset", default=None, help="Shift bar boundaries, e.g. 30m")
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--perp-fee-bps", type=float, default=5.0, help="Perp leg fee (funding_carry only)")
    p.add_argument("--no-scale", action="store_true", help="Do not rescale legacy bar lookbacks to the interval")
    p.add_argument("-p", "--param", action="append", help="Strategy param name=value (repeatable)")
    p.add_argument("--out", default=None, help="CSV output path")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="backtester", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List strategies and parameters").set_defaults(fn=cmd_list)

    d = sub.add_parser("download", help="Fetch and cache 1m data")
    d.add_argument("symbols", nargs="+")
    d.add_argument("--start", default="2021-01-01")
    d.add_argument("--end", default=None)
    d.add_argument("--markets", default="spot", help="spot, perp or spot,perp")
    d.set_defaults(fn=cmd_download)

    r = sub.add_parser("run", help="Single backtest")
    _add_run_args(r)
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("sweep", help="Robustness sweep across intervals/offsets/symbols/params")
    _add_run_args(s)
    s.add_argument("--symbols", default=None, help="Comma list; defaults to --symbol")
    s.add_argument("--intervals", default="15m,1h,4h,1d")
    s.add_argument("--offsets", default="0", help="Comma list of bar offsets, 0 = none")
    s.add_argument("--jitter", type=int, default=0, help="Random parameter variants to add")
    s.add_argument("--jitter-pct", type=float, default=0.25)
    s.add_argument("--workers", type=int, default=None)
    s.set_defaults(fn=cmd_sweep)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
