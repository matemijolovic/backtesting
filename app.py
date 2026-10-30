"""Backtest lab UI.

    uv run streamlit run app.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from datetime import timedelta

import pandas as pd
import streamlit as st

from backtester import charts
from backtester import metrics as M
from backtester import registry
from backtester.data import COMMON_INTERVALS
from backtester.data import interval_minutes
from backtester.data import load_1m
from backtester.data import quality
from backtester.data import resample
from backtester.engine import RunConfig
from backtester.engine import config_dict
from backtester.engine import run_backtest

st.set_page_config(page_title="Backtest Lab", page_icon="📈", layout="wide")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT"]
PCT_KEYS = {"total_return", "cagr", "volatility", "max_drawdown", "best_day", "worst_day", "win_rate", "time_in_market"}


# --------------------------------------------------------------------------- #
# Cached runners
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_run(cfg_json: str):
    return run_backtest(RunConfig(**json.loads(cfg_json)))


def run_cached(cfg: RunConfig):
    return _cached_run(json.dumps(config_dict(cfg), sort_keys=True, default=str))


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_1m(symbol: str, start: str, end: str):
    return load_1m(symbol, start, end)


def fmt(key: str, value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "–"
    if isinstance(value, float):
        if key in PCT_KEYS:
            return f"{value:+.1%}"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:,.2f}"
    return str(value)


# --------------------------------------------------------------------------- #
# Sidebar: market, costs, strategy
# --------------------------------------------------------------------------- #


def param_widget(p, key_prefix: str, value):
    key = f"{key_prefix}.{p.name}"
    label = p.name.replace("_", " ")
    if p.kind == "bool":
        return st.checkbox(label, value=bool(value), help=p.help, key=key)
    if p.kind == "choice":
        return st.selectbox(label, p.choices, index=list(p.choices).index(value), help=p.help, key=key)
    if p.kind == "int":
        return int(st.number_input(label, value=int(value), min_value=int(p.lo) if p.lo is not None else None,
                                   max_value=int(p.hi) if p.hi is not None else None, step=int(p.step or 1), help=p.help, key=key))
    if p.kind == "float":
        step = float(p.step or 0.01)
        decimals = max(0, -int(f"{step:e}".split("e")[1]))
        return float(st.number_input(label, value=float(value), min_value=float(p.lo) if p.lo is not None else None,
                                     max_value=float(p.hi) if p.hi is not None else None, step=step,
                                     format=f"%.{decimals}f", help=p.help, key=key))
    return st.text_input(label, value=str(value), help=p.help, key=key)


def sidebar() -> RunConfig:
    sb = st.sidebar
    sb.title("Backtest Lab")

    keys = list(registry.REGISTRY)
    strategy = sb.selectbox(
        "Strategy", keys, index=keys.index("ensemble"),
        format_func=lambda k: registry.REGISTRY[k].title,
    )
    spec = registry.get(strategy)
    sb.caption(spec.summary)

    sb.subheader("Market")
    symbol = sb.selectbox("Symbol", SYMBOLS + ["Other…"], index=0)
    if symbol == "Other…":
        symbol = sb.text_input("Binance USDT symbol", "LTCUSDT").upper().strip()
    options = list(COMMON_INTERVALS) + ["custom"]
    interval = sb.selectbox("Bar size", options, index=options.index("1h"))
    if interval == "custom":
        interval = sb.text_input("Custom bar size", "90m", help="Any <n>m, <n>h or <n>d")
    offset = sb.text_input("Bar offset", "", help="Shift bar boundaries, e.g. 30m makes 1h bars start at :30")
    c1, c2 = sb.columns(2)
    start = c1.date_input("Start", date(2022, 1, 1), min_value=date(2017, 8, 1))
    end = c2.date_input("End", date.today(), min_value=date(2017, 8, 2))

    sb.subheader("Costs & sizing")
    capital = sb.number_input("Capital (USDT)", value=100_000, step=10_000, min_value=1_000)
    c1, c2 = sb.columns(2)
    leverage = c1.number_input("Max leverage", value=1.0, step=0.25, min_value=0.25, max_value=5.0,
                               help="Exposure at full signal, as a multiple of equity")
    fee_bps = c2.number_input("Fee (bps/fill)", value=10.0, step=0.5, min_value=0.0,
                              help="Binance spot taker 10 bps, 7.5 with BNB; perps ~5")
    slippage_bps = c1.number_input("Slippage (bps/fill)", value=2.0, step=0.5, min_value=0.0)
    perp_fee_bps = 5.0
    if spec.kind == "carry":
        perp_fee_bps = sb.number_input("Perp fee (bps/fill)", value=5.0, step=0.5, min_value=0.0)
    scale = True
    if spec.kind == "legacy":
        scale = sb.checkbox(
            f"Scale lookbacks from {spec.ref_interval}", value=True,
            help=f"Lookbacks are bar counts calibrated on {spec.ref_interval}; rescale so they span the same time on other bar sizes.",
        )
    rebalance = 0.10
    use_stops = True
    if spec.kind == "signal":
        with sb.expander("Execution"):
            rebalance = st.slider("Rebalance threshold", 0.0, 0.5, 0.10, 0.01,
                                  help="Skip resizing trades smaller than this fraction of equity")
            use_stops = st.checkbox("Venue stop orders", True, help="Turn signal stops into real STOP_MARKET orders")

    sb.subheader("Parameters")
    params = {}
    with sb:
        for p in spec.params:
            params[p.name] = param_widget(p, strategy, p.default)
    if sb.button("Reset parameters"):
        for p in spec.params:
            st.session_state.pop(f"{strategy}.{p.name}", None)
        st.rerun()

    return RunConfig(
        strategy=strategy, symbol=symbol, interval=interval.strip(), start=str(start),
        end=str(end + timedelta(days=1)) if end < date.today() else None,
        params=params, offset=offset.strip() or None, capital=float(capital), leverage=float(leverage),
        fee_bps=float(fee_bps), slippage_bps=float(slippage_bps), perp_fee_bps=float(perp_fee_bps),
        scale_lookbacks=scale, rebalance_threshold=float(rebalance), use_stops=use_stops,
    )


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #


def kpi_row(result) -> None:
    m, b = result.metrics, result.benchmark_metrics
    cols = st.columns(6)
    cols[0].metric("CAGR", fmt("cagr", m.get("cagr")), delta=f"{m.get('cagr', 0) - b.get('cagr', 0):+.1%} vs B&H")
    cols[1].metric("Sharpe", fmt("sharpe", m.get("sharpe")), delta=f"{m.get('sharpe', 0) - b.get('sharpe', 0):+.2f} vs B&H")
    cols[2].metric("Max drawdown", fmt("max_drawdown", m.get("max_drawdown")),
                   delta=f"{m.get('max_drawdown', 0) - b.get('max_drawdown', 0):+.1%} vs B&H")
    cols[3].metric("Total return", fmt("total_return", m.get("total_return")))
    cols[4].metric("Trades", fmt("trades", m.get("trades")), delta=f"win {m.get('win_rate', float('nan')):.0%}"
                   if m.get("trades") else None, delta_color="off")
    cols[5].metric("Fees paid", f"{m.get('fees', 0):,.0f}", delta=f"{m.get('time_in_market', 0):.0%} in market", delta_color="off")


def tab_backtest(cfg: RunConfig) -> None:
    left, right = st.columns([3, 1])
    left.markdown(f"**{registry.get(cfg.strategy).title}** on **{cfg.symbol} {cfg.interval}**"
                  f"{' +' + cfg.offset if cfg.offset else ''} · {cfg.start} → {cfg.end or 'now'}")
    if right.button("Run backtest", type="primary", width="stretch"):
        with st.spinner("Loading data and running Nautilus…"):
            try:
                st.session_state["result"] = run_cached(cfg)
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
                return
    result = st.session_state.get("result")
    if result is None:
        st.info("Set up the strategy in the sidebar and press **Run backtest**. "
                "First runs download 1-minute Binance data; later runs read the local cache.")
        return
    if config_dict(result.config) != config_dict(cfg):
        st.caption("Showing the last run. Settings changed since; press Run backtest to refresh.")

    kpi_row(result)
    log = st.toggle("Log scale", value=False)
    st.plotly_chart(charts.equity(result, log=log), width="stretch")
    st.plotly_chart(charts.drawdown(result), width="stretch")
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.exposure(result), width="stretch")
    with c2:
        st.markdown("**Returns by year**")
        yearly = M.period_returns(result.equity, "YE")
        bh_yearly = M.period_returns(result.benchmark, "YE")
        st.dataframe(pd.DataFrame({"Strategy": yearly.values, "Buy & hold": bh_yearly.reindex(yearly.index).values},
                                  index=yearly.index.year).style.format("{:+.1%}"), width="stretch")
    st.markdown("**Returns by month**")
    st.plotly_chart(charts.monthly_heatmap(M.monthly_table(result.equity)), width="stretch")
    st.markdown("**Price and fills**")
    st.plotly_chart(charts.price_with_trades(result), width="stretch")

    with st.expander("All metrics"):
        rows = [(k, fmt(k, v), fmt(k, result.benchmark_metrics.get(k))) for k, v in result.metrics.items()]
        st.dataframe(pd.DataFrame(rows, columns=["metric", "strategy", "buy & hold"]), hide_index=True, width="stretch")
        st.json({"params (resolved)": result.params, "extras": {k: v for k, v in result.extras.items() if v is not None}})
    with st.expander(f"Round trips ({len(result.positions)})"):
        cols = [c for c in ("side", "ts_opened", "ts_closed", "avg_px_open", "avg_px_close", "peak_qty", "realized_return", "realized_pnl")
                if c in result.positions.columns]
        st.dataframe(result.positions[cols], width="stretch")
    with st.expander(f"Fills ({len(result.fills)})"):
        cols = [c for c in ("ts_last", "type", "side", "filled_qty", "avg_px", "commissions") if c in result.fills.columns]
        st.dataframe(result.fills[cols], width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Equity CSV", result.equity.to_csv().encode(), "equity.csv", "text/csv")
    c2.download_button("Round trips CSV", result.positions.to_csv().encode(), "positions.csv", "text/csv")
    c3.download_button("Fills CSV", result.fills.to_csv().encode(), "fills.csv", "text/csv")


def tab_robustness(cfg: RunConfig) -> None:
    from backtester.robustness import build_grid
    from backtester.robustness import combine_equity
    from backtester.robustness import jitter_params
    from backtester.robustness import run_sweep
    from backtester.robustness import summarize
    from backtester.robustness import table

    st.markdown(
        "Run the sidebar strategy across **bar sizes, bar offsets, symbols and perturbed parameters**. "
        "A real edge shows up as a broad plateau. A fitted one shows up as a single bright cell."
    )
    c1, c2 = st.columns(2)
    symbols = c1.multiselect("Symbols", SYMBOLS, default=[cfg.symbol] if cfg.symbol in SYMBOLS else ["BTCUSDT"])
    extra = c1.text_input("Extra symbols (comma separated)", "")
    symbols += [s.strip().upper() for s in extra.split(",") if s.strip()]
    intervals = c2.multiselect("Bar sizes", list(COMMON_INTERVALS) + ["1m", "3m", "45m", "90m", "3h", "6h", "2d"],
                               default=["15m", "1h", "4h", "1d"])
    offsets = c2.multiselect("Bar offsets", ["0", "5m", "15m", "30m", "1h", "2h"], default=["0"],
                             help="Same bar size, shifted boundaries. Offsets >= bar size wrap around.")
    c1, c2, c3 = st.columns(3)
    jitter = c1.number_input("Random parameter variants", 0, 50, 0, help="Each numeric parameter moved randomly")
    jitter_pct = c2.slider("Perturbation ±", 0.05, 0.5, 0.25, 0.05)
    workers = c3.number_input("Parallel workers", 1, 32, 8)

    param_sets = [cfg.params] + (jitter_params(cfg.strategy, cfg.params, int(jitter), jitter_pct) if jitter else [])
    grid = build_grid(cfg, symbols=symbols, intervals=intervals, offsets=[None if o == "0" else o for o in offsets], param_sets=param_sets)
    heavy = [i for i in intervals if interval_minutes(i) < 5]
    st.caption(f"{len(grid)} backtests" + (f" · {', '.join(heavy)} bars are slow over long windows" if heavy else ""))

    if st.button("Run sweep", type="primary", disabled=not grid):
        bar = st.progress(0.0, text="Preparing data…")

        def progress(done, total, run):
            bar.progress(done / total, text=f"{done}/{total} · {run.config.label()}")

        try:
            st.session_state["sweep"] = (cfg.strategy, run_sweep(grid, workers=int(workers), progress=progress))
        except Exception as exc:
            st.error(f"Sweep failed: {exc}")
            return
        bar.empty()

    if "sweep" not in st.session_state:
        return
    strategy, runs = st.session_state["sweep"]
    frame = table(runs)
    errors = frame[frame.get("error").notna()] if "error" in frame else pd.DataFrame()
    ok_runs = [r for r in runs if r.error is None]
    frame = frame[frame["sharpe"].notna()] if "sharpe" in frame else frame
    s = summarize(runs)
    st.subheader(f"{registry.get(strategy).title}: {s.get('runs', 0)} runs")
    cols = st.columns(6)
    cols[0].metric("Profitable runs", f"{s.get('profitable', 0):.0%}")
    cols[1].metric("Beat B&H Sharpe", f"{s.get('beats_bh_sharpe', 0):.0%}")
    cols[2].metric("Median Sharpe", f"{s.get('sharpe_median', float('nan')):.2f}")
    cols[3].metric("10th pct Sharpe", f"{s.get('sharpe_p10', float('nan')):.2f}")
    cols[4].metric("Median max DD", f"{s.get('max_dd_median', float('nan')):.1%}")
    cols[5].metric("Worst max DD", f"{s.get('max_dd_worst', float('nan')):.1%}")
    if len(errors):
        st.warning(f"{len(errors)} runs failed, e.g. {errors['error'].iloc[0]}")
    if frame.empty:
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Median Sharpe by symbol × bar size**")
        st.plotly_chart(charts.grid_heatmap(frame, "sharpe"), width="stretch")
    with c2:
        st.markdown("**Median CAGR by symbol × bar size**")
        st.plotly_chart(charts.grid_heatmap(frame, "cagr", ".0%"), width="stretch")
    st.markdown("**Every run's Sharpe (dash = buy & hold)**")
    st.plotly_chart(charts.sharpe_strip(frame, "interval"), width="stretch")

    named = {r.config.label() + (f" #{i}" if len(param_sets) > 1 else ""): r.equity for i, r in enumerate(ok_runs)}
    portfolio = None
    if len({r.config.symbol for r in ok_runs}) > 1:
        portfolio = combine_equity([r.equity for r in ok_runs], cfg.capital)
    st.markdown("**Equity curves** (growth of 1" + ("; bold = equal-weight portfolio of all runs" if portfolio is not None else "") + ")")
    st.plotly_chart(charts.curves(named, reference=portfolio, highlight="Equal-weight portfolio", log=True), width="stretch")
    if portfolio is not None:
        pm = M.summary(portfolio)
        st.caption(f"Equal-weight portfolio: CAGR {pm['cagr']:+.1%} · Sharpe {pm['sharpe']:.2f} · max DD {pm['max_drawdown']:.1%}")

    years = pd.concat({r.config.label() + f" #{i}": r.yearly for i, r in enumerate(ok_runs)}, axis=1)
    if not years.empty:
        st.markdown("**Returns by year** (median / worst across runs)")
        st.dataframe(pd.DataFrame({"median": years.median(axis=1), "worst": years.min(axis=1), "best": years.max(axis=1),
                                   "share positive": (years > 0).mean(axis=1)}).style.format("{:+.1%}"),
                     width="stretch")
    with st.expander("All runs"):
        st.dataframe(frame, width="stretch", hide_index=True)
    st.download_button("Sweep CSV", frame.to_csv(index=False).encode(), "sweep.csv", "text/csv")


def tab_compare(cfg: RunConfig) -> None:
    st.markdown("Run several strategies on the sidebar's market and costs, each with its **default** parameters.")
    keys = [k for k, s in registry.REGISTRY.items() if s.kind != "carry"]
    picked = st.multiselect("Strategies", keys, default=["trend_ensemble", "breakout", "supertrend", "ensemble", "dual_ma"],
                            format_func=lambda k: registry.REGISTRY[k].title, max_selections=7)
    if st.button("Compare", type="primary", disabled=not picked):
        results = {}
        bar = st.progress(0.0)
        for i, key in enumerate(picked):
            bar.progress(i / len(picked), text=f"Running {key}…")
            try:
                results[key] = run_cached(replace(cfg, strategy=key, params={}))
            except Exception as exc:
                st.error(f"{key}: {exc}")
        bar.empty()
        st.session_state["compare"] = results
    results = st.session_state.get("compare")
    if not results:
        return
    first = next(iter(results.values()))
    rows = {key: r.metrics for key, r in results.items()}
    rows["buy & hold"] = first.benchmark_metrics
    frame = pd.DataFrame(rows).T
    cols = [c for c in ("cagr", "sharpe", "sortino", "max_drawdown", "calmar", "volatility", "trades", "win_rate", "time_in_market", "fees")
            if c in frame.columns]
    st.dataframe(frame[cols].style.format({c: "{:+.1%}" for c in cols if c in PCT_KEYS} | {"sharpe": "{:.2f}", "sortino": "{:.2f}",
                                                                                         "calmar": "{:.2f}", "trades": "{:.0f}", "fees": "{:,.0f}"},
                                          na_rep="–"), width="stretch")
    named = {registry.get(k).title: r.equity for k, r in results.items()}
    named["Buy & hold"] = first.benchmark
    st.plotly_chart(charts.curves(named, log=True), width="stretch")


def tab_data(cfg: RunConfig) -> None:
    st.markdown("Every bar size is resampled from the same cached **1-minute** Binance data, so results across "
                "timeframes differ only in bar size, never in source.")
    if st.button("Inspect data"):
        with st.spinner("Loading…"):
            end = cfg.end or str(pd.Timestamp.now(tz="UTC").floor("min"))
            st.session_state["data"] = (cfg.symbol, cfg.interval, cfg.offset, _cached_1m(cfg.symbol, cfg.start, end))
    if "data" not in st.session_state:
        return
    symbol, interval, offset, base = st.session_state["data"]
    q = quality(base)
    cols = st.columns(5)
    cols[0].metric("1m bars", f"{q.bars:,}")
    cols[1].metric("Coverage", f"{q.coverage:.3%}")
    cols[2].metric("Missing minutes", f"{q.missing_minutes:,}")
    cols[3].metric("Largest gap", str(q.largest_gap))
    cols[4].metric("Zero-volume bars", f"{q.zero_volume_bars:,}")
    bars = resample(base, interval, offset)
    n = st.slider("Bars to show", 50, 2000, 300, 50)
    st.plotly_chart(charts.candles(bars.tail(n)), width="stretch")
    gaps = base.index.to_series().diff()
    big = gaps[gaps > pd.Timedelta(minutes=5)].sort_values(ascending=False).head(20)
    if len(big):
        st.markdown("**Largest gaps in the 1m data**")
        st.dataframe(pd.DataFrame({"gap ends": big.index, "length": big.values.astype("timedelta64[m]").astype(str)}),
                     hide_index=True, width="stretch")


def main() -> None:
    cfg = sidebar()
    tabs = st.tabs(["Backtest", "Robustness", "Compare", "Data"])
    with tabs[0]:
        tab_backtest(cfg)
    with tabs[1]:
        tab_robustness(cfg)
    with tabs[2]:
        tab_compare(cfg)
    with tabs[3]:
        tab_data(cfg)


if __name__ == "__main__":
    main()
