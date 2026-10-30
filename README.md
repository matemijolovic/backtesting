# Backtest Lab

Nautilus Trader backtests on Binance data, built to answer one question
honestly: **does this edge survive a change of timeframe, bar alignment,
symbol and parameters, or was it fitted to one chart?**

```bash
uv sync
uv run streamlit run app.py                         # UI
uv run python -m backtester list                    # strategies + parameters
uv run python -m backtester run ensemble --symbol ETHUSDT --interval 4h --start 2022-01-01
uv run python -m backtester sweep trend_ensemble --symbols BTCUSDT,ETHUSDT,SOLUSDT \
    --intervals 15m,1h,4h,1d --offsets 0,30m --jitter 8 --jitter-pct 0.3 --out sweep.csv
uv run pytest                                       # 51 tests, offline, ~5s
```

The first run downloads 1-minute data (about 20s per symbol-year, then cached in `data/`).

## What's here

| Path | Role |
|---|---|
| `app.py` | Streamlit UI: **Backtest**, **Robustness**, **Compare** and **Data** tabs |
| `backtester/data.py` | Binance 1m archives → cache → any bar size (`7m`, `90m`, `3h`, ...) with optional boundary offset; data-quality report; funding history |
| `backtester/instruments.py` | Instruments for any Binance USDT symbol (tick/lot size from `exchangeInfo`); fee + slippage per fill |
| `backtester/engine.py` | `run_backtest(RunConfig)`: warmup handling, per-bar mark-to-market equity, buy & hold benchmark, funding carry |
| `backtester/metrics.py` | CAGR, Sharpe/Sortino (from **daily** equity, so comparable across bar sizes), drawdown depth and duration, trade stats, monthly table |
| `backtester/robustness.py` | Parallel sweeps over intervals × offsets × symbols × jittered params, a consistency summary, equal-weight portfolios |
| `backtester/registry.py` | Every strategy, its parameters and how to build it |
| `strategies/signal_base.py`, `signal_strategy.py` | New-style strategies: a pure-Python *signal* returns a target exposure; one Nautilus adapter handles sizing, rebalancing and real stop orders |
| `strategies/*.py` | The strategies (below) and the original ones |
| `run_backtest.py`, `run_funding.py`, `download_binance.py` | Kept as shortcuts to the CLI |

## Robust to the underlying data

- **One source for every timeframe.** All bar sizes are resampled from the
  same cached 1m bars, so a 15m vs 4h difference is the bar size and
  nothing else. Odd sizes (`45m`, `3h`) and shifted boundaries
  (`--offset 30m`: 4h bars starting 00:30) are first-class, because an edge that
  only exists on 4h bars starting at midnight UTC is noise.
- **Lookbacks in time, not bars.** New strategies take `*_days`/`*_hours`
  and convert per timeframe; volatility is measured per bar and rescaled to
  daily. The trend ensemble decides on a fixed UTC clock, so 15m and 1d data
  give near-identical trades (tested in `tests/test_signals.py`).
- **Legacy strategies are rescaled.** Their bar-count lookbacks are calibrated
  on a reference interval (e.g. dual MA 20/50 on 1d) and scaled to the run's
  timeframe, so `dual_ma` on 1h uses SMA 480/1200 rather than a 20-hour average.
  Turn this off with `--no-scale` / the sidebar checkbox.
- **Metrics that compare.** Sharpe/vol use daily returns (365/yr) regardless
  of bar size; drawdown uses the full-resolution curve.
- **Realistic execution.** Sizing on mark-to-market equity; one order moves
  straight to target (flips included); a rebalance band stops vol-targeting
  from churning fees; stops are real `STOP_MARKET` orders that fill intrabar at
  the stop (or the gap). Fees default to 10 bps + 2 bps slippage per fill.
- **Warmup.** History before `start` is loaded so indicators are ready on day
  one; metrics and the benchmark start at `start`.
- **Data hygiene.** Spot archives switched to microsecond timestamps in 2025;
  futures archives have headers. Both are handled. Duplicates and broken OHLC
  are cleaned, gaps stay gaps (see the **Data** tab).

## Strategies

| Key | Idea |
|---|---|
| `trend_ensemble` | **New.** Sign of 10/30/90-day returns averaged into a conviction, sized to a 60% vol target, rebalanced daily. Long-only by default. |
| `breakout` | **New.** Turtle-style: 20-day high entry, 10-day low exit, 3σ trailing stop, vol-targeted size. |
| `supertrend` | **New.** Supertrend with bands in *daily* σ (timeframe-invariant), band = stop order. |
| `mean_reversion` | **New.** Buy 2σ dips below the 2-day mean while above the 50-day trend. Kept as a template and counter-example (see results). |
| `ensemble` | **New.** 50% trend ensemble + 25% breakout + 25% supertrend (mean reversion optional). |
| `funding_carry` | Original: long spot / short perp while funding is positive. |
| `dual_ma`, `donchian`, `regime`, `ema_cross` | Original strategies, now runnable on any timeframe/symbol. |

Add a strategy: subclass `Signal` (see `strategies/breakout.py`, ~80 lines),
add it to `strategies/catalog.py`. It then shows up in the CLI, UI and sweeps.

## Results so far

Defaults, 10 bps fee + 2 bps slippage per fill, no leverage. Development
used **2021–2024 only**; 2025-01 → 2026-10 was held out and run once at the end.

Equal-weight portfolio of BTC, ETH, SOL and BNB, 1h bars. Buy & hold is the
same basket, unrebalanced.

| | 2021–24 (in-sample) | 2025–26 (out-of-sample) | Full period |
|---|---|---|---|
| **ensemble** | CAGR +67%, Sharpe 1.91, max DD −35% | +0.4%, 0.14, −34% | +42%, 1.40, −35% |
| trend_ensemble | +68%, 1.79, −43% | −3.1%, 0.03, −35% | +41%, 1.28, −43% |
| breakout | +37%, 1.41, −18% | −1.5%, 0.02, −28% | +23%, 1.04, −28% |
| dual_ma (legacy, scaled to daily 20/50) | +57%, 0.99, −52% | +16%, 0.56, −33% | +43%, 0.88, −52% |
| buy & hold | +140%, 1.40, −92% | −26%, −0.12, −69% | +66%, 1.02, −92% |
| funding_carry (BTC / ETH) | +21% / +30%, DD < 3.5% | +3.1% / +2.8% | |

What the robustness tooling shows:

- **The trend strategies generalise across data, not just one chart.** In
  2021–24 the trend ensemble had nearly the same Sharpe on 15m, 1h, 4h and 1d
  (BTC 0.91 / 0.92 / 0.88 / 0.94) and beat buy & hold's Sharpe in 16/16
  symbol × timeframe cells. With every parameter randomly moved ±30%, the
  ensemble's worst Sharpe was 0.97 (median 1.21).
- **But they are long-only trend followers.** In the 2025–26 decline they cut
  drawdowns roughly in half and ended flat while the basket lost 26%/yr. They
  protected capital; they did not make money. Expect that in any choppy or
  falling market.
- **Mean reversion has no edge here.** It lost in-sample and out-of-sample in
  every variant tried (lookback 24h–5d, z 1.5–2.5, with and without the trend
  filter). Short-horizon reversion in crypto majors is smaller than 12 bps
  per fill. It is weight 0 in the ensemble.
- **Funding carry's yield has collapsed.** Mean funding fell from about 1.2 bp
  to 0.4 bp per 8h, so the carry return is now below T-bill yields.
- **The always-in long/short `dual_ma` was the only one up in 2025–26**,
  because it shorts. It was the weakest in-sample. That is one regime, not
  evidence; sweep `allow_short=true` on the new strategies before drawing
  conclusions.

None of this predicts the future. Use the Robustness tab to see how wide the
plateau is before trusting any single result.

## Caveats

- Fills are at bar close (decision bar) for market orders. Stops fill intrabar.
  There is no order-book impact model; raise slippage for size.
- Spot shorts are modelled on a margin account with no borrow cost.
- Funding carry assumes unified margin (spot collateralises the perp short).
- The symbol list is chosen today, so there is survivorship bias across
  symbols. SOL's 2021 run flatters any long-biased result.
