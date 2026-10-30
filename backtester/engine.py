"""Run one backtest and return everything needed to judge it.

    from backtester.engine import RunConfig, run_backtest
    result = run_backtest(RunConfig(strategy="trend_ensemble", symbol="BTCUSDT", interval="4h"))
    print(result.metrics)

Warmup: extra history before ``start`` is loaded so indicators are
ready on day one. Metrics and the buy-and-hold benchmark are measured
from ``start`` only.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from nautilus_trader.backtest.config import SimulationModuleConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.modules import SimulationModule
from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.core.datetime import as_utc_index
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import FundingRateUpdate
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler

from backtester import metrics as M
from backtester import registry
from backtester.data import interval_nautilus
from backtester.data import load_bars
from backtester.data import load_funding
from backtester.instruments import VENUE
from backtester.instruments import make_perp
from backtester.instruments import make_spot


@dataclass
class RunConfig:
    strategy: str = "trend_ensemble"
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    start: str = "2022-01-01"
    end: str | None = None  # None = now
    params: dict[str, Any] = field(default_factory=dict)
    offset: str | None = None  # shift bar boundaries, e.g. "30m"
    capital: float = 100_000.0
    leverage: float = 1.0
    fee_bps: float = 10.0  # per fill; Binance spot taker
    slippage_bps: float = 2.0  # per fill, added to the fee
    perp_fee_bps: float = 5.0  # carry only
    scale_lookbacks: bool = True  # legacy bar-count params follow the timeframe
    rebalance_threshold: float = 0.10
    use_stops: bool = True
    first_price: float = field(default=0.0, repr=False)

    def label(self) -> str:
        off = f"+{self.offset}" if self.offset else ""
        return f"{self.strategy} {self.symbol} {self.interval}{off}"


@dataclass
class BacktestResult:
    config: RunConfig
    params: dict[str, Any]
    equity: pd.Series
    exposure: pd.Series
    close: pd.Series
    benchmark: pd.Series
    fills: pd.DataFrame
    positions: pd.DataFrame
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]
    extras: dict[str, Any]
    elapsed: float

    def row(self) -> dict[str, Any]:
        """Flat dict for tables (one row per run)."""
        out = {
            "strategy": self.config.strategy,
            "symbol": self.config.symbol,
            "interval": self.config.interval,
            "offset": self.config.offset or "",
        }
        out.update(self.metrics)
        out["bh_cagr"] = self.benchmark_metrics.get("cagr")
        out["bh_sharpe"] = self.benchmark_metrics.get("sharpe")
        out["bh_max_drawdown"] = self.benchmark_metrics.get("max_drawdown")
        out["seconds"] = self.elapsed
        return out


# --------------------------------------------------------------------------- #
# Nautilus plumbing
# --------------------------------------------------------------------------- #


def bars_from_ohlcv(ohlcv: pd.DataFrame, instrument, bar_type) -> list[Bar]:
    """OHLCV (close-time index) -> Nautilus bars.

    Pandas 3 copy-on-write makes ``DataFrame.values`` read-only, which
    ``BarDataWrangler.process`` rejects, so rows go through ``_build_bar``.
    """
    wrangler = BarDataWrangler(bar_type, instrument)
    ohlcv = as_utc_index(ohlcv)
    values = np.ascontiguousarray(
        ohlcv.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64, copy=True)
    )
    ts = ohlcv.index.as_unit("ns").asi8.astype(np.uint64)
    return list(map(wrangler._build_bar, values, ts, ts))


class EquityRecorderConfig(ActorConfig, frozen=True):
    bar_type: BarType
    instrument_ids: list[InstrumentId]


class EquityRecorder(Actor):
    """Samples mark-to-market equity and net exposure on every bar."""

    def __init__(self, config: EquityRecorderConfig) -> None:
        super().__init__(config)
        self.ts: list[int] = []
        self.equity: list[float] = []
        self.exposure: list[float] = []

    def on_start(self) -> None:
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        eq = self.portfolio.equity(VENUE).get(USDT)
        if eq is None:
            return
        value = eq.as_double()
        close = bar.close.as_double()
        net = sum(float(self.portfolio.net_position(i)) for i in self.config.instrument_ids)
        self.ts.append(bar.ts_event)
        self.equity.append(value)
        self.exposure.append(net * close / value if value > 0 else 0.0)

    def series(self) -> tuple[pd.Series, pd.Series]:
        idx = pd.DatetimeIndex(np.asarray(self.ts, dtype="int64"), tz="UTC")
        return (
            pd.Series(self.equity, index=idx, name="equity"),
            pd.Series(self.exposure, index=idx, name="exposure"),
        )


class _ExchangeHook(SimulationModule):
    """Holds the SimulatedExchange so funding can be credited to the account."""

    def process(self, ts_now: int) -> None:
        return

    def log_diagnostics(self, logger) -> None:
        return

    def reset(self) -> None:
        return


def funding_updates(frame: pd.DataFrame, instrument_id) -> list[FundingRateUpdate]:
    rates = frame["funding_time"]
    ts = pd.DatetimeIndex(pd.to_datetime(rates, utc=True)).as_unit("ns").asi8
    out: list[FundingRateUpdate] = []
    for i, rate in enumerate(frame["funding_rate"].to_numpy()):
        interval = next_ns = None
        if i + 1 < len(ts):
            delta_min = int((int(ts[i + 1]) - int(ts[i])) // 60_000_000_000)
            interval = delta_min if delta_min > 0 else None
            next_ns = int(ts[i + 1])
        out.append(
            FundingRateUpdate(
                instrument_id=instrument_id,
                rate=Decimal(str(rate)),
                ts_event=int(ts[i]),
                ts_init=int(ts[i]),
                interval=interval,
                next_funding_ns=next_ns,
            )
        )
    return out


def _engine() -> BacktestEngine:
    return BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #


def _window(cfg: RunConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(cfg.start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    if cfg.end:
        end = pd.Timestamp(cfg.end)
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    else:
        end = pd.Timestamp.now(tz="UTC").floor("min")
    if end <= start:
        raise ValueError(f"end {end} must be after start {start}")
    return start, end


def run_backtest(
    cfg: RunConfig,
    *,
    bars: pd.DataFrame | None = None,
    progress=None,
) -> BacktestResult:
    """Run ``cfg``. ``bars`` (close-time indexed OHLCV incl. warmup) skips loading."""
    spec = registry.get(cfg.strategy)
    params = spec.resolve(cfg.params, cfg.interval, cfg.scale_lookbacks)
    start, end = _window(cfg)
    warmup = pd.Timedelta(days=spec.warmup_days(params, cfg.interval))
    if spec.kind == "carry":
        return _run_carry(cfg, spec, params, start, end, progress)

    t0 = time.perf_counter()
    if bars is None:
        bars = load_bars(cfg.symbol, cfg.interval, start - warmup, end, offset=cfg.offset, progress=progress)
    bars = bars[(bars.index > start - warmup) & (bars.index <= end)]
    if len(bars) < 10:
        raise ValueError(f"Not enough {cfg.symbol} {cfg.interval} data between {start - warmup} and {end}")

    first_price = float(bars["close"].iloc[0])
    cfg = replace(cfg, first_price=first_price)
    instrument = make_spot(cfg.symbol, fee_bps=cfg.fee_bps, slippage_bps=cfg.slippage_bps, ref_price=first_price)
    bar_type = BarType.from_str(f"{instrument.id}-{interval_nautilus(cfg.interval)}-LAST-EXTERNAL")
    nbars = bars_from_ohlcv(bars, instrument, bar_type)

    engine = _engine()
    engine.add_venue(
        venue=VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(cfg.capital, USDT)],
        default_leverage=Decimal(max(int(np.ceil(cfg.leverage)) + 1, 2)),
    )
    engine.add_instrument(instrument)
    engine.add_data(nbars)
    recorder = EquityRecorder(EquityRecorderConfig(bar_type=bar_type, instrument_ids=[instrument.id]))
    engine.add_actor(recorder)
    strategy = spec.build(spec, instrument, bar_type, params, cfg)
    engine.add_strategy(strategy)
    try:
        engine.run()
        fills = engine.trader.generate_order_fills_report()
        positions = engine.trader.generate_positions_report()
    finally:
        engine.dispose()

    extras = {"bars": len(nbars), "n_stops": getattr(strategy, "n_stops", None)}
    return _finish(cfg, params, recorder, bars["close"], fills, positions, start, extras, t0)


def _run_carry(cfg, spec, params, start, end, progress) -> BacktestResult:
    from strategies.funding_carry import FundingCarry
    from strategies.funding_carry import FundingCarryConfig

    t0 = time.perf_counter()
    spot_bars = load_bars(cfg.symbol, cfg.interval, start, end, market="spot", offset=cfg.offset, progress=progress)
    perp_bars = load_bars(cfg.symbol, cfg.interval, start, end, market="perp", offset=cfg.offset, progress=progress)
    common = spot_bars.index.intersection(perp_bars.index)
    spot_bars, perp_bars = spot_bars.loc[common], perp_bars.loc[common]
    if len(common) < 10:
        raise ValueError(f"Not enough overlapping spot/perp data for {cfg.symbol}")
    funding = load_funding(cfg.symbol, start, end)

    ref = float(spot_bars["close"].iloc[0])
    spot = make_spot(cfg.symbol, fee_bps=cfg.fee_bps, slippage_bps=cfg.slippage_bps, ref_price=ref)
    perp = make_perp(cfg.symbol, fee_bps=cfg.perp_fee_bps, slippage_bps=cfg.slippage_bps, ref_price=ref)
    step = interval_nautilus(cfg.interval)
    spot_type = BarType.from_str(f"{spot.id}-{step}-LAST-EXTERNAL")
    perp_type = BarType.from_str(f"{perp.id}-{step}-LAST-EXTERNAL")

    engine = _engine()
    hook = _ExchangeHook(SimulationModuleConfig())
    engine.add_venue(
        venue=VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(cfg.capital, USDT)],
        default_leverage=Decimal(10),
        modules=[hook],
        allow_cash_borrowing=True,
    )
    engine.add_instrument(spot)
    engine.add_instrument(perp)
    engine.add_data(bars_from_ohlcv(spot_bars, spot, spot_type), sort=False)
    engine.add_data(bars_from_ohlcv(perp_bars, perp, perp_type), sort=False)
    engine.add_data(funding_updates(funding, perp.id), sort=False)
    engine.sort_data()

    recorder = EquityRecorder(EquityRecorderConfig(bar_type=perp_type, instrument_ids=[spot.id, perp.id]))
    engine.add_actor(recorder)
    strategy = FundingCarry(
        FundingCarryConfig(
            spot_id=spot.id,
            perp_id=perp.id,
            spot_bar_type=spot_type,
            perp_bar_type=perp_type,
            **params,
        )
    )
    strategy.bind_exchange(hook.exchange)
    engine.add_strategy(strategy)
    try:
        engine.run()
        fills = engine.trader.generate_order_fills_report()
        positions = engine.trader.generate_positions_report()
    finally:
        engine.dispose()

    extras = {
        "bars": len(common),
        "funding_payments": len(funding),
        "funding_pnl": strategy.funding_pnl,
        "funding_received": strategy.n_received,
        "funding_paid": strategy.n_paid,
        "mean_funding_rate": float(funding["funding_rate"].mean()) if len(funding) else float("nan"),
    }
    return _finish(cfg, params, recorder, spot_bars["close"], fills, positions, start, extras, t0)


def _finish(cfg, params, recorder, close, fills, positions, start, extras, t0) -> BacktestResult:
    equity, exposure = recorder.series()
    # Measure from the first bar at/after start; earlier bars were warmup.
    equity, exposure = equity[equity.index >= start], exposure[exposure.index >= start]
    close = close[close.index >= start]
    if not equity.empty and equity.iloc[0] > 0:
        # Warmup trades may have moved equity; rebase so the run starts at capital.
        equity = equity * (cfg.capital / equity.iloc[0])
    benchmark = cfg.capital * close / close.iloc[0] if len(close) else close

    if positions is not None and not positions.empty and "ts_opened" in positions:
        opened = pd.to_datetime(positions["ts_opened"], utc=True)
        positions = positions[opened >= start]
    if fills is not None and not fills.empty and "ts_last" in fills:
        fills = fills[pd.to_datetime(fills["ts_last"], utc=True) >= start]

    stats = M.summary(equity)
    stats.update(M.trade_stats(positions))
    stats.update(M.exposure_stats(exposure))
    stats["fees"] = M.fees_paid(fills)
    stats["final_equity"] = float(equity.iloc[-1]) if len(equity) else float("nan")
    return BacktestResult(
        config=cfg,
        params=params,
        equity=equity,
        exposure=exposure,
        close=close,
        benchmark=benchmark,
        fills=fills,
        positions=positions,
        metrics=stats,
        benchmark_metrics=M.summary(benchmark),
        extras=extras,
        elapsed=time.perf_counter() - t0,
    )


def config_dict(cfg: RunConfig) -> dict[str, Any]:
    out = asdict(cfg)
    out.pop("first_price", None)
    return out
