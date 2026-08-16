"""Delta-neutral Binance funding carry: long spot / short perp (or flip).

    uv sync
    uv run python run_funding.py
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from download_binance import download_funding
from download_binance import download_klines
from nautilus_trader.backtest.config import SimulationModuleConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.modules import SimulationModule
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.core.datetime import as_utc_index
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import FundingRateUpdate
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from strategies.funding_carry import FundingCarry
from strategies.funding_carry import FundingCarryConfig

STARTING_USDT = 100_000
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
DAYS = 1825
SPY_TOTAL_RETURN = 0.8581  # 2021-08-16 adj close → 2026-08-14


class _ExchangeHook(SimulationModule):
    """Holds a SimulatedExchange handle so funding can credit the account."""

    def process(self, ts_now: int) -> None:
        return

    def log_diagnostics(self, logger) -> None:
        return

    def reset(self) -> None:
        return


def bars_from_ohlcv(ohlcv: pd.DataFrame, instrument, bar_type):
    wrangler = BarDataWrangler(bar_type, instrument)
    ohlcv = as_utc_index(ohlcv)
    values = np.ascontiguousarray(
        ohlcv.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(
            dtype=np.float64,
            copy=True,
        )
    )
    ts_events = ohlcv.index.view(np.uint64)
    return list(map(wrangler._build_bar, values, ts_events, ts_events))


def funding_updates(frame: pd.DataFrame, instrument_id) -> list[FundingRateUpdate]:
    rates = frame["funding_rate"].to_numpy()
    ts = (
        pd.DatetimeIndex(pd.to_datetime(frame["funding_time"], utc=True))
        .as_unit("ns")
        .asi8.astype(np.uint64)
    )
    out: list[FundingRateUpdate] = []
    for i, rate in enumerate(rates):
        interval = None
        next_ns = None
        if i + 1 < len(ts):
            delta_min = int((int(ts[i + 1]) - int(ts[i])) / 60_000_000_000)
            if delta_min > 0:
                interval = delta_min
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


def main() -> int:
    spot = TestInstrumentProvider.btcusdt_binance()
    perp = TestInstrumentProvider.btcusdt_perp_binance()
    spot_bar_type = BarType.from_str(f"{spot.id}-1-HOUR-LAST-EXTERNAL")
    perp_bar_type = BarType.from_str(f"{perp.id}-1-HOUR-LAST-EXTERNAL")

    spot_ohlcv = download_klines(SYMBOL, INTERVAL, DAYS, market="spot")
    perp_ohlcv = download_klines(SYMBOL, INTERVAL, DAYS, market="perp")
    funding = download_funding(SYMBOL, DAYS)

    spot_bars = bars_from_ohlcv(spot_ohlcv, spot, spot_bar_type)
    perp_bars = bars_from_ohlcv(perp_ohlcv, perp, perp_bar_type)
    rates = funding_updates(funding, perp.id)

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            logging=LoggingConfig(log_level="ERROR"),
        ),
    )
    binance = Venue("BINANCE")
    hook = _ExchangeHook(SimulationModuleConfig())
    engine.add_venue(
        venue=binance,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(STARTING_USDT, USDT)],
        default_leverage=Decimal(5),
        modules=[hook],
        allow_cash_borrowing=True,
    )
    engine.add_instrument(spot)
    engine.add_instrument(perp)
    engine.add_data(spot_bars, sort=False)
    engine.add_data(perp_bars, sort=False)
    engine.add_data(rates, sort=False)
    engine.sort_data()

    strategy = FundingCarry(
        FundingCarryConfig(
            spot_id=spot.id,
            perp_id=perp.id,
            spot_bar_type=spot_bar_type,
            perp_bar_type=perp_bar_type,
        ),
    )
    strategy.bind_exchange(hook.exchange)
    engine.add_strategy(strategy)

    start = min(spot_ohlcv.index.min(), perp_ohlcv.index.min())
    end = max(spot_ohlcv.index.max(), perp_ohlcv.index.max())
    print(
        f"Running funding carry on {SYMBOL} "
        f"({len(spot_bars):,} spot + {len(perp_bars):,} perp {INTERVAL} bars, "
        f"{len(rates):,} funding payments, {start} → {end})...",
        flush=True,
    )
    engine.run()

    fills = engine.trader.generate_order_fills_report()
    positions = engine.trader.generate_positions_report()
    account = engine.trader.generate_account_report(binance)

    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.max_colwidth", 28)

    print("\n=== Summary ===")
    print(f"Spot bars:         {len(spot_bars):,}")
    print(f"Perp bars:         {len(perp_bars):,}")
    print(f"Funding payments:  {len(rates):,}")
    print(f"Fills:             {len(fills):,}")
    print(f"Closed positions:  {len(positions):,}")
    print(f"Funding settled:   {strategy.n_settlements:,}")
    print(f"  received:        {strategy.n_received:,}")
    print(f"  paid:            {strategy.n_paid:,}")
    print(f"Funding PnL:       {strategy.funding_pnl:+,.2f} USDT")

    if not funding.empty:
        mean_rate = float(funding["funding_rate"].mean())
        print(f"Mean funding rate: {mean_rate:+.6%} per payment")

    if not account.empty:
        start_bal = float(account.iloc[0]["total"])
        end_bal = float(account.iloc[-1]["total"])
        ret = end_bal / start_bal - 1
        trade_pnl = (end_bal - start_bal) - strategy.funding_pnl
        spy_end = start_bal * (1 + SPY_TOTAL_RETURN)
        print(f"Trading/basis PnL: {trade_pnl:+,.2f} USDT")
        print(f"Starting balance:  {start_bal:,.2f} USDT")
        print(f"Ending balance:    {end_bal:,.2f} USDT")
        print(f"Strategy return:   {ret:+.1%}")
        print(f"SPY total return:  {SPY_TOTAL_RETURN:+.1%}  (${spy_end:,.0f})")
        print(f"Beat SPY:          {'YES' if end_bal > spy_end else 'NO'}")

    if not positions.empty:
        cols = [
            c
            for c in (
                "instrument_id",
                "side",
                "ts_opened",
                "ts_closed",
                "avg_px_open",
                "avg_px_close",
                "realized_pnl",
            )
            if c in positions.columns
        ]
        print("\n=== Last 12 positions ===")
        print(positions[cols].tail(12).to_string())

    if not fills.empty:
        fill_cols = [
            c
            for c in ("instrument_id", "side", "quantity", "avg_px", "ts_init")
            if c in fills.columns
        ]
        print("\n=== Last 8 fills ===")
        print(fills[fill_cols].tail(8).to_string())

    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
