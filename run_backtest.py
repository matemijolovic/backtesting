"""Run a fully invested 20/50 SMA trend follower on Binance BTCUSDT daily bars.

    uv sync
    uv run python run_backtest.py
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

from download_binance import download_klines
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.core.datetime import as_utc_index
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from strategies.dual_ma import DualMAConfig
from strategies.dual_ma import DualMATrend

STARTING_USDT = 100_000
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
DAYS = 730
SPY_TOTAL_RETURN = 0.4337  # 2024-08-16 adj close → 2026-08-14


def to_daily(ohlcv: pd.DataFrame) -> pd.DataFrame:
    daily = ohlcv.resample("1D").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return daily.dropna()


def bars_from_ohlcv(ohlcv: pd.DataFrame, instrument, bar_type):
    """Convert OHLCV into Nautilus bars.

    Pandas 3 Copy-on-Write makes DataFrame.values read-only, which
    BarDataWrangler.process rejects. Feed it writable NumPy rows instead.
    """
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


def main() -> int:
    instrument = TestInstrumentProvider.btcusdt_binance()
    bar_type = BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL")
    ohlcv = to_daily(download_klines(SYMBOL, INTERVAL, DAYS))
    bars = bars_from_ohlcv(ohlcv, instrument, bar_type)

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            logging=LoggingConfig(log_level="ERROR"),
        ),
    )
    binance = Venue("BINANCE")
    engine.add_venue(
        venue=binance,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(STARTING_USDT, USDT)],
        default_leverage=Decimal(2),
    )
    engine.add_instrument(instrument)
    engine.add_data(bars)
    engine.add_strategy(
        DualMATrend(
            DualMAConfig(
                instrument_id=instrument.id,
                bar_type=bar_type,
                fast=20,
                slow=50,
                leverage=1.0,
            ),
        ),
    )

    print(
        f"Running always-in SMA 20/50 on {len(bars):,} "
        f"{instrument.id} 1d bars ({ohlcv.index.min().date()} → {ohlcv.index.max().date()})...",
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
    print(f"Bars:              {len(bars):,}")
    print(f"Fills:             {len(fills):,}")
    print(f"Closed positions:  {len(positions):,}")
    if not account.empty:
        start = float(account.iloc[0]["total"])
        end = float(account.iloc[-1]["total"])
        ret = end / start - 1
        spy_end = start * (1 + SPY_TOTAL_RETURN)
        print(f"Starting balance:  {start:,.2f} USDT")
        print(f"Ending balance:    {end:,.2f} USDT")
        print(f"Strategy return:   {ret:+.1%}")
        print(f"SPY total return:  {SPY_TOTAL_RETURN:+.1%}  (${spy_end:,.0f})")
        print(f"Beat SPY:          {'YES' if end > spy_end else 'NO'}")

    if not positions.empty:
        cols = [
            c
            for c in (
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
            for c in ("side", "quantity", "avg_px", "ts_init")
            if c in fills.columns
        ]
        print("\n=== Last 8 fills ===")
        print(fills[fill_cols].tail(8).to_string())

    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
