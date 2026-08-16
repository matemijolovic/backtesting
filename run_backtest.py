"""Run the regime ATR strategy on Binance ETHUSDT 1-hour bars.

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

from strategies.regime import RegimeConfig
from strategies.regime import RegimeStrategy

STARTING_USDT = 100_000
SYMBOL = "ETHUSDT"
INTERVAL = "1h"
DAYS = 730


def to_hourly(ohlcv: pd.DataFrame) -> pd.DataFrame:
    if INTERVAL == "1h":
        return ohlcv
    hourly = ohlcv.resample("1h").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return hourly.dropna()


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
    instrument = TestInstrumentProvider.ethusdt_binance()
    bar_type = BarType.from_str(f"{instrument.id}-1-HOUR-LAST-EXTERNAL")
    ohlcv = to_hourly(download_klines(SYMBOL, INTERVAL, DAYS))
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
        default_leverage=Decimal(1),
    )
    engine.add_instrument(instrument)
    engine.add_data(bars)
    engine.add_strategy(
        RegimeStrategy(
            RegimeConfig(
                instrument_id=instrument.id,
                bar_type=bar_type,
            ),
        ),
    )

    print(
        f"Running regime ATR strategy on {len(bars):,} "
        f"{instrument.id} 1h bars ({ohlcv.index.min()} → {ohlcv.index.max()})...",
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
        start = account.iloc[0]["total"]
        end = account.iloc[-1]["total"]
        print(f"Starting balance:  {start} USDT")
        print(f"Ending balance:    {end} USDT")

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
