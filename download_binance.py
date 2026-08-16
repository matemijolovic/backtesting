"""Download public Binance OHLCV bars. No API key required."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

import pandas as pd

KLINE_URLS = (
    "https://api.binance.com/api/v3/klines",
    "https://data-api.binance.vision/api/v3/klines",
)
DATA_DIR = Path(__file__).resolve().parent / "data"
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}


def parquet_path(symbol: str, interval: str) -> Path:
    return DATA_DIR / f"{symbol}-{interval}.parquet"


def download_klines(
    symbol: str = "ETHUSDT",
    interval: str = "1m",
    days: int = 30,
) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = parquet_path(symbol, interval)
    if path.exists():
        bars = pd.read_parquet(path)
        print(f"Using cached {path} ({len(bars):,} bars)", flush=True)
        return bars

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    step = INTERVAL_MS[interval]

    print(
        f"Downloading {symbol} {interval} from Binance "
        f"({start.date()} → {end.date()})...",
        flush=True,
    )

    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _fetch_batch(symbol, interval, cursor, end_ms)
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        nxt = last_open + step
        if nxt <= cursor:
            break
        cursor = nxt
        print(f"  {len(rows):,} bars", flush=True)
        time.sleep(0.15)

    bars = _to_frame(rows)
    bars = bars[bars.index <= pd.Timestamp(end)]
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    bars.to_parquet(path)
    print(f"Wrote {path} ({len(bars):,} bars)", flush=True)
    return bars


def _fetch_batch(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list:
    query = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
    )
    last_error: Exception | None = None
    for base in KLINE_URLS:
        req = urllib.request.Request(
            f"{base}?{query}",
            headers={"User-Agent": "nautilus-backtest/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = json.loads(response.read().decode())
            if isinstance(payload, dict) and payload.get("code"):
                raise RuntimeError(payload)
            return payload
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Binance klines failed: {last_error}") from last_error


def _to_frame(rows: list[list]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    # Nautilus bar timestamps are the close of the bar.
    index = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    return frame.loc[:, ["open", "high", "low", "close", "volume"]].set_index(index)
