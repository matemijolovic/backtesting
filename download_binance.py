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
FAPI_KLINE_URLS = (
    "https://fapi.binance.com/fapi/v1/klines",
)
FAPI_FUNDING_URLS = (
    "https://fapi.binance.com/fapi/v1/fundingRate",
)
DATA_DIR = Path(__file__).resolve().parent / "data"
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}


def parquet_path(symbol: str, interval: str, market: str = "spot") -> Path:
    if market == "perp":
        return DATA_DIR / f"{symbol}-PERP-{interval}.parquet"
    return DATA_DIR / f"{symbol}-{interval}.parquet"


def funding_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol}-funding.parquet"


def download_klines(
    symbol: str = "ETHUSDT",
    interval: str = "1m",
    days: int = 30,
    *,
    market: str = "spot",
) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = parquet_path(symbol, interval, market)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    cached = pd.read_parquet(path) if path.exists() else None
    if cached is not None and _covers(cached.index.min(), start):
        print(f"Using cached {path} ({len(cached):,} bars)", flush=True)
        return cached

    end_ms = int(end.timestamp() * 1000)
    if cached is not None:
        end_ms = int(cached.index.min().timestamp() * 1000)
        print(
            f"Extending {path} back to {start.date()} "
            f"({len(cached):,} cached bars)...",
            flush=True,
        )
    else:
        label = "perp" if market == "perp" else "spot"
        print(
            f"Downloading {symbol} {interval} {label} from Binance "
            f"({start.date()} → {end.date()})...",
            flush=True,
        )

    start_ms = int(start.timestamp() * 1000)
    step = INTERVAL_MS[interval]
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _fetch_batch(symbol, interval, cursor, end_ms, market=market)
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

    bars = _to_frame(rows) if rows else pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
    )
    if cached is not None:
        bars = pd.concat([bars, cached])
    bars = bars[bars.index <= pd.Timestamp(end)]
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    bars.to_parquet(path)
    print(f"Wrote {path} ({len(bars):,} bars)", flush=True)
    return bars


def download_funding(
    symbol: str = "BTCUSDT",
    days: int = 730,
) -> pd.DataFrame:
    """Settled USD-M funding history. One row per payment."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = funding_path(symbol)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    cached = pd.read_parquet(path) if path.exists() else None
    if cached is not None and _covers(cached["funding_time"].min(), start):
        print(f"Using cached {path} ({len(cached):,} payments)", flush=True)
        return cached

    end_ms = int(end.timestamp() * 1000)
    if cached is not None:
        end_ms = int(pd.Timestamp(cached["funding_time"].min()).timestamp() * 1000)
        print(
            f"Extending {path} back to {start.date()} "
            f"({len(cached):,} cached payments)...",
            flush=True,
        )
    else:
        print(
            f"Downloading {symbol} funding from Binance "
            f"({start.date()} → {end.date()})...",
            flush=True,
        )

    start_ms = int(start.timestamp() * 1000)
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _fetch_funding(symbol, cursor, end_ms)
        if not batch:
            break
        rows.extend(batch)
        last_ms = int(batch[-1]["fundingTime"])
        nxt = last_ms + 1
        if nxt <= cursor:
            break
        cursor = nxt
        print(f"  {len(rows):,} payments", flush=True)
        time.sleep(0.15)

    frame = _funding_frame(rows) if rows else pd.DataFrame(
        columns=["funding_time", "funding_rate", "mark_price"],
    )
    if cached is not None:
        frame = pd.concat([frame, cached], ignore_index=True)
    frame = frame[frame["funding_time"] <= pd.Timestamp(end)]
    frame = frame.drop_duplicates("funding_time", keep="last").sort_values("funding_time")
    frame.to_parquet(path, index=False)
    print(f"Wrote {path} ({len(frame):,} payments)", flush=True)
    return frame


def _covers(have, want: datetime, slack_days: int = 2) -> bool:
    got = pd.Timestamp(have)
    if got.tzinfo is None:
        got = got.tz_localize("UTC")
    need = pd.Timestamp(want)
    if need.tzinfo is None:
        need = need.tz_localize("UTC")
    return got <= need + timedelta(days=slack_days)


def _funding_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["funding_time"] = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True)
    frame["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="raise")
    mark = frame["markPrice"] if "markPrice" in frame.columns else None
    if mark is not None:
        frame["mark_price"] = pd.to_numeric(mark, errors="coerce")
    else:
        frame["mark_price"] = pd.NA
    return frame.loc[:, ["funding_time", "funding_rate", "mark_price"]]


def _fetch_batch(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    market: str = "spot",
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
    bases = FAPI_KLINE_URLS if market == "perp" else KLINE_URLS
    last_error: Exception | None = None
    for base in bases:
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


def _fetch_funding(symbol: str, start_ms: int, end_ms: int) -> list:
    query = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
    )
    last_error: Exception | None = None
    for base in FAPI_FUNDING_URLS:
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
    raise RuntimeError(f"Binance funding failed: {last_error}") from last_error


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
