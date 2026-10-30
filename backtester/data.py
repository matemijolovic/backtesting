"""Market data: Binance 1-minute archives -> any bar size.

Every timeframe is resampled from the same 1-minute base. That keeps
5m, 1h and 1d results comparable (same source, same gaps) and lets you
test non-standard bar sizes (7m, 90m, 3h) and shifted bar boundaries,
both of which are cheap overfitting checks.

Source: https://data.binance.vision monthly zips, daily zips for months
not yet published, REST API for anything still missing. No API key.

Cache layout::

    data/1m/<market>/<SYMBOL>/<YYYY-MM>.parquet      complete month
    data/1m/<market>/<SYMBOL>/daily/<YYYY-MM-DD>.parquet
    data/funding/<SYMBOL>.parquet
"""

from __future__ import annotations

import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARCHIVE = "https://data.binance.vision/data"
REST_KLINES = {
    "spot": (
        "https://api.binance.com/api/v3/klines",
        "https://data-api.binance.vision/api/v3/klines",
    ),
    "perp": ("https://fapi.binance.com/fapi/v1/klines",),
}
REST_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
OHLCV = ["open", "high", "low", "close", "volume"]
MARKETS = ("spot", "perp")
USER_AGENT = {"User-Agent": "nautilus-backtest/2.0"}

_INTERVAL_RE = re.compile(r"^(\d+)(m|h|d)$")
_UNIT_MINUTES = {"m": 1, "h": 60, "d": 1440}

# Bar sizes offered by default in the UI / CLI. Any "<n>m|h|d" works.
COMMON_INTERVALS = ("5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d")


# --------------------------------------------------------------------------- #
# Intervals
# --------------------------------------------------------------------------- #


def interval_minutes(interval: str) -> int:
    """'15m' -> 15, '4h' -> 240, '1d' -> 1440."""
    match = _INTERVAL_RE.match(interval.strip().lower())
    if not match:
        raise ValueError(f"Bad interval {interval!r}; use e.g. 5m, 90m, 4h, 1d")
    n, unit = int(match.group(1)), match.group(2)
    if n <= 0:
        raise ValueError(f"Bad interval {interval!r}")
    return n * _UNIT_MINUTES[unit]


def interval_nautilus(interval: str) -> str:
    """'15m' -> '15-MINUTE', '4h' -> '4-HOUR', '1d' -> '1-DAY'."""
    match = _INTERVAL_RE.match(interval.strip().lower())
    if not match:
        raise ValueError(f"Bad interval {interval!r}")
    unit = {"m": "MINUTE", "h": "HOUR", "d": "DAY"}[match.group(2)]
    return f"{int(match.group(1))}-{unit}"


# --------------------------------------------------------------------------- #
# 1-minute base data
# --------------------------------------------------------------------------- #


def _utc(ts) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    return out.tz_localize("UTC") if out.tzinfo is None else out.tz_convert("UTC")


def _symbol_dir(symbol: str, market: str) -> Path:
    if market not in MARKETS:
        raise ValueError(f"market must be one of {MARKETS}")
    return DATA_DIR / "1m" / market / symbol.upper()


def _archive_url(symbol: str, market: str, period: str, kind: str) -> str:
    root = "spot" if market == "spot" else "futures/um"
    return f"{ARCHIVE}/{root}/{kind}/klines/{symbol}/1m/{symbol}-1m-{period}.zip"


def _http_get(url: str, timeout: int = 60, retries: int = 6) -> bytes | None:
    """GET bytes; None on 404. Retries transient errors."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=USER_AGENT)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
        time.sleep(min(2.0**attempt, 20.0))
    raise RuntimeError(f"GET {url} failed: {last}") from last


def _parse_kline_csv(raw: bytes) -> pd.DataFrame:
    """Binance kline CSV (with or without header) -> OHLCV indexed by open time."""
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open(zf.namelist()[0]) as fh:
            frame = pd.read_csv(fh, header=None, usecols=range(6), dtype=str)
    frame = frame[pd.to_numeric(frame[0], errors="coerce").notna()]
    return _klines_to_frame(frame.to_numpy().tolist())


def _klines_to_frame(rows: list) -> pd.DataFrame:
    if not rows:
        return _empty()
    arr = np.asarray([r[:6] for r in rows], dtype=object)
    open_time = arr[:, 0].astype(np.int64)
    # Spot archives switched to microsecond timestamps on 2025-01-01.
    unit = np.where(open_time > 10**14, 1_000, 1_000_000)
    index = pd.DatetimeIndex(open_time * unit, tz="UTC")
    frame = pd.DataFrame(arr[:, 1:6].astype(np.float64), columns=OHLCV, index=index)
    frame.index.name = "open_time"
    return frame


def _empty() -> pd.DataFrame:
    frame = pd.DataFrame(columns=OHLCV, dtype=np.float64)
    frame.index = pd.DatetimeIndex([], tz="UTC", name="open_time")
    return frame


def _rest_klines(symbol: str, market: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """REST fallback for minutes not yet in the archives."""
    rows: list = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000) - 1
    while cursor <= end_ms:
        query = urllib.parse.urlencode(
            {"symbol": symbol, "interval": "1m", "startTime": cursor,
             "endTime": end_ms, "limit": 1000}
        )
        payload = None
        for base in REST_KLINES[market]:
            raw = _http_get(f"{base}?{query}", timeout=30)
            if raw is None:
                continue
            payload = json.loads(raw.decode())
            if isinstance(payload, dict):
                payload = None
                continue
            break
        if not payload:
            break
        rows.extend(payload)
        nxt = int(payload[-1][0]) + 60_000
        if nxt <= cursor:
            break
        cursor = nxt
        time.sleep(0.1)
    return _klines_to_frame(rows)


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    first = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    return list(pd.date_range(first, end, freq="MS", tz="UTC"))


def _fetch_month(symbol: str, market: str, month: pd.Timestamp, today: date) -> pd.DataFrame | None:
    """Cached month, or download it. None = month not complete yet."""
    sdir = _symbol_dir(symbol, market)
    path = sdir / f"{month:%Y-%m}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    month_end = (month + pd.offsets.MonthBegin(1)).date()
    if month_end > today:
        return None  # current month: handled by the daily path
    raw = _http_get(_archive_url(symbol, market, f"{month:%Y-%m}", "monthly"))
    if raw is None:
        # Monthly zips appear a few days after month end; old 404s mean the
        # symbol did not trade yet.
        if (today - month_end).days < 10:
            return None
        frame = _empty()
    else:
        frame = _parse_kline_csv(raw)
    sdir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return frame


def _fetch_day(symbol: str, market: str, day: date, today: date) -> pd.DataFrame:
    ddir = _symbol_dir(symbol, market) / "daily"
    path = ddir / f"{day:%Y-%m-%d}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    raw = _http_get(_archive_url(symbol, market, f"{day:%Y-%m-%d}", "daily"))
    if raw is None:
        start = pd.Timestamp(day, tz="UTC")
        frame = _rest_klines(symbol, market, start, start + pd.Timedelta(days=1))
    else:
        frame = _parse_kline_csv(raw)
    # Only cache days that are fully in the past and non-empty.
    if day < today and len(frame):
        ddir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
    return frame


def load_1m(
    symbol: str,
    start,
    end,
    *,
    market: str = "spot",
    workers: int = 6,
    progress=None,
) -> pd.DataFrame:
    """1-minute OHLCV for [start, end), indexed by bar open time (UTC).

    Downloads and caches whatever is missing. ``progress(done, total, msg)``
    is called as months arrive (used by the UI).
    """
    symbol = symbol.upper()
    start, end = _utc(start), _utc(end)
    now = pd.Timestamp.now(tz="UTC").floor("min")
    end = min(end, now)
    if end <= start:
        return _empty()
    today = now.date()

    months = _month_starts(start, end)
    total = len(months)
    frames: dict[pd.Timestamp, pd.DataFrame | None] = {}

    def month_job(m: pd.Timestamp):
        return m, _fetch_month(symbol, market, m, today)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (m, frame) in enumerate(pool.map(month_job, months), start=1):
            frames[m] = frame
            if progress:
                progress(i, total, f"{symbol} {market} {m:%Y-%m}")

    # Months without a monthly archive yet: daily zips, then REST for today.
    days: list[date] = []
    for m, frame in frames.items():
        if frame is not None:
            continue
        lo = max(m, start.floor("D"))
        hi = min(m + pd.offsets.MonthBegin(1), end)
        days.extend(d.date() for d in pd.date_range(lo, hi, freq="D", inclusive="left"))

    def day_job(d: date):
        return _fetch_day(symbol, market, d, today)

    pieces = [f for f in frames.values() if f is not None and len(f)]
    if days:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pieces.extend(f for f in pool.map(day_job, days) if len(f))

    if not pieces:
        return _empty()
    bars = pd.concat(pieces)
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    return bars[(bars.index >= start) & (bars.index < end)]


# --------------------------------------------------------------------------- #
# Resampling + cleaning
# --------------------------------------------------------------------------- #


def clean(bars: pd.DataFrame) -> pd.DataFrame:
    """Drop broken rows, repair OHLC ordering. Keeps gaps as gaps."""
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    bars = bars.replace([np.inf, -np.inf], np.nan).dropna(subset=OHLCV)
    bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(axis=1)]
    bars = bars.copy()
    bars["high"] = bars[["open", "high", "low", "close"]].max(axis=1)
    bars["low"] = bars[["open", "high", "low", "close"]].min(axis=1)
    bars["volume"] = bars["volume"].clip(lower=0)
    return bars


def resample(bars_1m: pd.DataFrame, interval: str, offset: str | None = None) -> pd.DataFrame:
    """Aggregate 1m bars (open-time index) to ``interval``.

    ``offset`` shifts bar boundaries, e.g. ``"30m"`` turns 00:00/01:00 hourly
    bars into 00:30/01:30. Returned index is the bar *open* time. The
    trailing bucket is dropped if it is not complete.
    """
    minutes = interval_minutes(interval)
    if bars_1m.empty:
        return bars_1m.copy()
    if minutes == 1 and not offset:
        return bars_1m.copy()
    rule = f"{minutes}min"
    agg = bars_1m.resample(rule, offset=offset, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    agg = agg.dropna(subset=["open", "close"])
    last_open = bars_1m.index[-1]
    if len(agg) and agg.index[-1] + pd.Timedelta(minutes=minutes) > last_open + pd.Timedelta(minutes=1):
        agg = agg.iloc[:-1]
    agg.index.name = "open_time"
    return agg


def with_close_index(bars: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Re-index from bar open time to bar close time (Nautilus convention)."""
    out = bars.copy()
    out.index = out.index + pd.Timedelta(minutes=interval_minutes(interval))
    out.index.name = "ts"
    return out


def load_bars(
    symbol: str,
    interval: str,
    start,
    end,
    *,
    market: str = "spot",
    offset: str | None = None,
    progress=None,
) -> pd.DataFrame:
    """Clean OHLCV bars of any size, indexed by bar *close* time."""
    base = load_1m(symbol, start, end, market=market, progress=progress)
    bars = resample(clean(base), interval, offset)
    return with_close_index(bars, interval)


@dataclass(frozen=True)
class DataQuality:
    bars: int
    expected_minutes: int
    missing_minutes: int
    largest_gap: pd.Timedelta
    zero_volume_bars: int
    flat_bars: int

    @property
    def coverage(self) -> float:
        if self.expected_minutes == 0:
            return 0.0
        return 1.0 - self.missing_minutes / self.expected_minutes

    def as_dict(self) -> dict:
        return {
            "1m bars": self.bars,
            "coverage": f"{self.coverage:.3%}",
            "missing minutes": self.missing_minutes,
            "largest gap": str(self.largest_gap),
            "zero-volume bars": self.zero_volume_bars,
            "flat (O=H=L=C) bars": self.flat_bars,
        }


def quality(bars_1m: pd.DataFrame) -> DataQuality:
    if bars_1m.empty:
        return DataQuality(0, 0, 0, pd.Timedelta(0), 0, 0)
    idx = bars_1m.index
    expected = int((idx[-1] - idx[0]) / pd.Timedelta(minutes=1)) + 1
    gaps = idx.to_series().diff().dropna()
    flat = (bars_1m["high"] == bars_1m["low"]).sum()
    return DataQuality(
        bars=len(bars_1m),
        expected_minutes=expected,
        missing_minutes=max(expected - len(bars_1m), 0),
        largest_gap=gaps.max() if len(gaps) else pd.Timedelta(0),
        zero_volume_bars=int((bars_1m["volume"] <= 0).sum()),
        flat_bars=int(flat),
    )


# --------------------------------------------------------------------------- #
# Funding rates
# --------------------------------------------------------------------------- #


def load_funding(symbol: str, start, end) -> pd.DataFrame:
    """Settled USD-M funding: columns funding_time, funding_rate. Cached."""
    symbol = symbol.upper()
    start, end = _utc(start), _utc(end)
    path = DATA_DIR / "funding" / f"{symbol}.parquet"
    cached = pd.read_parquet(path) if path.exists() else None

    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if cached is None or cached.empty:
        spans.append((start, end))
    else:
        lo = _utc(cached["funding_time"].min())
        hi = _utc(cached["funding_time"].max())
        if start < lo - pd.Timedelta(hours=8):
            spans.append((start, lo))
        if end > hi + pd.Timedelta(hours=8):
            spans.append((hi + pd.Timedelta(milliseconds=1), end))

    rows: list[dict] = []
    for lo, hi in spans:
        cursor = int(lo.timestamp() * 1000)
        end_ms = int(hi.timestamp() * 1000)
        while cursor < end_ms:
            query = urllib.parse.urlencode(
                {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000}
            )
            raw = _http_get(f"{REST_FUNDING}?{query}", timeout=30)
            batch = json.loads(raw.decode()) if raw else []
            if not batch or isinstance(batch, dict):
                break
            rows.extend(batch)
            nxt = int(batch[-1]["fundingTime"]) + 1
            if nxt <= cursor:
                break
            cursor = nxt
            time.sleep(0.1)

    frames = [cached] if cached is not None else []
    if rows:
        fresh = pd.DataFrame(rows)
        frames.append(
            pd.DataFrame(
                {
                    "funding_time": pd.to_datetime(fresh["fundingTime"].astype(np.int64), unit="ms", utc=True),
                    "funding_rate": pd.to_numeric(fresh["fundingRate"]),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["funding_time", "funding_rate"])
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates("funding_time", keep="last").sort_values("funding_time")
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    mask = (frame["funding_time"] >= start) & (frame["funding_time"] < end)
    return frame.loc[mask].reset_index(drop=True)


def default_window(days: int) -> tuple[datetime, datetime]:
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    return end - timedelta(days=days), end
