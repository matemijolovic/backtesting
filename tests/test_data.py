import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from backtester.data import _parse_kline_csv
from backtester.data import clean
from backtester.data import interval_minutes
from backtester.data import interval_nautilus
from backtester.data import quality
from backtester.data import resample
from backtester.data import with_close_index


@pytest.mark.parametrize(
    "text,minutes,naut",
    [("1m", 1, "1-MINUTE"), ("15m", 15, "15-MINUTE"), ("90m", 90, "90-MINUTE"), ("4h", 240, "4-HOUR"), ("1d", 1440, "1-DAY")],
)
def test_intervals(text, minutes, naut):
    assert interval_minutes(text) == minutes
    assert interval_nautilus(text) == naut


@pytest.mark.parametrize("bad", ["", "4", "h4", "0m", "1w"])
def test_bad_interval(bad):
    with pytest.raises(ValueError):
        interval_minutes(bad)


def _zip(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("k.csv", text)
    return buf.getvalue()


def test_parse_ms_and_us_and_header():
    ms = "1704067200000,1,2,0.5,1.5,10,1704067259999,0,0,0,0,0\n"
    us = "1735689600000000,1,2,0.5,1.5,10,1735689659999999,0,0,0,0,0\n"
    header = "open_time,open,high,low,close,volume,close_time,qv,count,tb,tbq,ignore\n"
    a = _parse_kline_csv(_zip(ms))
    b = _parse_kline_csv(_zip(us))
    c = _parse_kline_csv(_zip(header + ms))
    assert a.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert b.index[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert c.equals(a)
    assert list(a.columns) == ["open", "high", "low", "close", "volume"]


def test_resample_matches_manual(bars_1m):
    h4 = resample(bars_1m, "4h")
    first = bars_1m.iloc[:240]
    row = h4.iloc[0]
    assert row["open"] == first["open"].iloc[0]
    assert row["close"] == first["close"].iloc[-1]
    assert row["high"] == first["high"].max()
    assert row["low"] == first["low"].min()
    assert row["volume"] == pytest.approx(first["volume"].sum())
    assert len(h4) == len(bars_1m) // 240


def test_resample_offset_and_partial_tail(bars_1m):
    cut = bars_1m.iloc[: 240 * 3 + 100]  # 3 full 4h bars + a partial one
    plain = resample(cut, "4h")
    assert len(plain) == 3
    shifted = resample(cut, "4h", offset="1h")
    assert (shifted.index.hour % 4 == 1).all()
    assert shifted.index[0] == pd.Timestamp("2022-12-31 21:00", tz="UTC")


def test_close_index_and_clean():
    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    frame = pd.DataFrame({"open": [1, 2, -1], "high": [0.5, 3, 1], "low": [1, 1, 1], "close": [1, 2, 1], "volume": [1, -1, 1]}, index=idx, dtype=float)
    cleaned = clean(frame)
    assert len(cleaned) == 2  # negative price dropped
    assert (cleaned["high"] >= cleaned[["open", "close"]].max(axis=1)).all()
    assert (cleaned["volume"] >= 0).all()
    closed = with_close_index(cleaned, "1h")
    assert closed.index[0] == pd.Timestamp("2024-01-01 01:00", tz="UTC")


def test_quality_counts_gaps(bars_1m):
    holed = bars_1m.drop(bars_1m.index[100:130])
    q = quality(holed)
    assert q.missing_minutes == 30
    assert q.largest_gap == pd.Timedelta(minutes=31)
    assert 0.99 < q.coverage < 1.0
