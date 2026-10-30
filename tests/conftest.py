from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def synthetic_1m(days: int = 120, seed: int = 0, drift: float = 0.0, vol: float = 0.0008) -> pd.DataFrame:
    """Random-walk 1m OHLCV indexed by open time, with regime-y drift swings."""
    rng = np.random.default_rng(seed)
    n = days * 1440
    swings = drift + 0.00004 * np.sin(np.arange(n) / (1440 * 9.0))
    rets = rng.normal(swings, vol, n)
    close = 20_000 * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]]
    wiggle = np.abs(rng.normal(0, vol / 2, n))
    high = np.maximum(open_, close) * (1 + wiggle)
    low = np.minimum(open_, close) * (1 - wiggle)
    idx = pd.date_range("2023-01-01", periods=n, freq="1min", tz="UTC", name="open_time")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 5.0}, index=idx)


@pytest.fixture(scope="session")
def bars_1m() -> pd.DataFrame:
    return synthetic_1m()


@pytest.fixture(autouse=True)
def offline_instruments(monkeypatch, tmp_path):
    """Never hit Binance from tests: instrument specs are inferred, cache is temporary."""
    import backtester.instruments as inst

    monkeypatch.setattr(inst, "_http_get", lambda *a, **k: None)
    monkeypatch.setattr(inst, "_CACHE", tmp_path / "instruments.json")
