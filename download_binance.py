"""Download and cache Binance 1-minute data (shortcut for ``python -m backtester download``).

    uv run python download_binance.py BTCUSDT ETHUSDT --start 2021-01-01
    uv run python download_binance.py ETHUSDT --markets spot,perp

Data comes from data.binance.vision archives (REST for the last hours),
lands in ``data/1m/<market>/<SYMBOL>/`` and is resampled to any bar size
on load. No API key required. See ``backtester/data.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtester.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["download", *sys.argv[1:]]))
