"""Run one backtest from the command line (shortcut for ``python -m backtester run``).

    uv sync
    uv run python run_backtest.py                                  # ensemble, BTCUSDT 1h since 2022
    uv run python run_backtest.py trend_ensemble --symbol ETHUSDT --interval 4h
    uv run python run_backtest.py dual_ma --interval 5m -p fast=20 -p slow=50 --no-scale

See ``uv run python -m backtester list`` for strategies and parameters, and
``uv run streamlit run app.py`` for the UI.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtester.cli import main  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        args = ["ensemble", *args]
    raise SystemExit(main(["run", *args]))
