"""Delta-neutral Binance funding carry: long spot / short perp.

    uv sync
    uv run python run_funding.py                     # ETHUSDT 1h, last 5 years
    uv run python run_funding.py --symbol BTCUSDT --start 2023-01-01

Fees default to BNB-discount maker-style rates (spot 7.5 bps, perp 1.6 bps),
as in the original script. Any ``backtester run`` flag works.
"""

from __future__ import annotations

import sys
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtester.cli import main  # noqa: E402

DEFAULTS = {
    "--symbol": "ETHUSDT",
    "--interval": "1h",
    "--start": (datetime.now(timezone.utc) - timedelta(days=1825)).date().isoformat(),
    "--fee-bps": "7.5",
    "--perp-fee-bps": "1.6",
    "--slippage-bps": "0",
}

if __name__ == "__main__":
    args = sys.argv[1:]
    for flag, value in DEFAULTS.items():
        if flag not in args:
            args += [flag, value]
    raise SystemExit(main(["run", "funding_carry", *args]))
