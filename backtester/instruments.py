"""Nautilus instruments for any Binance USDT symbol.

Tick size, lot size and min notional come from Binance ``exchangeInfo``
(cached in ``data/instruments.json``). If that is unreachable they are
inferred from the price level, so offline runs still work.

Fees include slippage: ``taker_fee = fee_bps + slippage_bps``. Every fill
in these backtests is a market order at the bar price, so modelling
slippage as an extra per-fill cost is equivalent and simpler than a
tick-based fill model.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path

from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import CurrencyType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Currency
from nautilus_trader.model.objects import Money
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity

from backtester.data import DATA_DIR
from backtester.data import _http_get

VENUE = Venue("BINANCE")
EXCHANGE_INFO = {
    "spot": "https://api.binance.com/api/v3/exchangeInfo?symbol={symbol}",
    "perp": "https://fapi.binance.com/fapi/v1/exchangeInfo",
}
_CACHE = DATA_DIR / "instruments.json"


def _precision(step: str) -> int:
    d = Decimal(step).normalize()
    return max(-d.as_tuple().exponent, 0)


def _load_cache() -> dict:
    if _CACHE.exists():
        try:
            return json.loads(_CACHE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))


def symbol_spec(symbol: str, market: str = "spot", ref_price: float | None = None) -> dict:
    """{'base', 'quote', 'tick', 'step', 'min_notional'} for a Binance symbol."""
    symbol = symbol.upper()
    key = f"{market}:{symbol}"
    cache = _load_cache()
    if key in cache:
        return cache[key]
    spec = None
    try:
        raw = _http_get(EXCHANGE_INFO[market].format(symbol=symbol), timeout=15, retries=1)
        if raw:
            payload = json.loads(raw.decode())
            for item in payload.get("symbols", []):
                if item.get("symbol") != symbol:
                    continue
                filters = {f["filterType"]: f for f in item["filters"]}
                notional = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
                spec = {
                    "base": item["baseAsset"],
                    "quote": item["quoteAsset"],
                    "tick": filters["PRICE_FILTER"]["tickSize"],
                    "step": filters["LOT_SIZE"]["stepSize"],
                    "min_notional": str(notional.get("minNotional", notional.get("notional", "5"))),
                }
                break
    except (RuntimeError, ValueError, KeyError):
        spec = None
    if spec is None:
        spec = _infer_spec(symbol, ref_price)
    else:
        cache[key] = spec
        _save_cache(cache)
    return spec


def _infer_spec(symbol: str, ref_price: float | None) -> dict:
    quote = "USDT" if symbol.endswith("USDT") else symbol[-4:]
    base = symbol[: -len(quote)]
    px = ref_price or 100.0
    # ~5 significant digits of price, ~$1 of size granularity.
    price_dp = min(max(4 - int(math.floor(math.log10(px))), 0), 8)
    size_dp = min(max(int(math.ceil(math.log10(px))) , 0), 8)
    return {
        "base": base,
        "quote": quote,
        "tick": f"{10 ** -price_dp:.{price_dp}f}" if price_dp else "1",
        "step": f"{10 ** -size_dp:.{size_dp}f}" if size_dp else "1",
        "min_notional": "5",
    }


def _currency(code: str) -> Currency:
    if code == "USDT":
        return USDT
    try:
        return Currency.from_str(code)
    except (ValueError, KeyError):
        return Currency(code, 8, 0, code, CurrencyType.CRYPTO)


def make_spot(
    symbol: str,
    *,
    fee_bps: float = 10.0,
    slippage_bps: float = 0.0,
    ref_price: float | None = None,
) -> CurrencyPair:
    """Spot pair traded on a margin account (so it can be shorted)."""
    spec = symbol_spec(symbol, "spot", ref_price)
    if spec["quote"] != "USDT":
        raise ValueError(f"Only USDT-quoted symbols are supported, got {symbol}")
    price_dp, size_dp = _precision(spec["tick"]), _precision(spec["step"])
    fee = Decimal(str((fee_bps + slippage_bps) / 10_000))
    return CurrencyPair(
        instrument_id=InstrumentId(Symbol(symbol.upper()), VENUE),
        raw_symbol=Symbol(symbol.upper()),
        base_currency=_currency(spec["base"]),
        quote_currency=USDT,
        price_precision=price_dp,
        size_precision=size_dp,
        price_increment=Price(float(spec["tick"]), price_dp),
        size_increment=Quantity(float(spec["step"]), size_dp),
        lot_size=None,
        max_quantity=None,
        min_quantity=Quantity(float(spec["step"]), size_dp),
        max_notional=None,
        min_notional=Money(float(spec["min_notional"]), USDT),
        max_price=None,
        min_price=None,
        margin_init=Decimal(0),
        margin_maint=Decimal(0),
        maker_fee=fee,
        taker_fee=fee,
        ts_event=0,
        ts_init=0,
    )


def make_perp(
    symbol: str,
    *,
    fee_bps: float = 5.0,
    slippage_bps: float = 0.0,
    ref_price: float | None = None,
) -> CryptoPerpetual:
    """USD-M linear perpetual, 5% initial / 2.5% maintenance margin."""
    spec = symbol_spec(symbol, "perp", ref_price)
    price_dp, size_dp = _precision(spec["tick"]), _precision(spec["step"])
    fee = Decimal(str((fee_bps + slippage_bps) / 10_000))
    return CryptoPerpetual(
        instrument_id=InstrumentId(Symbol(f"{symbol.upper()}-PERP"), VENUE),
        raw_symbol=Symbol(symbol.upper()),
        base_currency=_currency(spec["base"]),
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=price_dp,
        size_precision=size_dp,
        price_increment=Price(float(spec["tick"]), price_dp),
        size_increment=Quantity(float(spec["step"]), size_dp),
        multiplier=Quantity.from_int(1),
        lot_size=Quantity.from_int(1),
        max_quantity=None,
        min_quantity=Quantity(float(spec["step"]), size_dp),
        max_notional=None,
        min_notional=Money(float(spec["min_notional"]), USDT),
        max_price=None,
        min_price=None,
        margin_init=Decimal("0.05"),
        margin_maint=Decimal("0.025"),
        maker_fee=fee,
        taker_fee=fee,
        ts_event=0,
        ts_init=0,
    )


def cache_path() -> Path:
    return _CACHE
