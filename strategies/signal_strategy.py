"""Execute any Signal in Nautilus.

Turns a target exposure into orders:

- Sizing is on mark-to-market equity (cash + unrealised PnL), so gains
  compound and losses shrink size.
- One market order moves the position straight to target, flips
  included, so there is no flatten-then-wait-a-bar gap.
- Small rebalances below ``rebalance_threshold`` (fraction of equity) are
  skipped to stop vol-targeting signals from churning fees.
- ``signal.stop_px`` becomes a reduce-only STOP_MARKET at the venue. It
  fills intrabar at the stop price (or the gap open), not at the close.
"""

from __future__ import annotations

from typing import Any

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.enums import bar_aggregation_to_str
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

from strategies.signal_base import BarData
from strategies.signal_base import Signal

_AGG_MINUTES = {"MINUTE": 1, "HOUR": 60, "DAY": 1440}


class SignalStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    signal: str
    params: dict[str, Any] = {}
    leverage: float = 1.0
    rebalance_threshold: float = 0.05
    use_stops: bool = True


def bar_minutes(bar_type: BarType) -> int:
    spec = bar_type.spec
    return int(spec.step) * _AGG_MINUTES[bar_aggregation_to_str(spec.aggregation)]


class SignalStrategy(Strategy):
    def __init__(self, config: SignalStrategyConfig) -> None:
        super().__init__(config)
        from strategies.catalog import SIGNALS

        cls = SIGNALS[config.signal]
        self.signal: Signal = cls(bar_minutes(config.bar_type), dict(config.params))
        self.instrument = None
        self._stop_order = None
        self.n_stops = 0

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        self.subscribe_bars(self.config.bar_type)

    # ------------------------------------------------------------------ #

    def on_bar(self, bar: Bar) -> None:
        self._check_stop_filled()
        close = bar.close.as_double()
        target = self.signal.on_bar(
            BarData(
                bar.ts_event,
                bar.open.as_double(),
                bar.high.as_double(),
                bar.low.as_double(),
                close,
                bar.volume.as_double(),
            )
        )
        if target is not None:
            self._rebalance(target, close)
        if self.config.use_stops:
            self._sync_stop(close)

    def on_order_filled(self, event: OrderFilled) -> None:
        if self._stop_order is not None and event.client_order_id == self._stop_order.client_order_id:
            self._check_stop_filled()

    # ------------------------------------------------------------------ #

    def _equity(self) -> float | None:
        account = self.portfolio.account(self.config.instrument_id.venue)
        if account is None:
            return None
        equity = self.portfolio.equity(self.config.instrument_id.venue).get(USDT)
        if equity is None:
            balance = account.balance_total(USDT)
            return None if balance is None else balance.as_double()
        return equity.as_double()

    def _net(self) -> float:
        return float(self.portfolio.net_position(self.config.instrument_id))

    def _rebalance(self, target: float, price: float) -> None:
        equity = self._equity()
        if equity is None or equity <= 0 or price <= 0:
            return
        current = self._net()
        if target == 0.0:
            if current != 0.0:
                self._cancel_stop()
                self.close_all_positions(self.config.instrument_id)
            return
        desired = target * self.config.leverage * equity / price
        delta = desired - current
        flips = current != 0.0 and (desired > 0) != (current > 0)
        opening = current == 0.0
        small = abs(delta) * price < self.config.rebalance_threshold * equity
        if small and not flips and not opening:
            return
        instrument = self.instrument
        try:
            qty = instrument.make_qty(abs(delta))
        except ValueError:
            return
        if qty.as_double() <= 0:
            return
        min_notional = instrument.min_notional.as_double() if instrument.min_notional else 0.0
        if qty.as_double() * price < min_notional:
            return
        self._cancel_stop()
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        self.submit_order(self.order_factory.market(self.config.instrument_id, side, qty))

    # -- protective stop ------------------------------------------------ #

    def _check_stop_filled(self) -> None:
        order = self._stop_order
        if order is None:
            return
        if order.is_closed:
            self._stop_order = None
            if order.filled_qty.as_double() > 0:
                self.n_stops += 1
                self.signal.on_stopped()

    def _cancel_stop(self) -> None:
        order = self._stop_order
        self._stop_order = None
        if order is not None and order.is_open:
            self.cancel_order(order)

    def _sync_stop(self, close: float) -> None:
        net = self._net()
        stop = self.signal.stop_px
        if net == 0.0 or stop is None or stop <= 0:
            self._cancel_stop()
            return
        long = net > 0
        # A stop already through the market means "get out now".
        if (long and stop >= close) or (not long and stop <= close):
            self._cancel_stop()
            self.close_all_positions(self.config.instrument_id)
            self.signal.on_stopped()
            self.n_stops += 1
            return
        instrument = self.instrument
        trigger = instrument.make_price(stop)
        qty = instrument.make_qty(abs(net))
        order = self._stop_order
        if (
            order is not None
            and order.is_open
            and order.trigger_price == trigger
            and order.quantity == qty
        ):
            return
        self._cancel_stop()
        if qty.as_double() <= 0:
            return
        self._stop_order = self.order_factory.stop_market(
            self.config.instrument_id,
            OrderSide.SELL if long else OrderSide.BUY,
            qty,
            trigger_price=trigger,
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
        )
        self.submit_order(self._stop_order)

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)

    def on_reset(self) -> None:
        self._stop_order = None
        self.n_stops = 0
