"""Always-in Donchian trend follower.

Long when price closes above the prior N-bar high, short when it closes
below the prior N-bar low. Fully invested on each entry. ATR stop is a
catastrophe brake, not a tight scalp.

Designed to use the account (unlike the 0.75% risk regime bot). Still not
a guarantee it beats the S&P 500.
"""

from __future__ import annotations

from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators import AverageTrueRange
from nautilus_trader.indicators import DonchianChannel
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy


class DonchianConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    channel: int = 20
    atr_period: int = 20
    stop_atr: float = 5.0
    leverage: float = 1.0


class DonchianTrend(Strategy):
    def __init__(self, config: DonchianConfig) -> None:
        super().__init__(config)
        self.channel = DonchianChannel(config.channel)
        self.atr = AverageTrueRange(config.atr_period)
        self.pending: str | None = None
        self.stop_px: float | None = None
        self.extreme: float | None = None
        self.prev_upper = 0.0
        self.prev_lower = 0.0

    def on_start(self) -> None:
        self.register_indicator_for_bars(self.config.bar_type, self.channel)
        self.register_indicator_for_bars(self.config.bar_type, self.atr)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self.indicators_initialized() or self.prev_upper <= 0:
            self._remember()
            return
        if bar.is_single_price() or self.atr.value <= 0:
            self._remember()
            return

        instrument_id = self.config.instrument_id
        close = bar.close.as_double()

        if self.pending and self.portfolio.is_flat(instrument_id):
            self._enter(bar, self.pending)
            self.pending = None
            self._remember()
            return

        if not self.portfolio.is_flat(instrument_id) and self._stop_hit(bar):
            self._flatten()
            self.pending = None
            self._remember()
            return

        if close > self.prev_upper:
            if self.portfolio.is_net_short(instrument_id):
                self._flatten()
                self.pending = "long"
            elif self.portfolio.is_flat(instrument_id):
                self._enter(bar, "long")
        elif close < self.prev_lower:
            if self.portfolio.is_net_long(instrument_id):
                self._flatten()
                self.pending = "short"
            elif self.portfolio.is_flat(instrument_id):
                self._enter(bar, "short")

        if not self.portfolio.is_flat(instrument_id):
            self._ratchet(bar)

        self._remember()

    def _enter(self, bar: Bar, side: str) -> None:
        instrument = self.cache.instrument(self.config.instrument_id)
        account = self.portfolio.account(instrument.id.venue)
        if account is None:
            return
        balance = account.balance_total(USDT)
        if balance is None:
            return
        price = bar.close.as_double()
        equity = balance.as_double()
        qty = (equity * self.config.leverage) / price
        try:
            quantity = instrument.make_qty(qty)
        except ValueError:
            return
        order_side = OrderSide.BUY if side == "long" else OrderSide.SELL
        self.submit_order(
            self.order_factory.market(self.config.instrument_id, order_side, quantity)
        )
        if side == "long":
            self.extreme = bar.high.as_double()
            self.stop_px = price - self.config.stop_atr * self.atr.value
        else:
            self.extreme = bar.low.as_double()
            self.stop_px = price + self.config.stop_atr * self.atr.value
        self.stop_px = float(instrument.make_price(self.stop_px))

    def _stop_hit(self, bar: Bar) -> bool:
        if self.stop_px is None:
            return False
        if self.portfolio.is_net_long(self.config.instrument_id):
            return bar.low.as_double() <= self.stop_px
        return bar.high.as_double() >= self.stop_px

    def _ratchet(self, bar: Bar) -> None:
        if self.stop_px is None or self.extreme is None:
            return
        instrument = self.cache.instrument(self.config.instrument_id)
        offset = self.config.stop_atr * self.atr.value
        if self.portfolio.is_net_long(self.config.instrument_id):
            self.extreme = max(self.extreme, bar.high.as_double())
            self.stop_px = max(self.stop_px, self.extreme - offset)
        else:
            self.extreme = min(self.extreme, bar.low.as_double())
            self.stop_px = min(self.stop_px, self.extreme + offset)
        self.stop_px = float(instrument.make_price(self.stop_px))

    def _flatten(self) -> None:
        self.close_all_positions(self.config.instrument_id)
        self.stop_px = None
        self.extreme = None

    def _remember(self) -> None:
        self.prev_upper = self.channel.upper
        self.prev_lower = self.channel.lower

    def on_stop(self) -> None:
        self.close_all_positions(self.config.instrument_id)

    def on_reset(self) -> None:
        self.pending = None
        self.stop_px = None
        self.extreme = None
        self.prev_upper = 0.0
        self.prev_lower = 0.0
