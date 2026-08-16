"""Always-in dual moving-average trend follower.

Long while the fast SMA is above the slow SMA, short while it is below.
Fully invested on each entry. The cross is the exit — no ATR stop.

20/50 daily is a standard trend rule, not a fitted grid. It can still lose
to the S&P 500 on other windows (50/200 did on this one).
"""

from __future__ import annotations

from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators import SimpleMovingAverage
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy


class DualMAConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    fast: int = 20
    slow: int = 50
    leverage: float = 1.0


class DualMATrend(Strategy):
    def __init__(self, config: DualMAConfig) -> None:
        super().__init__(config)
        self.fast_sma = SimpleMovingAverage(config.fast)
        self.slow_sma = SimpleMovingAverage(config.slow)
        self.pending: str | None = None

    def on_start(self) -> None:
        self.register_indicator_for_bars(self.config.bar_type, self.fast_sma)
        self.register_indicator_for_bars(self.config.bar_type, self.slow_sma)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self.indicators_initialized():
            return
        if bar.is_single_price():
            return

        instrument_id = self.config.instrument_id
        want = "long" if self.fast_sma.value > self.slow_sma.value else "short"

        if self.pending and self.portfolio.is_flat(instrument_id):
            self._enter(bar, self.pending)
            self.pending = None
            return

        if want == "long" and self.portfolio.is_net_short(instrument_id):
            self._flatten()
            self.pending = "long"
        elif want == "short" and self.portfolio.is_net_long(instrument_id):
            self._flatten()
            self.pending = "short"
        elif self.portfolio.is_flat(instrument_id):
            self._enter(bar, want)

    def _enter(self, bar: Bar, side: str) -> None:
        instrument = self.cache.instrument(self.config.instrument_id)
        account = self.portfolio.account(instrument.id.venue)
        if account is None:
            return
        balance = account.balance_total(USDT)
        if balance is None:
            return
        price = bar.close.as_double()
        qty = (balance.as_double() * self.config.leverage) / price
        try:
            quantity = instrument.make_qty(qty)
        except ValueError:
            return
        order_side = OrderSide.BUY if side == "long" else OrderSide.SELL
        self.submit_order(
            self.order_factory.market(self.config.instrument_id, order_side, quantity)
        )

    def _flatten(self) -> None:
        self.close_all_positions(self.config.instrument_id)

    def on_stop(self) -> None:
        self.close_all_positions(self.config.instrument_id)

    def on_reset(self) -> None:
        self.pending = None
