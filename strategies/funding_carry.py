"""Delta-neutral Binance funding carry (cash-and-carry).

When perpetual funding is positive, longs pay shorts. The hedge is long
spot and short the same notional of perp. When funding is deeply negative,
flip both legs. Price delta nets out; PnL is funding minus fees minus
basis change.

The signal is the last *settled* rate, used for the *next* payment. That
avoids reading the rate you are about to collect. Entries are fast;
exits wait for several consecutive dead/negative prints so round-trip
spot fees do not chew up a 0.5 bp 8h premium.
"""

from __future__ import annotations

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import FundingRateUpdate
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Money
from nautilus_trader.trading.strategy import Strategy


class FundingCarryConfig(StrategyConfig, frozen=True):
    spot_id: InstrumentId
    perp_id: InstrumentId
    spot_bar_type: BarType
    perp_bar_type: BarType
    enter_rate: float = 0.00002
    exit_rate: float = 0.0
    flip_rate: float = 0.00015
    enter_confirm: int = 1
    exit_confirm: int = 9
    notional_frac: float = 0.40
    min_basis: float = -0.0005


class FundingCarry(Strategy):
    def __init__(self, config: FundingCarryConfig) -> None:
        super().__init__(config)
        self.exchange = None
        self.last_rate: float | None = None
        self.signal: str | None = None
        self.want: str | None = None
        self.pending: str | None = None
        self._pending_signal: str | None = None
        self._pending_count = 0
        self.funding_pnl = 0.0
        self.n_settlements = 0
        self.n_paid = 0
        self.n_received = 0

    def bind_exchange(self, exchange) -> None:
        self.exchange = exchange

    def on_start(self) -> None:
        self.subscribe_bars(self.config.spot_bar_type)
        self.subscribe_bars(self.config.perp_bar_type)
        self.subscribe_funding_rates(self.config.perp_id)

    def on_bar(self, bar: Bar) -> None:
        if bar.is_single_price():
            return
        if bar.bar_type != self.config.perp_bar_type:
            return
        if self.cache.bar(self.config.spot_bar_type) is None:
            return
        if self._orders_working():
            return

        if self.pending is not None and self._both_flat():
            if self.pending == "flat":
                self.pending = None
                return
            self._enter(self.pending)
            self.pending = None
            return

        want = self.signal
        if want == self.want and not self._both_flat():
            return
        if want is None:
            if not self._both_flat():
                self._flatten()
                self.pending = "flat"
            self.want = None
            return
        if not self._both_flat():
            self._flatten()
            self.pending = want
            self.want = want
            return
        self._enter(want)
        self.want = want

    def on_funding_rate(self, funding_rate: FundingRateUpdate) -> None:
        rate = float(funding_rate.rate)
        self.last_rate = rate
        implied = self._implied_side(rate)
        if implied == self.signal:
            self._pending_signal = implied
            self._pending_count = 0
        elif implied == self._pending_signal:
            self._pending_count += 1
        else:
            self._pending_signal = implied
            self._pending_count = 1
        need = (
            self.config.enter_confirm
            if self.signal is None
            else self.config.exit_confirm
        )
        if implied != self.signal and self._pending_count >= need:
            self.signal = implied
            self._pending_count = 0
        self._settle(rate)

    def _implied_side(self, rate: float) -> str | None:
        if rate >= self.config.enter_rate:
            return "short_perp"
        if rate <= -self.config.flip_rate:
            return "long_perp"
        if self.signal == "short_perp" and rate > self.config.exit_rate:
            return "short_perp"
        if self.signal == "long_perp" and rate < -self.config.exit_rate:
            return "long_perp"
        return None

    def _basis_ok(self, side: str) -> bool:
        spot_bar = self.cache.bar(self.config.spot_bar_type)
        perp_bar = self.cache.bar(self.config.perp_bar_type)
        if spot_bar is None or perp_bar is None:
            return False
        spot = spot_bar.close.as_double()
        perp = perp_bar.close.as_double()
        if spot <= 0:
            return False
        basis = (perp - spot) / spot
        if side == "short_perp":
            return basis >= self.config.min_basis
        return basis <= -self.config.min_basis

    def _enter(self, side: str) -> None:
        if not self._basis_ok(side):
            return
        qty = self._hedge_qty()
        if qty is None:
            return
        spot_qty, perp_qty = qty
        if side == "short_perp":
            self.submit_order(
                self.order_factory.market(self.config.spot_id, OrderSide.BUY, spot_qty)
            )
            self.submit_order(
                self.order_factory.market(self.config.perp_id, OrderSide.SELL, perp_qty)
            )
        else:
            self.submit_order(
                self.order_factory.market(self.config.spot_id, OrderSide.SELL, spot_qty)
            )
            self.submit_order(
                self.order_factory.market(self.config.perp_id, OrderSide.BUY, perp_qty)
            )

    def _hedge_qty(self):
        spot = self.cache.instrument(self.config.spot_id)
        perp = self.cache.instrument(self.config.perp_id)
        account = self.portfolio.account(spot.id.venue)
        spot_bar = self.cache.bar(self.config.spot_bar_type)
        if account is None or spot_bar is None:
            return None
        balance = account.balance_total(USDT)
        if balance is None:
            return None
        price = spot_bar.close.as_double()
        if price <= 0:
            return None
        raw = (balance.as_double() * self.config.notional_frac) / price
        try:
            perp_qty = perp.make_qty(raw)
            spot_qty = spot.make_qty(perp_qty.as_double())
        except ValueError:
            return None
        if spot_qty.as_double() <= 0 or perp_qty.as_double() <= 0:
            return None
        return spot_qty, perp_qty

    def _settle(self, rate: float) -> None:
        if self.portfolio.is_flat(self.config.perp_id):
            return
        net = float(self.portfolio.net_position(self.config.perp_id))
        if net == 0.0:
            return
        mark = self._mark_price()
        if mark is None:
            return
        payment = -net * mark * rate
        self.funding_pnl += payment
        self.n_settlements += 1
        if payment >= 0:
            self.n_received += 1
        else:
            self.n_paid += 1
        if self.exchange is None or abs(payment) < 1e-8:
            return
        self.exchange.adjust_account(Money(payment, USDT))

    def _mark_price(self) -> float | None:
        bar = self.cache.bar(self.config.perp_bar_type)
        if bar is None:
            return None
        px = bar.close.as_double()
        return px if px > 0 else None

    def _flatten(self) -> None:
        self.close_all_positions(self.config.spot_id)
        self.close_all_positions(self.config.perp_id)

    def _both_flat(self) -> bool:
        return self.portfolio.is_flat(self.config.spot_id) and self.portfolio.is_flat(
            self.config.perp_id
        )

    def _orders_working(self) -> bool:
        return bool(
            self.cache.orders_open(instrument_id=self.config.spot_id)
            or self.cache.orders_open(instrument_id=self.config.perp_id)
            or self.cache.orders_inflight(instrument_id=self.config.spot_id)
            or self.cache.orders_inflight(instrument_id=self.config.perp_id)
        )

    def on_stop(self) -> None:
        self._flatten()

    def on_reset(self) -> None:
        self.last_rate = None
        self.signal = None
        self.want = None
        self.pending = None
        self._pending_signal = None
        self._pending_count = 0
        self.funding_pnl = 0.0
        self.n_settlements = 0
        self.n_paid = 0
        self.n_received = 0
