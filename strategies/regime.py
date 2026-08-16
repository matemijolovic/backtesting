"""Regime-switching ETH strategy.

Kaufman efficiency ratio splits the tape into trend vs chop:

- Trend: Donchian breakout, chandelier ATR trail, channel exit.
- Chop: fade Keltner extremes with RSI confirmation, mean-revert to the
  middle band, tighter stop, time stop.

Position size is a fixed fraction of equity over ATR distance. Stops are
checked against the bar low/high. After a stop-out, entries pause.

This is a systematic baseline, not a promise of alpha.
"""

from __future__ import annotations

from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators import AverageTrueRange
from nautilus_trader.indicators import DonchianChannel
from nautilus_trader.indicators import EfficiencyRatio
from nautilus_trader.indicators import KeltnerChannel
from nautilus_trader.indicators import RelativeStrengthIndex
from nautilus_trader.indicators import SimpleMovingAverage
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy


class RegimeConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    donchian_entry: int = 72
    donchian_exit: int = 24
    atr_period: int = 24
    er_period: int = 24
    keltner_period: int = 24
    keltner_k: float = 1.5
    rsi_period: int = 14
    vol_period: int = 24
    er_trend: float = 0.35
    er_chop: float = 0.25
    rsi_buy: float = 0.30
    rsi_sell: float = 0.70
    stop_atr: float = 2.0
    trail_atr: float = 3.0
    fade_stop_atr: float = 1.25
    fade_max_bars: int = 12
    cooldown_bars: int = 6
    risk_pct: float = 0.0075
    max_equity_frac: float = 0.25


class RegimeStrategy(Strategy):
    def __init__(self, config: RegimeConfig) -> None:
        super().__init__(config)
        self.donchian_entry = DonchianChannel(config.donchian_entry)
        self.donchian_exit = DonchianChannel(config.donchian_exit)
        self.atr = AverageTrueRange(config.atr_period)
        self.er = EfficiencyRatio(config.er_period)
        self.keltner = KeltnerChannel(config.keltner_period, config.keltner_k)
        self.rsi = RelativeStrengthIndex(config.rsi_period)
        self.vol_sma = SimpleMovingAverage(config.vol_period)

        self.kind: str | None = None
        self.stop_px: float | None = None
        self.extreme: float | None = None
        self.bars_held = 0
        self.cooldown = 0
        self.prev_entry_upper = 0.0
        self.prev_entry_lower = 0.0
        self.prev_exit_upper = 0.0
        self.prev_exit_lower = 0.0

    def on_start(self) -> None:
        for indicator in (
            self.donchian_entry,
            self.donchian_exit,
            self.atr,
            self.er,
            self.keltner,
            self.rsi,
        ):
            self.register_indicator_for_bars(self.config.bar_type, indicator)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        self.vol_sma.update_raw(bar.volume.as_double())
        if not self.indicators_initialized() or not self.vol_sma.initialized:
            self._remember_channels()
            return
        if bar.is_single_price() or self.atr.value <= 0:
            self._remember_channels()
            return

        if self.cooldown > 0:
            self.cooldown -= 1

        if not self.portfolio.is_flat(self.config.instrument_id):
            self.bars_held += 1
            if self._should_exit(bar):
                stopped = self._stop_hit(bar)
                self._flatten()
                if stopped:
                    self.cooldown = self.config.cooldown_bars
                self._remember_channels()
                return
        elif self.cooldown == 0 and self.prev_entry_upper > 0:
            self._try_enter(bar)

        self._remember_channels()

    def _try_enter(self, bar: Bar) -> None:
        close = bar.close.as_double()
        er = self.er.value
        volume_ok = bar.volume.as_double() >= self.vol_sma.value * 0.8

        if er >= self.config.er_trend and volume_ok:
            if close > self.prev_entry_upper:
                self._enter(bar, OrderSide.BUY, "trend_long", self.config.stop_atr)
                return
            if close < self.prev_entry_lower:
                self._enter(bar, OrderSide.SELL, "trend_short", self.config.stop_atr)
                return

        if er <= self.config.er_chop:
            if close < self.keltner.lower and self.rsi.value <= self.config.rsi_buy:
                self._enter(bar, OrderSide.BUY, "fade_long", self.config.fade_stop_atr)
                return
            if close > self.keltner.upper and self.rsi.value >= self.config.rsi_sell:
                self._enter(bar, OrderSide.SELL, "fade_short", self.config.fade_stop_atr)

    def _enter(self, bar: Bar, side: OrderSide, kind: str, stop_mult: float) -> None:
        close = bar.close.as_double()
        stop_dist = stop_mult * self.atr.value
        qty = self._size(close, stop_dist)
        if qty is None:
            return

        instrument = self.cache.instrument(self.config.instrument_id)
        order = self.order_factory.market(
            self.config.instrument_id,
            side,
            qty,
        )
        self.submit_order(order)

        self.kind = kind
        self.bars_held = 0
        if side == OrderSide.BUY:
            self.stop_px = close - stop_dist
            self.extreme = bar.high.as_double()
        else:
            self.stop_px = close + stop_dist
            self.extreme = bar.low.as_double()
        self.stop_px = float(instrument.make_price(self.stop_px))

    def _should_exit(self, bar: Bar) -> bool:
        if self._stop_hit(bar):
            return True

        close = bar.close.as_double()
        kind = self.kind or ""

        if kind.startswith("trend"):
            self._ratchet_trail(bar)
            if kind == "trend_long" and close < self.prev_exit_lower:
                return True
            if kind == "trend_short" and close > self.prev_exit_upper:
                return True
            return False

        if kind == "fade_long" and close >= self.keltner.middle:
            return True
        if kind == "fade_short" and close <= self.keltner.middle:
            return True
        return self.bars_held >= self.config.fade_max_bars

    def _stop_hit(self, bar: Bar) -> bool:
        if self.stop_px is None:
            return False
        kind = self.kind or ""
        if kind.endswith("long"):
            return bar.low.as_double() <= self.stop_px
        return bar.high.as_double() >= self.stop_px

    def _ratchet_trail(self, bar: Bar) -> None:
        if self.stop_px is None or self.extreme is None:
            return
        instrument = self.cache.instrument(self.config.instrument_id)
        offset = self.config.trail_atr * self.atr.value
        if (self.kind or "").endswith("long"):
            self.extreme = max(self.extreme, bar.high.as_double())
            self.stop_px = max(self.stop_px, self.extreme - offset)
        else:
            self.extreme = min(self.extreme, bar.low.as_double())
            self.stop_px = min(self.stop_px, self.extreme + offset)
        self.stop_px = float(instrument.make_price(self.stop_px))

    def _flatten(self) -> None:
        self.close_all_positions(self.config.instrument_id)
        self.kind = None
        self.stop_px = None
        self.extreme = None
        self.bars_held = 0

    def _size(self, price: float, stop_dist: float):
        instrument = self.cache.instrument(self.config.instrument_id)
        account = self.portfolio.account(instrument.id.venue)
        if account is None or stop_dist <= 0 or price <= 0:
            return None
        balance = account.balance_total(USDT)
        if balance is None:
            return None
        equity = balance.as_double()
        risk = equity * self.config.risk_pct
        qty = min(risk / stop_dist, (equity * self.config.max_equity_frac) / price)
        if qty <= 0:
            return None
        try:
            return instrument.make_qty(qty)
        except ValueError:
            return None

    def _remember_channels(self) -> None:
        self.prev_entry_upper = self.donchian_entry.upper
        self.prev_entry_lower = self.donchian_entry.lower
        self.prev_exit_upper = self.donchian_exit.upper
        self.prev_exit_lower = self.donchian_exit.lower

    def on_stop(self) -> None:
        self.close_all_positions(self.config.instrument_id)

    def on_reset(self) -> None:
        self.kind = None
        self.stop_px = None
        self.extreme = None
        self.bars_held = 0
        self.cooldown = 0
        self.prev_entry_upper = 0.0
        self.prev_entry_lower = 0.0
        self.prev_exit_upper = 0.0
        self.prev_exit_lower = 0.0
        self.vol_sma.reset()
