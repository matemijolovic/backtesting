"""Every runnable strategy, with its parameters and how to build it.

Three kinds:

- ``signal``: new-style strategies (``strategies/*``). Lookbacks are in
  days/hours, so they mean the same on any timeframe.
- ``legacy``: the original bar-count strategies. Their lookbacks are
  calibrated on ``ref_interval`` and rescaled to the run's timeframe
  when ``scale_lookbacks`` is on, e.g. SMA 20 on 1d becomes SMA 480 on 1h.
- ``carry``: delta-neutral spot/perp funding carry (needs perp + funding data).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from decimal import Decimal
from typing import Any
from typing import Callable

from strategies.catalog import SIGNALS
from strategies.signal_base import Param

from backtester.data import interval_minutes


@dataclass(frozen=True)
class StrategySpec:
    key: str
    title: str
    summary: str
    kind: str  # "signal" | "legacy" | "carry"
    params: tuple[Param, ...]
    ref_interval: str = "1h"
    build: Callable[..., Any] | None = field(default=None, repr=False)

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def resolve(self, params: dict[str, Any] | None, interval: str, scale_lookbacks: bool) -> dict[str, Any]:
        """Defaults + overrides, with bar lookbacks rescaled to ``interval``."""
        out = self.defaults()
        known = {p.name for p in self.params}
        for name, value in (params or {}).items():
            if name not in known:
                raise ValueError(f"{self.key}: unknown param {name!r}; have {sorted(known)}")
            out[name] = value
        if scale_lookbacks:
            ratio = interval_minutes(self.ref_interval) / interval_minutes(interval)
            for p in self.params:
                if p.lookback:
                    out[p.name] = max(2, int(round(float(out[p.name]) * ratio)))
        for p in self.params:
            if p.kind == "int":
                out[p.name] = int(round(float(out[p.name])))
            elif p.kind == "float":
                out[p.name] = float(out[p.name])
        return out

    def warmup_days(self, params: dict[str, Any], interval: str) -> float:
        """How much history to load before the measured window starts."""
        if self.kind == "carry":
            return 1.0
        if self.kind == "legacy":
            bars = max((float(params[p.name]) for p in self.params if p.lookback), default=50.0)
            return bars * interval_minutes(interval) / 1440.0 + 1.0
        days = [float(v) for k, v in params.items() if k.endswith("_days") and isinstance(v, (int, float))]
        hours = [
            float(v) / 24.0
            for k, v in params.items()
            if k.endswith("_hours") and k not in ("max_hold_hours", "rebalance_hours")
        ]
        if self.key == "ensemble":
            days.append(95.0)  # slowest child (trend_ensemble slow_days + slack)
        return max(days + hours + [1.0]) + 2.0


# --------------------------------------------------------------------------- #
# Builders: (spec, instrument, bar_type, params, run) -> Strategy
# --------------------------------------------------------------------------- #


def _build_signal(spec, instrument, bar_type, params, run):
    from strategies.signal_strategy import SignalStrategy
    from strategies.signal_strategy import SignalStrategyConfig

    return SignalStrategy(
        SignalStrategyConfig(
            instrument_id=instrument.id,
            bar_type=bar_type,
            signal=spec.key,
            params=params,
            leverage=run.leverage,
            rebalance_threshold=run.rebalance_threshold,
            use_stops=run.use_stops,
        )
    )


def _build_dual_ma(spec, instrument, bar_type, params, run):
    from strategies.dual_ma import DualMAConfig
    from strategies.dual_ma import DualMATrend

    return DualMATrend(
        DualMAConfig(instrument_id=instrument.id, bar_type=bar_type, leverage=run.leverage, **params)
    )


def _build_donchian(spec, instrument, bar_type, params, run):
    from strategies.donchian import DonchianConfig
    from strategies.donchian import DonchianTrend

    return DonchianTrend(
        DonchianConfig(instrument_id=instrument.id, bar_type=bar_type, leverage=run.leverage, **params)
    )


def _build_regime(spec, instrument, bar_type, params, run):
    from strategies.regime import RegimeConfig
    from strategies.regime import RegimeStrategy

    return RegimeStrategy(RegimeConfig(instrument_id=instrument.id, bar_type=bar_type, **params))


def _build_ema_cross(spec, instrument, bar_type, params, run):
    from strategies.ema_cross import EMACross
    from strategies.ema_cross import EMACrossConfig

    # Fixed size: whatever the starting capital buys at the first price.
    size = run.capital * run.leverage * 0.98 / max(run.first_price, 1e-12)
    return EMACross(
        EMACrossConfig(
            instrument_id=instrument.id,
            bar_type=bar_type,
            trade_size=Decimal(str(instrument.make_qty(size))),
            **params,
        )
    )


def _signal_spec(cls) -> StrategySpec:
    return StrategySpec(
        key=cls.key,
        title=cls.title,
        summary=cls.summary,
        kind="signal",
        params=cls.PARAMS,
        build=_build_signal,
    )


REGISTRY: dict[str, StrategySpec] = {key: _signal_spec(cls) for key, cls in SIGNALS.items()}

REGISTRY.update(
    {
        "dual_ma": StrategySpec(
            key="dual_ma",
            title="Dual SMA (legacy)",
            summary="Always-in long/short on fast vs slow SMA. Original repo strategy.",
            kind="legacy",
            ref_interval="1d",
            params=(
                Param("fast", 20, 2, 400, 1, "Fast SMA (bars on ref interval)", lookback=True),
                Param("slow", 50, 5, 1000, 1, "Slow SMA (bars on ref interval)", lookback=True),
            ),
            build=_build_dual_ma,
        ),
        "donchian": StrategySpec(
            key="donchian",
            title="Donchian always-in (legacy)",
            summary="Reverse on N-bar channel breaks with a wide ATR trail. Original repo strategy.",
            kind="legacy",
            ref_interval="1d",
            params=(
                Param("channel", 20, 2, 400, 1, "Channel (bars on ref interval)", lookback=True),
                Param("atr_period", 20, 2, 400, 1, "ATR period (bars on ref interval)", lookback=True),
                Param("stop_atr", 5.0, 0.5, 15.0, 0.25, "ATR multiple for the trail"),
            ),
            build=_build_donchian,
        ),
        "regime": StrategySpec(
            key="regime",
            title="Regime switch (legacy)",
            summary="Efficiency-ratio switch between breakout and Keltner fade. Original repo strategy.",
            kind="legacy",
            ref_interval="1h",
            params=(
                Param("donchian_entry", 72, 5, 1000, 1, "Entry channel (bars)", lookback=True),
                Param("donchian_exit", 24, 2, 500, 1, "Exit channel (bars)", lookback=True),
                Param("atr_period", 24, 2, 500, 1, "ATR period (bars)", lookback=True),
                Param("er_period", 24, 2, 500, 1, "Efficiency ratio period (bars)", lookback=True),
                Param("keltner_period", 24, 2, 500, 1, "Keltner period (bars)", lookback=True),
                Param("rsi_period", 14, 2, 300, 1, "RSI period (bars)", lookback=True),
                Param("vol_period", 24, 2, 500, 1, "Volume SMA period (bars)", lookback=True),
                Param("fade_max_bars", 12, 1, 300, 1, "Fade time stop (bars)", lookback=True),
                Param("cooldown_bars", 6, 0, 300, 1, "Pause after a stop-out (bars)", lookback=True),
                Param("er_trend", 0.35, 0.05, 0.9, 0.05, "ER above which = trend"),
                Param("er_chop", 0.25, 0.05, 0.9, 0.05, "ER below which = chop"),
                Param("risk_pct", 0.0075, 0.001, 0.05, 0.0005, "Equity risked per trade"),
                Param("max_equity_frac", 0.25, 0.05, 2.0, 0.05, "Max position as fraction of equity"),
            ),
            build=_build_regime,
        ),
        "ema_cross": StrategySpec(
            key="ema_cross",
            title="EMA cross long-only (legacy)",
            summary="Long while fast EMA > slow EMA, fixed size. The repo's first baseline.",
            kind="legacy",
            ref_interval="1d",
            params=(
                Param("fast_ema_period", 20, 2, 400, 1, "Fast EMA (bars on ref interval)", lookback=True),
                Param("slow_ema_period", 50, 5, 1000, 1, "Slow EMA (bars on ref interval)", lookback=True),
            ),
            build=_build_ema_cross,
        ),
        "funding_carry": StrategySpec(
            key="funding_carry",
            title="Funding carry (spot long / perp short)",
            summary="Delta-neutral: collect positive perp funding, hedge price with spot.",
            kind="carry",
            params=(
                Param("enter_rate", 0.00002, 0.0, 0.001, 0.00001, "Enter when last funding >= this"),
                Param("exit_rate", 0.0, -0.001, 0.001, 0.00001, "Exit side threshold"),
                Param("enter_confirm", 1, 1, 10, 1, "Prints above enter_rate needed"),
                Param("exit_confirm", 9, 1, 30, 1, "Prints below exit_rate needed"),
                Param("notional_frac", 0.90, 0.1, 1.0, 0.05, "Hedge notional / equity"),
                Param("min_basis", -0.0005, -0.01, 0.01, 0.0001, "Min perp premium to enter"),
            ),
            build=None,
        ),
    }
)


def get(key: str) -> StrategySpec:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"Unknown strategy {key!r}. Choose from: {', '.join(REGISTRY)}") from None
