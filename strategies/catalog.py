"""All Signal classes by key."""

from __future__ import annotations

from strategies.breakout import Breakout
from strategies.ensemble import Ensemble
from strategies.mean_reversion import MeanReversion
from strategies.signal_base import Signal
from strategies.supertrend import Supertrend
from strategies.trend_ensemble import TrendEnsemble

SIGNALS: dict[str, type[Signal]] = {
    cls.key: cls
    for cls in (TrendEnsemble, Breakout, Supertrend, MeanReversion, Ensemble)
}
