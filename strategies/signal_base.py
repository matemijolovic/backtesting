"""Signal interface: bars in, target exposure out.

A Signal owns the *decision* (where to be), not execution. Lookbacks are
expressed in hours or days and converted to bars for the timeframe being
run, so the same parameter set means the same thing on 15m and 4h data.
Volatility is measured per bar and rescaled to a daily figure for the
same reason.

``on_bar`` returns a target in [-1, 1] (fraction of the strategy's
maximum exposure), or ``None`` to keep the current position. A signal
may also set ``stop_px``; the execution layer turns it into a real stop
order at the venue and calls ``on_stopped`` if it fills.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import ClassVar


@dataclass(frozen=True)
class Param:
    name: str
    default: Any
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    help: str = ""
    choices: tuple | None = None
    # Bar-count lookback calibrated on the strategy's reference interval.
    # Scaled to the run's timeframe when "scale lookbacks" is on.
    lookback: bool = False
    # Perturb this parameter in robustness tests.
    jitter: bool = True

    @property
    def kind(self) -> str:
        if self.choices is not None:
            return "choice"
        if isinstance(self.default, bool):
            return "bool"
        if isinstance(self.default, int):
            return "int"
        if isinstance(self.default, float):
            return "float"
        return "str"


@dataclass(slots=True)
class BarData:
    ts: int  # close time, ns
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    """Base class. Subclasses set ``key``/``title``/``PARAMS`` and implement ``step``."""

    key: ClassVar[str] = ""
    title: ClassVar[str] = ""
    summary: ClassVar[str] = ""
    PARAMS: ClassVar[tuple[Param, ...]] = ()

    bar_minutes: int
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        known = {p.name: p for p in self.PARAMS}
        unknown = set(self.params) - set(known)
        if unknown:
            raise ValueError(f"{self.key}: unknown params {sorted(unknown)}")
        resolved = {p.name: p.default for p in self.PARAMS}
        for name, value in self.params.items():
            resolved[name] = _coerce(known[name], value)
        self.p = resolved
        self.stop_px: float | None = None
        self.position = 0.0  # last target handed to execution
        self.n_bars = 0
        # True when nothing executes stop_px at a venue (e.g. inside an
        # ensemble): the signal then checks it against bar highs/lows.
        self.self_stops = False
        self.setup()

    # -- hooks ----------------------------------------------------------- #

    def setup(self) -> None:
        """Create indicators. ``self.p`` is available."""

    def step(self, bar: BarData) -> float | None:
        raise NotImplementedError

    def on_stopped(self) -> None:
        """A protective stop filled; the position is now flat."""
        self.position = 0.0
        self.stop_px = None

    # -- helpers --------------------------------------------------------- #

    def on_bar(self, bar: BarData) -> float | None:
        self.n_bars += 1
        if self.self_stops and self.stop_px is not None and self.position != 0.0:
            hit = bar.low <= self.stop_px if self.position > 0 else bar.high >= self.stop_px
            if hit:
                self.on_stopped()
        target = self.step(bar)
        if target is not None:
            target = max(-1.0, min(1.0, float(target)))
            if math.isnan(target):
                return None
            self.position = target
            if target == 0.0:
                self.stop_px = None
        return target

    def bars(self, hours: float) -> int:
        """Hours -> bar count on this timeframe (at least 2)."""
        return max(2, int(round(hours * 60.0 / self.bar_minutes)))

    def days(self, days: float) -> int:
        return self.bars(days * 24.0)

    @property
    def bars_per_day(self) -> float:
        return 1440.0 / self.bar_minutes

    def boundary(self, ts_ns: int, hours: float) -> int:
        """Index of the ``hours``-long UTC slot this bar closes in."""
        slot = int(hours * 3600 * 1_000_000_000)
        return (ts_ns - 1) // slot if slot > 0 else ts_ns


def _coerce(param: Param, value: Any) -> Any:
    kind = param.kind
    if kind == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if kind == "int":
        return int(round(float(value)))
    if kind == "float":
        return float(value)
    return value
