"""Small streaming indicators for signals. Pure Python, O(1) per update.

Kept independent of Nautilus so signals can be unit-tested on plain
floats and run inside any loop.
"""

from __future__ import annotations

import math
from collections import deque


class SMA:
    def __init__(self, period: int) -> None:
        self.period = max(int(period), 1)
        self.window: deque[float] = deque()
        self.total = 0.0
        self.value = math.nan

    @property
    def ready(self) -> bool:
        return len(self.window) >= self.period

    def update(self, x: float) -> float:
        self.window.append(x)
        self.total += x
        if len(self.window) > self.period:
            self.total -= self.window.popleft()
        self.value = self.total / len(self.window)
        return self.value


class EMA:
    def __init__(self, period: int) -> None:
        self.period = max(int(period), 1)
        self.alpha = 2.0 / (self.period + 1.0)
        self.count = 0
        self.value = math.nan

    @property
    def ready(self) -> bool:
        return self.count >= self.period

    def update(self, x: float) -> float:
        self.count += 1
        self.value = x if self.count == 1 else self.value + self.alpha * (x - self.value)
        return self.value


class RollingStd:
    """Sample std over a fixed window (sum / sum-of-squares)."""

    def __init__(self, period: int) -> None:
        self.period = max(int(period), 2)
        self.window: deque[float] = deque()
        self.s1 = 0.0
        self.s2 = 0.0
        self.value = math.nan

    @property
    def ready(self) -> bool:
        return len(self.window) >= self.period

    def update(self, x: float) -> float:
        self.window.append(x)
        self.s1 += x
        self.s2 += x * x
        if len(self.window) > self.period:
            old = self.window.popleft()
            self.s1 -= old
            self.s2 -= old * old
        n = len(self.window)
        if n >= 2:
            var = (self.s2 - self.s1 * self.s1 / n) / (n - 1)
            self.value = math.sqrt(max(var, 0.0))
        return self.value


class RollingExtreme:
    """Rolling max (or min) via a monotonic deque."""

    def __init__(self, period: int, mode: str = "max") -> None:
        self.period = max(int(period), 1)
        self.sign = 1.0 if mode == "max" else -1.0
        self.q: deque[tuple[int, float]] = deque()
        self.i = 0
        self.value = math.nan

    @property
    def ready(self) -> bool:
        return self.i >= self.period

    def update(self, x: float) -> float:
        key = self.sign * x
        while self.q and self.sign * self.q[-1][1] <= key:
            self.q.pop()
        self.q.append((self.i, x))
        self.i += 1
        while self.q[0][0] <= self.i - 1 - self.period:
            self.q.popleft()
        self.value = self.q[0][1]
        return self.value


class ATR:
    """Wilder average true range."""

    def __init__(self, period: int) -> None:
        self.period = max(int(period), 1)
        self.prev_close = math.nan
        self.count = 0
        self.value = math.nan

    @property
    def ready(self) -> bool:
        return self.count >= self.period

    def update(self, high: float, low: float, close: float) -> float:
        if math.isnan(self.prev_close):
            tr = high - low
        else:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        self.prev_close = close
        self.count += 1
        if self.count == 1:
            self.value = tr
        else:
            n = min(self.count, self.period)
            self.value += (tr - self.value) / n
        return self.value


class RSI:
    """Wilder RSI in [0, 100]."""

    def __init__(self, period: int) -> None:
        self.period = max(int(period), 1)
        self.prev = math.nan
        self.gain = 0.0
        self.loss = 0.0
        self.count = 0
        self.value = 50.0

    @property
    def ready(self) -> bool:
        return self.count > self.period

    def update(self, x: float) -> float:
        if math.isnan(self.prev):
            self.prev = x
            self.count = 1
            return self.value
        change = x - self.prev
        self.prev = x
        self.count += 1
        n = min(self.count - 1, self.period)
        self.gain += (max(change, 0.0) - self.gain) / n
        self.loss += (max(-change, 0.0) - self.loss) / n
        if self.loss == 0.0:
            self.value = 100.0 if self.gain > 0 else 50.0
        else:
            self.value = 100.0 - 100.0 / (1.0 + self.gain / self.loss)
        return self.value


class KAMA:
    """Kaufman adaptive moving average + its efficiency ratio."""

    def __init__(self, period: int, fast: int = 2, slow: int = 30) -> None:
        self.period = max(int(period), 1)
        self.fast_sc = 2.0 / (fast + 1.0)
        self.slow_sc = 2.0 / (slow + 1.0)
        self.prices: deque[float] = deque(maxlen=self.period + 1)
        self.noise = 0.0
        self.value = math.nan
        self.er = 0.0

    @property
    def ready(self) -> bool:
        return len(self.prices) > self.period

    def update(self, x: float) -> float:
        if self.prices:
            self.noise += abs(x - self.prices[-1])
        if len(self.prices) == self.prices.maxlen:
            dropped = self.prices[0]
            self.noise -= abs(self.prices[1] - dropped)
        self.prices.append(x)
        if math.isnan(self.value):
            self.value = x
            return self.value
        signal = abs(x - self.prices[0])
        self.er = signal / self.noise if self.noise > 1e-12 else 0.0
        sc = (self.er * (self.fast_sc - self.slow_sc) + self.slow_sc) ** 2
        self.value += sc * (x - self.value)
        return self.value


class LogReturnVol:
    """Annualised volatility of bar log returns over a rolling window."""

    def __init__(self, period: int, bar_minutes: int) -> None:
        self.std = RollingStd(period)
        self.prev = math.nan
        self.scale = math.sqrt(365.0 * 1440.0 / bar_minutes)
        self.value = math.nan

    @property
    def ready(self) -> bool:
        return self.std.ready

    def update(self, close: float) -> float:
        if not math.isnan(self.prev) and self.prev > 0 and close > 0:
            self.std.update(math.log(close / self.prev))
            self.value = self.std.value * self.scale
        self.prev = close
        return self.value
