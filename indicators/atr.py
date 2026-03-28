"""Incremental (online) Average True Range."""

from __future__ import annotations

from typing import Optional


class IncrementalATR:
    """Online ATR using Wilder's smoothing, one bar at a time."""

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError("ATR period must be >= 1")
        self._period = period
        self._value: Optional[float] = None
        self._prev_close: Optional[float] = None
        self._count: int = 0
        self._sum: float = 0.0

    @property
    def value(self) -> Optional[float]:
        return self._value

    @property
    def ready(self) -> bool:
        return self._count >= self._period

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        """Feed one bar's HLC. Returns current ATR or None if warming up."""
        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )
        self._prev_close = close
        self._count += 1

        if self._value is None:
            self._sum += tr
            if self._count >= self._period:
                self._value = self._sum / self._period
            return self._value

        # Wilder's smoothing
        self._value = (self._value * (self._period - 1) + tr) / self._period
        return self._value

    def reset(self) -> None:
        self._value = None
        self._prev_close = None
        self._count = 0
        self._sum = 0.0
