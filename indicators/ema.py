"""Incremental (online) Exponential Moving Average."""

from __future__ import annotations

from typing import Optional


class IncrementalEMA:
    """Online EMA that updates one value at a time.

    No pandas. No full-array computation. Purely causal.
    """

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("EMA period must be >= 1")
        self._period = period
        self._multiplier = 2.0 / (period + 1)
        self._value: Optional[float] = None
        self._count: int = 0
        self._sum: float = 0.0

    @property
    def value(self) -> Optional[float]:
        return self._value

    @property
    def ready(self) -> bool:
        return self._count >= self._period

    def update(self, price: float) -> Optional[float]:
        """Feed one new price. Returns current EMA value or None if warming up."""
        self._count += 1

        if self._value is None:
            # Seed phase: use SMA for the first `period` values
            self._sum += price
            if self._count >= self._period:
                self._value = self._sum / self._period
            return self._value

        self._value = (price - self._value) * self._multiplier + self._value
        return self._value

    def reset(self) -> None:
        self._value = None
        self._count = 0
        self._sum = 0.0
