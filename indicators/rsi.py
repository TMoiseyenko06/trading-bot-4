"""Incremental (online) Relative Strength Index."""

from __future__ import annotations

from typing import Optional


class IncrementalRSI:
    """Online RSI using Wilder's smoothing, one value at a time."""

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError("RSI period must be >= 1")
        self._period = period
        self._prev_price: Optional[float] = None
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        self._value: Optional[float] = None
        self._count: int = 0
        self._gains: list[float] = []
        self._losses: list[float] = []

    @property
    def value(self) -> Optional[float]:
        return self._value

    @property
    def ready(self) -> bool:
        return self._count > self._period

    def update(self, price: float) -> Optional[float]:
        """Feed one new price. Returns RSI or None if warming up."""
        if self._prev_price is None:
            self._prev_price = price
            self._count += 1
            return None

        change = price - self._prev_price
        self._prev_price = price
        gain = max(0.0, change)
        loss = max(0.0, -change)
        self._count += 1

        if self._avg_gain is None:
            # Seed phase
            self._gains.append(gain)
            self._losses.append(loss)
            if len(self._gains) >= self._period:
                self._avg_gain = sum(self._gains) / self._period
                self._avg_loss = sum(self._losses) / self._period
                self._gains = []
                self._losses = []
                if self._avg_loss == 0:
                    self._value = 100.0
                else:
                    rs = self._avg_gain / self._avg_loss
                    self._value = 100.0 - (100.0 / (1.0 + rs))
            return self._value

        # Wilder's smoothing
        self._avg_gain = (self._avg_gain * (self._period - 1) + gain) / self._period
        self._avg_loss = (self._avg_loss * (self._period - 1) + loss) / self._period

        if self._avg_loss == 0:
            self._value = 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            self._value = 100.0 - (100.0 / (1.0 + rs))
        return self._value

    def reset(self) -> None:
        self._prev_price = None
        self._avg_gain = None
        self._avg_loss = None
        self._value = None
        self._count = 0
        self._gains = []
        self._losses = []
