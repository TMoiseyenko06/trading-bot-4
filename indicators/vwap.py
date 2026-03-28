"""Incremental (online) VWAP with session reset support."""

from __future__ import annotations

from typing import Optional


class IncrementalVWAP:
    """Online VWAP that accumulates price*volume and volume.

    Automatically resets on session open for session-anchored VWAP.
    """

    def __init__(self) -> None:
        self._cum_pv: float = 0.0
        self._cum_volume: int = 0
        self._value: Optional[float] = None
        # Standard deviation bands
        self._cum_pv2: float = 0.0  # sum of price^2 * volume

    @property
    def value(self) -> Optional[float]:
        return self._value

    @property
    def ready(self) -> bool:
        return self._cum_volume > 0

    @property
    def std_dev(self) -> float:
        """Current VWAP standard deviation."""
        if self._cum_volume == 0 or self._value is None:
            return 0.0
        mean_p2 = self._cum_pv2 / self._cum_volume
        variance = max(0.0, mean_p2 - self._value ** 2)
        return variance ** 0.5

    def update(
        self, typical_price: float, volume: int, session_open: bool = False
    ) -> Optional[float]:
        """Feed one bar. If session_open, reset accumulation.

        typical_price is usually (high + low + close) / 3.
        """
        if session_open:
            self.reset()

        if volume <= 0:
            return self._value

        self._cum_pv += typical_price * volume
        self._cum_pv2 += (typical_price ** 2) * volume
        self._cum_volume += volume
        self._value = self._cum_pv / self._cum_volume
        return self._value

    def reset(self) -> None:
        self._cum_pv = 0.0
        self._cum_pv2 = 0.0
        self._cum_volume = 0
        self._value = None
