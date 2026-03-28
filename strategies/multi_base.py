"""Abstract base class for multi-instrument futures strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from engine.execution import Fill, Order
from engine.multi_state import MultiInstrumentState


# Type: (symbol, order) -> order_id or None
MultiSubmitFn = Callable[[str, Order], Optional[str]]


class MultiInstrumentStrategy(ABC):
    """Base class for strategies that trade across multiple instruments.

    Lifecycle:
        on_init  -> called once with the list of instrument symbols
        on_bar   -> called at each timestamp with multi-instrument state
        on_fill  -> called when an order fills on any instrument
        on_end   -> called after the last bar

    Constraints:
        - No access to raw data, engine internals, or future data
        - All indicator computation must be incremental
        - Submit orders via submit_order(symbol, order)
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def on_init(self, symbols: list[str]) -> None:
        """Called once before the first bar with the instrument universe."""
        ...

    @abstractmethod
    def on_bar(
        self, state: MultiInstrumentState, submit_order: MultiSubmitFn
    ) -> None:
        """Called at each timestamp with all instrument data."""
        ...

    def on_fill(self, symbol: str, fill: Fill) -> None:
        """Called when an order fills. Override for fill-based logic."""
        pass

    def on_end(self) -> None:
        """Called after the last bar. Override for cleanup."""
        pass
