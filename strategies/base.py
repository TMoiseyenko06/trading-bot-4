"""Abstract base class for futures trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from engine.execution import Fill, Order
from engine.market_state import MarketState


# Type alias for the order submission function injected into strategies
SubmitOrderFn = Callable[[Order], Optional[str]]


class Strategy(ABC):
    """Abstract base class for futures trading strategies.

    Lifecycle:
        on_init  -> called once before the first bar
        on_bar   -> called for every bar with market state + order function
        on_fill  -> called when an order fills
        on_end   -> called after the last bar

    Constraints:
        - No access to the data source, engine internals, or other strategies
        - All indicator computation must be incremental (online)
        - Cannot mutate MarketState; can only submit orders via submit_order
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def on_init(self) -> None:
        """Called once before the first bar. Set up indicators and state."""
        ...

    @abstractmethod
    def on_bar(self, state: MarketState, submit_order: SubmitOrderFn) -> None:
        """Called on every bar. Evaluate signals and submit orders."""
        ...

    def on_fill(self, fill: Fill) -> None:
        """Called when an order fills. Override for fill-based logic."""
        pass

    def on_end(self) -> None:
        """Called after the last bar. Override for cleanup."""
        pass
