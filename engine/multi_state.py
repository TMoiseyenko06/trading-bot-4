"""Read-only multi-instrument market state snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from engine.account import Position, Trade
from engine.data_feed import Bar
from engine.execution import Order
from engine.session_handler import SessionContext


@dataclass(frozen=True)
class InstrumentState:
    """Per-instrument snapshot within a multi-instrument state."""

    symbol: str
    bar: Bar
    position_direction: int
    position_quantity: int
    position_avg_entry_price: float
    open_orders: tuple[Order, ...]
    unrealized_pnl: float
    realized_pnl: float
    trade_history: tuple[Trade, ...]
    session: SessionContext
    margin_used: float
    notional_value: float


@dataclass(frozen=True)
class MultiInstrumentState:
    """Immutable multi-instrument market state snapshot.

    Provides both per-instrument data and aggregate account data.
    Strategies cannot mutate this — they submit orders via the
    injected order function.
    """

    # Timestamp of this bar group
    timestamp: object  # datetime
    bar_index: int

    # Per-instrument states (symbol -> InstrumentState)
    instruments: dict[str, InstrumentState]

    # Which instruments have bars at this timestamp
    active_symbols: tuple[str, ...]

    # Aggregate account
    equity: float
    available_margin: float
    total_margin_used: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_trade_count: int

    # Session (from any instrument — all CME equity share the same session)
    session: SessionContext

    # Flags
    margin_call_active: bool = False

    def has_all_instruments(self, symbols: list[str]) -> bool:
        """Check if all requested instruments have bars at this timestamp."""
        return all(s in self.instruments for s in symbols)
