"""Read-only market state snapshot provided to strategies on each bar."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from engine.account import Position, Trade
from engine.data_feed import Bar
from engine.execution import Order
from engine.session_handler import SessionContext


@dataclass(frozen=True)
class MarketState:
    """Immutable snapshot of market state for a strategy.

    Strategies receive this on every bar. They cannot mutate it —
    they can only submit orders through the injected order function.
    """

    # Current bar data
    bar: Bar
    bar_index: int

    # Position
    position_direction: int  # +1 long, -1 short, 0 flat
    position_quantity: int  # Contracts (absolute)
    position_avg_entry_price: float

    # Open orders (copies, not references)
    open_orders: tuple[Order, ...]

    # Account
    equity: float
    available_margin: float
    margin_used: float
    maintenance_margin_remaining: float

    # PnL
    unrealized_pnl: float
    realized_pnl: float

    # Trade history (copies)
    trade_history: tuple[Trade, ...]

    # Session context
    session: SessionContext

    # Margin call flag
    margin_call_active: bool = False

    # Contract roll
    contract_roll_pending: bool = False

    # Notional value of current position
    notional_value: float = 0.0
