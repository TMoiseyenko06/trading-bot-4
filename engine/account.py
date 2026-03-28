"""Account tracking: positions, PnL, margin, and trade records."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from engine.contract_registry import ContractSpec
from engine.execution import Fill, OrderSide

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Current futures position."""

    direction: int = 0  # +1 long, -1 short, 0 flat
    quantity: int = 0  # Absolute number of contracts
    avg_entry_price: float = 0.0

    @property
    def signed_quantity(self) -> int:
        return self.direction * self.quantity

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0


@dataclass
class Trade:
    """Completed round-trip trade record."""

    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int  # Contracts
    side: str  # "long" or "short"
    gross_pnl: float
    net_pnl: float
    commissions: float
    slippage_ticks: float
    slippage_dollars: float
    hold_duration_bars: int
    entry_bar_index: int
    exit_bar_index: int
    notional_value: float


class AccountTracker:
    """Tracks a single strategy's account: position, equity, margin, PnL."""

    def __init__(
        self,
        initial_capital: float,
        contract_spec: ContractSpec,
        max_position_size: int,
        enforce_daily_settlement: bool = True,
    ) -> None:
        self._spec = contract_spec
        self._max_position = max_position_size
        self._enforce_settlement = enforce_daily_settlement

        # Account state
        self.equity: float = initial_capital
        self.initial_capital: float = initial_capital
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.total_commissions: float = 0.0
        self.total_slippage_dollars: float = 0.0

        # Position
        self.position = Position()

        # Margin tracking
        self.margin_used: float = 0.0

        # Trade tracking
        self.trades: list[Trade] = []
        self._pending_entry: Optional[_PendingEntry] = None

        # Daily settlement
        self._last_settlement_price: Optional[float] = None
        self._daily_pnl: float = 0.0

        # Margin call state
        self.margin_call_active: bool = False
        self.margin_call_bar_index: int = -1
        self.total_margin_calls: int = 0
        self.total_forced_liquidations: int = 0

        # Equity curve
        self.equity_curve: list[tuple[datetime, float, Optional[float]]] = []

    @property
    def available_margin(self) -> float:
        return max(0.0, self.equity - self.margin_used)

    @property
    def maintenance_margin(self) -> float:
        return self.position.quantity * self._spec.maintenance_margin

    def check_margin_for_order(
        self, side: OrderSide, quantity: int, is_intraday: bool
    ) -> tuple[bool, str]:
        """Check if there is sufficient margin for a new order.

        Returns (ok, reason).
        """
        # Calculate the net position after the order
        signed_qty = quantity if side == OrderSide.BUY else -quantity
        new_signed = self.position.signed_quantity + signed_qty

        # If reducing position, always allow
        if abs(new_signed) < abs(self.position.signed_quantity):
            return True, ""

        # If increasing or flipping, check initial margin for the increase
        additional_contracts = abs(new_signed) - abs(self.position.signed_quantity)
        if additional_contracts <= 0:
            return True, ""

        margin_rate = (
            self._spec.intraday_margin if is_intraday else self._spec.initial_margin
        )
        required = additional_contracts * margin_rate

        if required > self.available_margin:
            return False, (
                f"Insufficient margin: need ${required:,.2f} for "
                f"{additional_contracts} contracts, "
                f"available ${self.available_margin:,.2f}"
            )

        # Check max position
        if abs(new_signed) > self._max_position:
            return False, (
                f"Would exceed max position size of {self._max_position} contracts"
            )

        return True, ""

    def process_fill(
        self, fill: Fill, current_bar_index: int
    ) -> list[Trade]:
        """Process a fill and update position, PnL, and margin.

        Returns any completed trades (0, 1, or 2 in case of position flip).
        """
        completed_trades: list[Trade] = []

        # Track costs
        self.total_commissions += fill.commission
        self.total_slippage_dollars += fill.slippage_dollars
        self.realized_pnl -= fill.commission
        self.equity -= fill.commission

        fill_signed = (
            fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
        )

        if self.position.is_flat:
            # Opening a new position
            self._open_position(fill, current_bar_index)
        elif (fill.side == OrderSide.BUY and self.position.direction == 1) or (
            fill.side == OrderSide.SELL and self.position.direction == -1
        ):
            # Adding to existing position
            self._add_to_position(fill, current_bar_index)
        else:
            # Reducing or flipping
            if fill.quantity <= self.position.quantity:
                # Partial or full close
                trade = self._close_position(fill, fill.quantity, current_bar_index)
                completed_trades.append(trade)
            else:
                # Position flip: close existing, then open remainder
                close_qty = self.position.quantity
                trade = self._close_position(fill, close_qty, current_bar_index)
                completed_trades.append(trade)

                # Open the remainder in the opposite direction
                remainder = fill.quantity - close_qty
                remainder_fill = Fill(
                    order_id=fill.order_id,
                    side=fill.side,
                    quantity=remainder,
                    fill_price=fill.fill_price,
                    fill_bar_index=fill.fill_bar_index,
                    fill_timestamp=fill.fill_timestamp,
                    slippage_ticks=fill.slippage_ticks,
                    slippage_dollars=0.0,  # Already accounted
                    commission=0.0,  # Already accounted
                )
                self._open_position(remainder_fill, current_bar_index)

        # Update margin
        self._update_margin(is_intraday=True)

        return completed_trades

    def _open_position(self, fill: Fill, bar_index: int) -> None:
        self.position.direction = 1 if fill.side == OrderSide.BUY else -1
        self.position.quantity = fill.quantity
        self.position.avg_entry_price = fill.fill_price
        self._pending_entry = _PendingEntry(
            time=fill.fill_timestamp,
            price=fill.fill_price,
            bar_index=bar_index,
            quantity=fill.quantity,
            side="long" if fill.side == OrderSide.BUY else "short",
            total_slippage_ticks=fill.slippage_ticks,
            total_slippage_dollars=fill.slippage_dollars,
            total_commissions=fill.commission,
        )

    def _add_to_position(self, fill: Fill, bar_index: int) -> None:
        old_qty = self.position.quantity
        new_qty = old_qty + fill.quantity
        # Weighted average entry
        self.position.avg_entry_price = (
            self.position.avg_entry_price * old_qty + fill.fill_price * fill.quantity
        ) / new_qty
        self.position.quantity = new_qty

        if self._pending_entry is not None:
            self._pending_entry.quantity = new_qty
            self._pending_entry.total_slippage_ticks += fill.slippage_ticks
            self._pending_entry.total_slippage_dollars += fill.slippage_dollars
            self._pending_entry.total_commissions += fill.commission

    def _close_position(
        self, fill: Fill, close_qty: int, bar_index: int
    ) -> Trade:
        pv = self._spec.point_value
        price_diff = fill.fill_price - self.position.avg_entry_price
        gross_pnl = price_diff * self.position.direction * close_qty * pv

        # Commissions for this close are already deducted from equity
        # Get entry-side commissions from pending entry
        entry_commissions = 0.0
        if self._pending_entry is not None:
            # Proportional share of entry commissions
            entry_commissions = (
                self._pending_entry.total_commissions
                * close_qty
                / self._pending_entry.quantity
            )

        total_commissions = entry_commissions + fill.commission
        net_pnl = gross_pnl  # commissions already deducted from equity

        # Realize PnL
        self.realized_pnl += gross_pnl
        self.equity += gross_pnl

        # Slippage tracking
        entry_slippage_ticks = 0.0
        entry_slippage_dollars = 0.0
        entry_time = fill.fill_timestamp
        entry_bar = bar_index
        side = "long"
        if self._pending_entry is not None:
            entry_slippage_ticks = (
                self._pending_entry.total_slippage_ticks
                * close_qty
                / self._pending_entry.quantity
            )
            entry_slippage_dollars = (
                self._pending_entry.total_slippage_dollars
                * close_qty
                / self._pending_entry.quantity
            )
            entry_time = self._pending_entry.time
            entry_bar = self._pending_entry.bar_index
            side = self._pending_entry.side

        notional = fill.fill_price * pv * close_qty

        trade = Trade(
            entry_time=entry_time,
            exit_time=fill.fill_timestamp,
            entry_price=self.position.avg_entry_price,
            exit_price=fill.fill_price,
            quantity=close_qty,
            side=side,
            gross_pnl=gross_pnl,
            net_pnl=gross_pnl - total_commissions,
            commissions=total_commissions,
            slippage_ticks=entry_slippage_ticks + fill.slippage_ticks,
            slippage_dollars=entry_slippage_dollars + fill.slippage_dollars,
            hold_duration_bars=bar_index - entry_bar,
            entry_bar_index=entry_bar,
            exit_bar_index=bar_index,
            notional_value=notional,
        )

        # Update position
        self.position.quantity -= close_qty
        if self.position.quantity == 0:
            self.position.direction = 0
            self.position.avg_entry_price = 0.0
            self._pending_entry = None
        elif self._pending_entry is not None:
            # Reduce pending entry proportionally
            ratio = self.position.quantity / (self.position.quantity + close_qty)
            self._pending_entry.total_commissions *= ratio
            self._pending_entry.total_slippage_ticks *= ratio
            self._pending_entry.total_slippage_dollars *= ratio
            self._pending_entry.quantity = self.position.quantity

        return trade

    def update_unrealized_pnl(self, current_price: float) -> None:
        """Mark-to-market the current position."""
        if self.position.is_flat:
            self.unrealized_pnl = 0.0
            return
        pv = self._spec.point_value
        price_diff = current_price - self.position.avg_entry_price
        self.unrealized_pnl = (
            price_diff * self.position.direction * self.position.quantity * pv
        )

    def perform_daily_settlement(
        self, settlement_price: float, timestamp: datetime
    ) -> float:
        """Mark all open positions to daily settlement price.

        Returns daily settlement PnL (realized through clearing).
        """
        if not self._enforce_settlement or self.position.is_flat:
            self._last_settlement_price = settlement_price
            return 0.0

        pv = self._spec.point_value
        ref_price = (
            self._last_settlement_price
            if self._last_settlement_price is not None
            else self.position.avg_entry_price
        )

        daily_pnl = (
            (settlement_price - ref_price)
            * self.position.direction
            * self.position.quantity
            * pv
        )

        # Daily settlement realizes PnL through clearing
        self.realized_pnl += daily_pnl
        self.equity += daily_pnl
        self._daily_pnl = daily_pnl
        self._last_settlement_price = settlement_price

        # After settlement, avg entry resets to settlement price
        self.position.avg_entry_price = settlement_price

        logger.debug(
            "Daily settlement at %.2f: PnL $%.2f", settlement_price, daily_pnl
        )
        return daily_pnl

    def check_margin_call(self, bar_index: int) -> bool:
        """Check if equity has dropped below maintenance margin.

        Returns True if a margin call is triggered.
        """
        if self.position.is_flat:
            self.margin_call_active = False
            return False

        if self.equity < self.maintenance_margin:
            if not self.margin_call_active:
                self.margin_call_active = True
                self.margin_call_bar_index = bar_index
                self.total_margin_calls += 1
                logger.warning(
                    "MARGIN CALL: equity $%.2f < maintenance $%.2f at bar %d",
                    self.equity,
                    self.maintenance_margin,
                    bar_index,
                )
            return True
        else:
            self.margin_call_active = False
            return False

    def should_force_liquidate(self, bar_index: int) -> bool:
        """Check if forced liquidation is needed (margin call not resolved)."""
        if not self.margin_call_active:
            return False
        # Force liquidate if margin call was not resolved within one bar
        if bar_index > self.margin_call_bar_index + 1:
            self.total_forced_liquidations += 1
            logger.warning(
                "FORCED LIQUIDATION at bar %d: margin call not resolved", bar_index
            )
            return True
        return False

    def record_equity_point(
        self, timestamp: datetime, settlement_mark: Optional[float] = None
    ) -> None:
        """Record a point on the equity curve."""
        self.equity_curve.append((timestamp, self.equity, settlement_mark))

    def _update_margin(self, is_intraday: bool = True) -> None:
        rate = (
            self._spec.intraday_margin if is_intraday else self._spec.initial_margin
        )
        self.margin_used = self.position.quantity * rate

    def get_notional_value(self, current_price: float) -> float:
        """Get the notional value of the current position."""
        return current_price * self._spec.point_value * self.position.quantity


@dataclass
class _PendingEntry:
    """Internal tracker for an open position's entry details."""

    time: datetime
    price: float
    bar_index: int
    quantity: int
    side: str
    total_slippage_ticks: float
    total_slippage_dollars: float
    total_commissions: float
