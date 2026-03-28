"""Order types, execution simulator, and fill logic."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from engine.config import SlippageConfig, SlippageModel
from engine.contract_registry import ContractSpec
from engine.data_feed import Bar

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class TimeInForce(Enum):
    GTC = "gtc"  # Good-til-cancelled
    DAY = "day"  # Expire at session close
    IOC = "ioc"  # Immediate-or-cancel


class OrderStatus(Enum):
    PENDING = "pending"
    TRIGGERED = "triggered"  # Stop has triggered, now acts as market/limit
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """A futures order."""

    side: OrderSide
    quantity: int  # Whole contracts only
    order_type: OrderType
    time_in_force: TimeInForce = TimeInForce.GTC
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: str = field(default_factory=lambda: uuid4().hex[:12])
    status: OrderStatus = OrderStatus.PENDING
    submitted_bar_index: int = -1
    symbol: str = ""

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("Order quantity must be a positive integer")
        if not isinstance(self.quantity, int):
            raise ValueError("Order quantity must be an integer (whole contracts)")


@dataclass(frozen=True)
class Fill:
    """Record of a filled order."""

    order_id: str
    side: OrderSide
    quantity: int
    fill_price: float
    fill_bar_index: int
    fill_timestamp: datetime
    slippage_ticks: float
    slippage_dollars: float
    commission: float


class ExecutionSimulator:
    """Simulates futures order execution with zero lookahead bias.

    Orders submitted on bar N can only fill at bar N+1 or later.
    Market orders fill at next bar's open + slippage.
    Limit/stop orders evaluated against future bar OHLC.
    """

    def __init__(
        self,
        contract_spec: ContractSpec,
        slippage_config: SlippageConfig,
        queue_position_penalty: bool = False,
    ) -> None:
        self._spec = contract_spec
        self._slippage_config = slippage_config
        self._queue_penalty = queue_position_penalty

    def validate_and_prepare_order(
        self, order: Order, bar_index: int
    ) -> Optional[str]:
        """Validate order prices against tick increments. Returns error or None."""
        order.submitted_bar_index = bar_index
        order.symbol = self._spec.symbol

        # Validate and snap prices to tick
        if order.limit_price is not None:
            if not self._spec.is_valid_tick(order.limit_price):
                snapped = self._spec.snap_price_to_tick(order.limit_price)
                logger.warning(
                    "Limit price %.6f snapped to %.6f for %s",
                    order.limit_price,
                    snapped,
                    self._spec.symbol,
                )
                order.limit_price = snapped

        if order.stop_price is not None:
            if not self._spec.is_valid_tick(order.stop_price):
                snapped = self._spec.snap_price_to_tick(order.stop_price)
                logger.warning(
                    "Stop price %.6f snapped to %.6f for %s",
                    order.stop_price,
                    snapped,
                    self._spec.symbol,
                )
                order.stop_price = snapped

        # Validate order type has required prices
        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            return "Limit order requires a limit_price"
        if order.order_type == OrderType.STOP and order.stop_price is None:
            return "Stop order requires a stop_price"
        if order.order_type == OrderType.STOP_LIMIT:
            if order.stop_price is None or order.limit_price is None:
                return "Stop-limit order requires both stop_price and limit_price"

        return None

    def process_orders(
        self, orders: list[Order], bar: Bar, bar_index: int
    ) -> list[Fill]:
        """Process pending orders against the current bar.

        This is called at the START of each bar iteration, using the new bar
        to fill orders that were submitted on previous bars.
        """
        fills: list[Fill] = []

        for order in orders:
            if order.status not in (OrderStatus.PENDING, OrderStatus.TRIGGERED):
                continue
            # Orders can only fill on bars after submission
            if bar_index <= order.submitted_bar_index:
                continue

            fill = self._try_fill(order, bar, bar_index)
            if fill is not None:
                fills.append(fill)

        return fills

    def _try_fill(
        self, order: Order, bar: Bar, bar_index: int
    ) -> Optional[Fill]:
        """Attempt to fill a single order against the given bar."""

        if order.order_type == OrderType.MARKET:
            return self._fill_market(order, bar, bar_index)

        if order.order_type == OrderType.LIMIT:
            return self._fill_limit(order, bar, bar_index)

        if order.order_type == OrderType.STOP:
            return self._fill_stop(order, bar, bar_index)

        if order.order_type == OrderType.STOP_LIMIT:
            return self._fill_stop_limit(order, bar, bar_index)

        return None

    def _fill_market(
        self, order: Order, bar: Bar, bar_index: int
    ) -> Fill:
        """Market order: fill at bar open + slippage."""
        slippage_ticks = self._compute_slippage_ticks()
        fill_price = self._apply_slippage(bar.open, order.side, slippage_ticks)
        fill_price = self._spec.snap_price_to_tick(fill_price)
        order.status = OrderStatus.FILLED

        slippage_dollars = abs(slippage_ticks) * self._spec.tick_value * order.quantity
        commission = self._spec.commission_per_side * order.quantity

        return Fill(
            order_id=order.order_id,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            fill_bar_index=bar_index,
            fill_timestamp=bar.timestamp,
            slippage_ticks=slippage_ticks,
            slippage_dollars=slippage_dollars,
            commission=commission,
        )

    def _fill_limit(
        self, order: Order, bar: Bar, bar_index: int
    ) -> Optional[Fill]:
        """Limit order: fill if price reaches limit level."""
        assert order.limit_price is not None

        if order.side == OrderSide.BUY:
            # Buy limit fills if bar low <= limit price
            if self._queue_penalty:
                triggered = bar.low < order.limit_price
            else:
                triggered = bar.low <= order.limit_price
        else:
            # Sell limit fills if bar high >= limit price
            if self._queue_penalty:
                triggered = bar.high > order.limit_price
            else:
                triggered = bar.high >= order.limit_price

        if not triggered:
            return None

        # If bar opens through the limit, fill at open (better price)
        if order.side == OrderSide.BUY:
            fill_price = min(order.limit_price, bar.open)
        else:
            fill_price = max(order.limit_price, bar.open)

        fill_price = self._spec.snap_price_to_tick(fill_price)
        order.status = OrderStatus.FILLED
        commission = self._spec.commission_per_side * order.quantity

        return Fill(
            order_id=order.order_id,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            fill_bar_index=bar_index,
            fill_timestamp=bar.timestamp,
            slippage_ticks=0.0,
            slippage_dollars=0.0,
            commission=commission,
        )

    def _fill_stop(
        self, order: Order, bar: Bar, bar_index: int
    ) -> Optional[Fill]:
        """Stop order: triggers to market when stop is breached."""
        assert order.stop_price is not None

        if order.side == OrderSide.BUY:
            triggered = bar.high >= order.stop_price
        else:
            triggered = bar.low <= order.stop_price

        if not triggered:
            return None

        # Fill at worse of stop price or bar open, plus slippage
        # This models gap-through scenarios
        slippage_ticks = self._compute_slippage_ticks()

        if order.side == OrderSide.BUY:
            base_price = max(order.stop_price, bar.open)
        else:
            base_price = min(order.stop_price, bar.open)

        fill_price = self._apply_slippage(base_price, order.side, slippage_ticks)
        fill_price = self._spec.snap_price_to_tick(fill_price)
        order.status = OrderStatus.FILLED

        slippage_dollars = abs(slippage_ticks) * self._spec.tick_value * order.quantity
        commission = self._spec.commission_per_side * order.quantity

        return Fill(
            order_id=order.order_id,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            fill_bar_index=bar_index,
            fill_timestamp=bar.timestamp,
            slippage_ticks=slippage_ticks,
            slippage_dollars=slippage_dollars,
            commission=commission,
        )

    def _fill_stop_limit(
        self, order: Order, bar: Bar, bar_index: int
    ) -> Optional[Fill]:
        """Stop-limit: triggers to limit when stop breached. May not fill."""
        assert order.stop_price is not None
        assert order.limit_price is not None

        if order.status == OrderStatus.PENDING:
            # Check if stop is triggered
            if order.side == OrderSide.BUY:
                triggered = bar.high >= order.stop_price
            else:
                triggered = bar.low <= order.stop_price

            if not triggered:
                return None

            # Stop triggered — convert to limit order
            order.status = OrderStatus.TRIGGERED
            logger.debug(
                "Stop-limit %s triggered at bar %d", order.order_id, bar_index
            )

        # Now behave as a limit order
        if order.side == OrderSide.BUY:
            if self._queue_penalty:
                can_fill = bar.low < order.limit_price
            else:
                can_fill = bar.low <= order.limit_price
            # But also check: if price gapped through limit, no fill
            if bar.open > order.limit_price:
                can_fill = False
        else:
            if self._queue_penalty:
                can_fill = bar.high > order.limit_price
            else:
                can_fill = bar.high >= order.limit_price
            if bar.open < order.limit_price:
                can_fill = False

        if not can_fill:
            return None

        if order.side == OrderSide.BUY:
            fill_price = min(order.limit_price, bar.open)
        else:
            fill_price = max(order.limit_price, bar.open)

        fill_price = self._spec.snap_price_to_tick(fill_price)
        order.status = OrderStatus.FILLED
        commission = self._spec.commission_per_side * order.quantity

        return Fill(
            order_id=order.order_id,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            fill_bar_index=bar_index,
            fill_timestamp=bar.timestamp,
            slippage_ticks=0.0,
            slippage_dollars=0.0,
            commission=commission,
        )

    def _compute_slippage_ticks(self) -> float:
        """Compute slippage in ticks based on the configured model."""
        if self._slippage_config.model == SlippageModel.FIXED:
            return self._slippage_config.fixed_ticks
        # Normal distribution
        return max(
            0.0,
            random.gauss(
                self._slippage_config.normal_mean_ticks,
                self._slippage_config.normal_std_ticks,
            ),
        )

    def _apply_slippage(
        self, price: float, side: OrderSide, slippage_ticks: float
    ) -> float:
        """Apply slippage to a price. Buys slip up, sells slip down."""
        tick_size = self._spec.tick_size
        if side == OrderSide.BUY:
            return price + slippage_ticks * tick_size
        return price - slippage_ticks * tick_size

    def expire_day_orders(
        self, orders: list[Order], is_session_close: bool
    ) -> list[Order]:
        """Cancel DAY orders at session close."""
        cancelled = []
        if not is_session_close:
            return cancelled
        for order in orders:
            if (
                order.time_in_force == TimeInForce.DAY
                and order.status in (OrderStatus.PENDING, OrderStatus.TRIGGERED)
            ):
                order.status = OrderStatus.CANCELLED
                cancelled.append(order)
                logger.info("DAY order %s expired at session close", order.order_id)
        return cancelled
