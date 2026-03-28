"""EMA Crossover strategy — all indicators computed incrementally."""

from __future__ import annotations

import logging

from engine.execution import Fill, Order, OrderSide, OrderType, TimeInForce
from engine.market_state import MarketState
from indicators.ema import IncrementalEMA
from strategies.base import Strategy, SubmitOrderFn

logger = logging.getLogger(__name__)


class EMACrossoverStrategy(Strategy):
    """Long-only EMA crossover for futures.

    Enters long when fast EMA crosses above slow EMA.
    Exits when fast EMA crosses below slow EMA.
    All EMA values computed incrementally, one bar at a time.
    """

    def __init__(
        self,
        name: str = "EMA_Crossover",
        fast_period: int = 9,
        slow_period: int = 21,
        contracts: int = 1,
        rth_only: bool = True,
    ) -> None:
        super().__init__(name)
        self._fast_period = fast_period
        self._slow_period = slow_period
        self._contracts = contracts
        self._rth_only = rth_only
        self._ema_fast: IncrementalEMA = IncrementalEMA(1)  # placeholder
        self._ema_slow: IncrementalEMA = IncrementalEMA(1)
        self._prev_fast: float | None = None
        self._prev_slow: float | None = None

    def on_init(self) -> None:
        self._ema_fast = IncrementalEMA(self._fast_period)
        self._ema_slow = IncrementalEMA(self._slow_period)
        self._prev_fast = None
        self._prev_slow = None

    def on_bar(self, state: MarketState, submit_order: SubmitOrderFn) -> None:
        bar = state.bar

        # Update EMAs incrementally
        fast_val = self._ema_fast.update(bar.close)
        slow_val = self._ema_slow.update(bar.close)

        # Need both EMAs ready
        if fast_val is None or slow_val is None:
            self._prev_fast = fast_val
            self._prev_slow = slow_val
            return

        # Optionally only trade during RTH
        if self._rth_only and not state.session.is_rth:
            self._prev_fast = fast_val
            self._prev_slow = slow_val
            return

        # Skip if we don't have previous values for crossover detection
        if self._prev_fast is None or self._prev_slow is None:
            self._prev_fast = fast_val
            self._prev_slow = slow_val
            return

        # Detect crossover
        prev_above = self._prev_fast > self._prev_slow
        curr_above = fast_val > slow_val

        if not prev_above and curr_above:
            # Bullish crossover: go long if flat
            if state.position_quantity == 0:
                order = Order(
                    side=OrderSide.BUY,
                    quantity=self._contracts,
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.GTC,
                )
                result = submit_order(order)
                if result:
                    logger.debug(
                        "EMA crossover BUY signal at bar %d, close=%.2f",
                        state.bar_index,
                        bar.close,
                    )

        elif prev_above and not curr_above:
            # Bearish crossover: exit long
            if state.position_direction == 1:
                order = Order(
                    side=OrderSide.SELL,
                    quantity=state.position_quantity,
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.GTC,
                )
                result = submit_order(order)
                if result:
                    logger.debug(
                        "EMA crossover SELL signal at bar %d, close=%.2f",
                        state.bar_index,
                        bar.close,
                    )

        # Handle margin calls
        if state.margin_call_active and state.position_quantity > 0:
            order = Order(
                side=OrderSide.SELL if state.position_direction == 1 else OrderSide.BUY,
                quantity=state.position_quantity,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.IOC,
            )
            submit_order(order)
            logger.warning("EMA strategy responding to margin call — flattening")

        self._prev_fast = fast_val
        self._prev_slow = slow_val

    def on_fill(self, fill: Fill) -> None:
        logger.debug(
            "EMA fill: %s %d @ %.2f (slippage: %.1f ticks)",
            fill.side.value,
            fill.quantity,
            fill.fill_price,
            fill.slippage_ticks,
        )
