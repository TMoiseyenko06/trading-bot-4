"""VWAP Mean Reversion strategy — all indicators computed incrementally."""

from __future__ import annotations

import logging

from engine.execution import Fill, Order, OrderSide, OrderType, TimeInForce
from engine.market_state import MarketState
from indicators.ema import IncrementalEMA
from indicators.vwap import IncrementalVWAP
from strategies.base import Strategy, SubmitOrderFn

logger = logging.getLogger(__name__)


class VWAPMeanReversionStrategy(Strategy):
    """Mean reversion around session VWAP for futures.

    Fades moves away from VWAP by a configurable number of standard
    deviations. Exits when price reverts to VWAP. VWAP resets at
    session open automatically.

    All indicators computed incrementally.
    """

    def __init__(
        self,
        name: str = "VWAP_Reversion",
        entry_std_devs: float = 2.0,
        exit_std_devs: float = 0.5,
        contracts: int = 1,
        max_trades_per_session: int = 3,
        trend_filter_period: int = 50,
    ) -> None:
        super().__init__(name)
        self._entry_std = entry_std_devs
        self._exit_std = exit_std_devs
        self._contracts = contracts
        self._max_trades = max_trades_per_session
        self._trend_period = trend_filter_period

        self._vwap: IncrementalVWAP = IncrementalVWAP()
        self._trend_ema: IncrementalEMA = IncrementalEMA(1)
        self._session_trades: int = 0

    def on_init(self) -> None:
        self._vwap = IncrementalVWAP()
        self._trend_ema = IncrementalEMA(self._trend_period)
        self._session_trades = 0

    def on_bar(self, state: MarketState, submit_order: SubmitOrderFn) -> None:
        bar = state.bar

        # Only trade during RTH
        if not state.session.is_rth:
            return

        # Reset trade counter on session open
        if state.session.session_open_bar:
            self._session_trades = 0

        # Update VWAP (auto-resets on session open)
        typical_price = (bar.high + bar.low + bar.close) / 3.0
        vwap_val = self._vwap.update(
            typical_price, bar.volume, session_open=state.session.session_open_bar
        )

        # Update trend filter
        trend_val = self._trend_ema.update(bar.close)

        if vwap_val is None or not self._vwap.ready:
            return

        std = self._vwap.std_dev
        if std <= 0:
            return

        # Distance from VWAP in standard deviations
        z_score = (bar.close - vwap_val) / std

        # Handle margin calls
        if state.margin_call_active and state.position_quantity > 0:
            side = OrderSide.SELL if state.position_direction == 1 else OrderSide.BUY
            order = Order(
                side=side,
                quantity=state.position_quantity,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.IOC,
            )
            submit_order(order)
            return

        # Don't trade too close to session close
        if state.session.minutes_to_session_close < 15:
            # Flatten before close
            if state.position_quantity > 0:
                side = (
                    OrderSide.SELL
                    if state.position_direction == 1
                    else OrderSide.BUY
                )
                order = Order(
                    side=side,
                    quantity=state.position_quantity,
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.IOC,
                )
                submit_order(order)
            return

        # Entry logic
        if state.position_quantity == 0:
            if self._session_trades >= self._max_trades:
                return

            # Short entry: price extended above VWAP
            if z_score >= self._entry_std:
                order = Order(
                    side=OrderSide.SELL,
                    quantity=self._contracts,
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.GTC,
                )
                result = submit_order(order)
                if result:
                    self._session_trades += 1
                    logger.debug(
                        "VWAP reversion SHORT at bar %d, z=%.2f, close=%.2f, vwap=%.2f",
                        state.bar_index,
                        z_score,
                        bar.close,
                        vwap_val,
                    )

            # Long entry: price extended below VWAP
            elif z_score <= -self._entry_std:
                order = Order(
                    side=OrderSide.BUY,
                    quantity=self._contracts,
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.GTC,
                )
                result = submit_order(order)
                if result:
                    self._session_trades += 1
                    logger.debug(
                        "VWAP reversion LONG at bar %d, z=%.2f, close=%.2f, vwap=%.2f",
                        state.bar_index,
                        z_score,
                        bar.close,
                        vwap_val,
                    )

        # Exit logic: price reverts toward VWAP
        elif state.position_direction == 1 and z_score >= -self._exit_std:
            order = Order(
                side=OrderSide.SELL,
                quantity=state.position_quantity,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
            )
            submit_order(order)
            logger.debug(
                "VWAP reversion EXIT LONG at bar %d, z=%.2f", state.bar_index, z_score
            )

        elif state.position_direction == -1 and z_score <= self._exit_std:
            order = Order(
                side=OrderSide.BUY,
                quantity=state.position_quantity,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
            )
            submit_order(order)
            logger.debug(
                "VWAP reversion EXIT SHORT at bar %d, z=%.2f",
                state.bar_index,
                z_score,
            )

    def on_fill(self, fill: Fill) -> None:
        logger.debug(
            "VWAP fill: %s %d @ %.2f", fill.side.value, fill.quantity, fill.fill_price
        )
