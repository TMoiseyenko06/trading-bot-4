"""Core backtesting engine: single-threaded synchronous chronological loop."""

from __future__ import annotations

import logging
from typing import Callable

from engine.account import AccountTracker
from engine.config import EngineConfig, SessionFilter
from engine.contract_registry import ContractRegistry
from engine.data_feed import Bar, DataFeed
from engine.execution import (
    ExecutionSimulator,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from engine.market_state import MarketState
from engine.session_handler import SessionHandler
from strategies.base import Strategy

logger = logging.getLogger(__name__)


class _StrategyContext:
    """Internal per-strategy state managed by the engine."""

    def __init__(
        self,
        strategy: Strategy,
        account: AccountTracker,
        executor: ExecutionSimulator,
        session: SessionHandler,
    ) -> None:
        self.strategy = strategy
        self.account = account
        self.executor = executor
        self.session = session
        self.pending_orders: list[Order] = []
        self.new_orders: list[Order] = []


class BacktestEngine:
    """Multi-strategy backtesting engine for futures.

    Performs a single forward pass through the data. All strategies
    process the same bar before the loop advances. Zero lookahead bias
    by design.
    """

    def __init__(
        self,
        dbn_path: str,
        strategies: list[Strategy],
        config: EngineConfig,
        registry: ContractRegistry | None = None,
    ) -> None:
        self._dbn_path = dbn_path
        self._config = config
        self._registry = registry or ContractRegistry()
        self._contract_spec = self._registry.get(config.instrument)

        # Initialize per-strategy contexts
        self._contexts: list[_StrategyContext] = []
        for strat in strategies:
            account = AccountTracker(
                initial_capital=config.initial_capital,
                contract_spec=self._contract_spec,
                max_position_size=config.max_position_size,
                enforce_daily_settlement=config.enforce_daily_settlement,
            )
            executor = ExecutionSimulator(
                contract_spec=self._contract_spec,
                slippage_config=config.slippage,
                queue_position_penalty=config.queue_position_penalty,
            )
            session = SessionHandler(self._contract_spec)
            self._contexts.append(
                _StrategyContext(strat, account, executor, session)
            )

    def run(self) -> dict[str, object]:
        """Execute the backtest. Returns {strategy_name: result_data}."""
        feed = DataFeed(self._dbn_path)

        # Initialize all strategies
        for ctx in self._contexts:
            ctx.strategy.on_init()

        bar_index = 0
        for bar in feed:
            # Session filtering
            if not self._passes_session_filter(bar, bar_index):
                bar_index += 1
                continue

            # For each strategy: process pending orders, update state, call on_bar
            for ctx in self._contexts:
                self._process_strategy_bar(ctx, bar, bar_index)

            bar_index += 1

        # Finalize
        results: dict[str, object] = {}
        for ctx in self._contexts:
            ctx.strategy.on_end()
            results[ctx.strategy.name] = self._build_result(ctx)

        logger.info("Backtest complete: %d bars processed", bar_index)
        return results

    def _process_strategy_bar(
        self, ctx: _StrategyContext, bar: Bar, bar_index: int
    ) -> None:
        """Process a single bar for one strategy."""

        # 1. Compute session context
        session_ctx = ctx.session.compute_context(bar, bar_index)

        # 2. Process pending orders from previous bars (fills happen here)
        fills = ctx.executor.process_orders(ctx.pending_orders, bar, bar_index)
        for fill in fills:
            trades = ctx.account.process_fill(fill, bar_index)
            ctx.strategy.on_fill(fill)

        # Remove filled/cancelled orders
        ctx.pending_orders = [
            o
            for o in ctx.pending_orders
            if o.status in (OrderStatus.PENDING, OrderStatus.TRIGGERED)
        ]

        # 3. Handle IOC orders that didn't fill
        for o in ctx.pending_orders[:]:
            if o.time_in_force == TimeInForce.IOC and bar_index > o.submitted_bar_index:
                o.status = OrderStatus.CANCELLED
                ctx.pending_orders.remove(o)

        # 4. Expire DAY orders at session close
        ctx.executor.expire_day_orders(ctx.pending_orders, session_ctx.session_close_bar)
        ctx.pending_orders = [
            o
            for o in ctx.pending_orders
            if o.status in (OrderStatus.PENDING, OrderStatus.TRIGGERED)
        ]

        # 5. Check for forced liquidation
        if ctx.account.should_force_liquidate(bar_index):
            self._force_liquidate(ctx, bar, bar_index)

        # 6. Daily settlement
        if session_ctx.is_daily_settlement and self._config.enforce_daily_settlement:
            ctx.account.perform_daily_settlement(bar.close, bar.timestamp)

        # 7. Update unrealized PnL
        ctx.account.update_unrealized_pnl(bar.close)

        # 8. Check margin call
        ctx.account.check_margin_call(bar_index)

        # 9. Contract roll handling
        if session_ctx.contract_roll_pending:
            # If strategy still has position from the old contract, flatten
            if not ctx.account.position.is_flat:
                logger.warning(
                    "Strategy '%s' has open position during contract roll. "
                    "Auto-flattening.",
                    ctx.strategy.name,
                )
                self._force_liquidate(ctx, bar, bar_index)

        # 10. Build market state snapshot
        state = self._build_market_state(ctx, bar, bar_index, session_ctx)

        # 11. Clear new orders collector
        ctx.new_orders = []

        # 12. Create order submission function
        def submit_order(order: Order, _ctx=ctx, _bi=bar_index) -> str | None:
            return self._submit_order(_ctx, order, _bi)

        # 13. Call strategy's on_bar
        ctx.strategy.on_bar(state, submit_order)

        # 14. Collect new orders into pending
        ctx.pending_orders.extend(ctx.new_orders)

        # 15. Record equity curve point
        settlement_mark = (
            bar.close if session_ctx.is_daily_settlement else None
        )
        ctx.account.record_equity_point(bar.timestamp, settlement_mark)

    def _submit_order(
        self, ctx: _StrategyContext, order: Order, bar_index: int
    ) -> str | None:
        """Validate and accept an order from a strategy.

        Returns the order_id on success, or None with a warning on rejection.
        """
        # Validate prices and set metadata
        error = ctx.executor.validate_and_prepare_order(order, bar_index)
        if error:
            logger.warning(
                "Order rejected for '%s': %s", ctx.strategy.name, error
            )
            order.status = OrderStatus.REJECTED
            return None

        # Margin check
        is_intraday = True  # Simplified; could check session context
        ok, reason = ctx.account.check_margin_for_order(
            order.side, order.quantity, is_intraday
        )
        if not ok:
            logger.warning(
                "Order rejected for '%s': %s", ctx.strategy.name, reason
            )
            order.status = OrderStatus.REJECTED
            return None

        # Accept order
        ctx.new_orders.append(order)
        return order.order_id

    def _force_liquidate(
        self, ctx: _StrategyContext, bar: Bar, bar_index: int
    ) -> None:
        """Force-liquidate a strategy's position at the bar's open."""
        pos = ctx.account.position
        if pos.is_flat:
            return
        side = OrderSide.SELL if pos.direction == 1 else OrderSide.BUY
        order = Order(
            side=side,
            quantity=pos.quantity,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.IOC,
        )
        order.submitted_bar_index = bar_index - 1  # Allow immediate fill
        ctx.executor.validate_and_prepare_order(order, bar_index - 1)
        fills = ctx.executor.process_orders([order], bar, bar_index)
        for fill in fills:
            ctx.account.process_fill(fill, bar_index)
            ctx.strategy.on_fill(fill)

    def _build_market_state(
        self,
        ctx: _StrategyContext,
        bar: Bar,
        bar_index: int,
        session_ctx,
    ) -> MarketState:
        pos = ctx.account.position
        return MarketState(
            bar=bar,
            bar_index=bar_index,
            position_direction=pos.direction,
            position_quantity=pos.quantity,
            position_avg_entry_price=pos.avg_entry_price,
            open_orders=tuple(ctx.pending_orders),
            equity=ctx.account.equity,
            available_margin=ctx.account.available_margin,
            margin_used=ctx.account.margin_used,
            maintenance_margin_remaining=max(
                0.0, ctx.account.equity - ctx.account.maintenance_margin
            ),
            unrealized_pnl=ctx.account.unrealized_pnl,
            realized_pnl=ctx.account.realized_pnl,
            trade_history=tuple(ctx.account.trades),
            session=session_ctx,
            margin_call_active=ctx.account.margin_call_active,
            contract_roll_pending=session_ctx.contract_roll_pending,
            notional_value=ctx.account.get_notional_value(bar.close),
        )

    def _passes_session_filter(self, bar: Bar, bar_index: int) -> bool:
        """Check if bar passes the configured session filter."""
        if self._config.session_filter == SessionFilter.FULL_GLOBEX:
            return True

        # We need a temporary session handler to check
        # Use the first context's session handler for filtering
        if not self._contexts:
            return True

        ctx = self._contexts[0]
        session_ctx = ctx.session.compute_context(bar, bar_index)

        if self._config.session_filter == SessionFilter.RTH_ONLY:
            return session_ctx.is_rth
        if self._config.session_filter == SessionFilter.ETH_ONLY:
            return session_ctx.is_eth
        return True

    def _build_result(self, ctx: _StrategyContext) -> dict:
        """Build the result dictionary for a strategy."""
        return {
            "strategy_name": ctx.strategy.name,
            "equity_curve": ctx.account.equity_curve,
            "trades": ctx.account.trades,
            "final_equity": ctx.account.equity,
            "initial_capital": ctx.account.initial_capital,
            "realized_pnl": ctx.account.realized_pnl,
            "unrealized_pnl": ctx.account.unrealized_pnl,
            "total_commissions": ctx.account.total_commissions,
            "total_slippage_dollars": ctx.account.total_slippage_dollars,
            "position": ctx.account.position,
            "total_margin_calls": ctx.account.total_margin_calls,
            "total_forced_liquidations": ctx.account.total_forced_liquidations,
        }
