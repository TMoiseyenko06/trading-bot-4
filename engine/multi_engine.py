"""Multi-instrument backtesting engine.

Extends the core engine to support strategies that trade across multiple
instruments simultaneously, each with independent position tracking and
execution simulation but sharing a single account equity pool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from engine.account import AccountTracker, Trade
from engine.config import EngineConfig, SessionFilter, SlippageConfig
from engine.contract_registry import ContractRegistry, ContractSpec
from engine.data_feed import Bar
from engine.execution import (
    ExecutionSimulator,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from engine.multi_feed import MultiBar, MultiInstrumentFeed
from engine.multi_state import InstrumentState, MultiInstrumentState
from engine.session_handler import SessionHandler

logger = logging.getLogger(__name__)


# Type for order submission: takes (symbol, order) -> order_id or None
MultiSubmitFn = Callable[[str, Order], Optional[str]]


@dataclass
class _InstrumentContext:
    """Per-instrument engine state."""

    symbol: str
    spec: ContractSpec
    account: AccountTracker
    executor: ExecutionSimulator
    session: SessionHandler
    pending_orders: list[Order] = field(default_factory=list)
    new_orders: list[Order] = field(default_factory=list)
    last_bar: Optional[Bar] = None


class MultiInstrumentEngine:
    """Multi-instrument backtesting engine for basket strategies.

    Reads multiple .dbn files simultaneously, synchronizes by timestamp,
    and processes all instruments at each time step. Maintains per-instrument
    positions with a shared equity pool.
    """

    def __init__(
        self,
        dbn_paths: dict[str, str],
        strategy: "MultiInstrumentStrategy",
        config: EngineConfig,
        registry: ContractRegistry | None = None,
    ) -> None:
        from strategies.multi_base import MultiInstrumentStrategy

        self._dbn_paths = dbn_paths
        self._strategy = strategy
        self._config = config
        self._registry = registry or ContractRegistry()

        # Per-instrument contexts
        self._instruments: dict[str, _InstrumentContext] = {}
        for symbol in dbn_paths:
            spec = self._registry.get(symbol)
            # Each instrument gets its own account tracker with shared
            # initial capital divided equally for margin purposes,
            # but we track a unified equity separately
            account = AccountTracker(
                initial_capital=config.initial_capital,
                contract_spec=spec,
                max_position_size=config.max_position_size,
                enforce_daily_settlement=config.enforce_daily_settlement,
            )
            executor = ExecutionSimulator(
                contract_spec=spec,
                slippage_config=config.slippage,
                queue_position_penalty=config.queue_position_penalty,
            )
            session = SessionHandler(spec)
            self._instruments[symbol] = _InstrumentContext(
                symbol=symbol,
                spec=spec,
                account=account,
                executor=executor,
                session=session,
            )

        # Unified account tracking
        self._equity = config.initial_capital
        self._initial_capital = config.initial_capital

    def run(self) -> dict:
        """Execute the multi-instrument backtest."""
        feed = MultiInstrumentFeed(self._dbn_paths)

        self._strategy.on_init(list(self._dbn_paths.keys()))

        bar_count = 0
        for multi_bar in feed:
            self._process_multi_bar(multi_bar)
            bar_count += 1

        self._strategy.on_end()

        logger.info(
            "Multi-instrument backtest complete: %d time steps processed",
            bar_count,
        )
        return self._build_results()

    def _process_multi_bar(self, multi_bar: MultiBar) -> None:
        """Process one synchronized time step across all instruments."""
        bar_index = multi_bar.bar_index

        # 1. Process pending orders for instruments that have bars
        all_fills: dict[str, list[Fill]] = {}
        for symbol, bar in multi_bar.bars.items():
            ctx = self._instruments[symbol]
            ctx.last_bar = bar

            # Process pending orders from previous bars
            fills = ctx.executor.process_orders(
                ctx.pending_orders, bar, bar_index
            )
            all_fills[symbol] = fills

            for fill in fills:
                trades = ctx.account.process_fill(fill, bar_index)
                self._strategy.on_fill(symbol, fill)

            # Remove filled/cancelled orders
            ctx.pending_orders = [
                o for o in ctx.pending_orders
                if o.status in (OrderStatus.PENDING, OrderStatus.TRIGGERED)
            ]

            # Handle IOC
            for o in ctx.pending_orders[:]:
                if (
                    o.time_in_force == TimeInForce.IOC
                    and bar_index > o.submitted_bar_index
                ):
                    o.status = OrderStatus.CANCELLED
                    ctx.pending_orders.remove(o)

        # 2. Update session contexts and unrealized PnL
        session_ctx = None
        for symbol, bar in multi_bar.bars.items():
            ctx = self._instruments[symbol]
            sess = ctx.session.compute_context(bar, bar_index)

            # Expire DAY orders at session close
            ctx.executor.expire_day_orders(ctx.pending_orders, sess.session_close_bar)
            ctx.pending_orders = [
                o for o in ctx.pending_orders
                if o.status in (OrderStatus.PENDING, OrderStatus.TRIGGERED)
            ]

            # Daily settlement
            if sess.is_daily_settlement and self._config.enforce_daily_settlement:
                ctx.account.perform_daily_settlement(bar.close, bar.timestamp)

            # Update unrealized PnL
            ctx.account.update_unrealized_pnl(bar.close)

            if session_ctx is None:
                session_ctx = sess

        if session_ctx is None:
            return

        # 3. Compute unified equity
        self._sync_equity()

        # 4. Build multi-instrument state
        state = self._build_state(multi_bar, session_ctx)

        # 5. Clear new orders
        for ctx in self._instruments.values():
            ctx.new_orders = []

        # 6. Create order submission function
        def submit_order(
            symbol: str, order: Order, _bi=bar_index
        ) -> str | None:
            return self._submit_order(symbol, order, _bi)

        # 7. Call strategy
        self._strategy.on_bar(state, submit_order)

        # 8. Collect new orders into pending
        for ctx in self._instruments.values():
            ctx.pending_orders.extend(ctx.new_orders)

        # 9. Record equity curve points
        for symbol, bar in multi_bar.bars.items():
            ctx = self._instruments[symbol]
            settlement = (
                bar.close if session_ctx.is_daily_settlement else None
            )
            ctx.account.record_equity_point(bar.timestamp, settlement)

    def _submit_order(
        self, symbol: str, order: Order, bar_index: int
    ) -> str | None:
        """Validate and accept an order for a specific instrument."""
        if symbol not in self._instruments:
            logger.warning("Unknown instrument '%s' in order", symbol)
            return None

        ctx = self._instruments[symbol]

        error = ctx.executor.validate_and_prepare_order(order, bar_index)
        if error:
            logger.warning("Order rejected for %s: %s", symbol, error)
            order.status = OrderStatus.REJECTED
            return None

        # Margin check using per-instrument account
        ok, reason = ctx.account.check_margin_for_order(
            order.side, order.quantity, is_intraday=True
        )
        if not ok:
            logger.warning("Order rejected for %s: %s", symbol, reason)
            order.status = OrderStatus.REJECTED
            return None

        ctx.new_orders.append(order)
        return order.order_id

    def _sync_equity(self) -> None:
        """Synchronize the unified equity from per-instrument accounts."""
        total_realized = sum(
            ctx.account.realized_pnl for ctx in self._instruments.values()
        )
        total_unrealized = sum(
            ctx.account.unrealized_pnl for ctx in self._instruments.values()
        )
        total_commissions = sum(
            ctx.account.total_commissions for ctx in self._instruments.values()
        )
        self._equity = self._initial_capital + total_realized + total_unrealized

    def _build_state(
        self, multi_bar: MultiBar, session_ctx
    ) -> MultiInstrumentState:
        """Build the read-only multi-instrument state snapshot."""
        instrument_states: dict[str, InstrumentState] = {}

        for symbol, ctx in self._instruments.items():
            bar = multi_bar.bars.get(symbol)
            if bar is None:
                # Use last known bar if no bar at this timestamp
                bar = ctx.last_bar
            if bar is None:
                continue

            sess = ctx.session.compute_context(bar, multi_bar.bar_index)
            pos = ctx.account.position

            instrument_states[symbol] = InstrumentState(
                symbol=symbol,
                bar=bar,
                position_direction=pos.direction,
                position_quantity=pos.quantity,
                position_avg_entry_price=pos.avg_entry_price,
                open_orders=tuple(ctx.pending_orders),
                unrealized_pnl=ctx.account.unrealized_pnl,
                realized_pnl=ctx.account.realized_pnl,
                trade_history=tuple(ctx.account.trades),
                session=sess,
                margin_used=ctx.account.margin_used,
                notional_value=ctx.account.get_notional_value(bar.close),
            )

        total_margin = sum(
            ctx.account.margin_used for ctx in self._instruments.values()
        )
        total_unrealized = sum(
            ctx.account.unrealized_pnl for ctx in self._instruments.values()
        )
        total_realized = sum(
            ctx.account.realized_pnl for ctx in self._instruments.values()
        )
        total_trades = sum(
            len(ctx.account.trades) for ctx in self._instruments.values()
        )

        return MultiInstrumentState(
            timestamp=multi_bar.timestamp,
            bar_index=multi_bar.bar_index,
            instruments=instrument_states,
            active_symbols=tuple(multi_bar.bars.keys()),
            equity=self._equity,
            available_margin=max(0.0, self._equity - total_margin),
            total_margin_used=total_margin,
            total_unrealized_pnl=total_unrealized,
            total_realized_pnl=total_realized,
            total_trade_count=total_trades,
            session=session_ctx,
        )

    def force_liquidate_all(self, bar_index: int) -> None:
        """Force-liquidate all positions across all instruments."""
        for symbol, ctx in self._instruments.items():
            pos = ctx.account.position
            if pos.is_flat:
                continue
            side = OrderSide.SELL if pos.direction == 1 else OrderSide.BUY
            order = Order(
                side=side,
                quantity=pos.quantity,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.IOC,
            )
            order.submitted_bar_index = bar_index - 1
            ctx.executor.validate_and_prepare_order(order, bar_index - 1)
            if ctx.last_bar is not None:
                fills = ctx.executor.process_orders(
                    [order], ctx.last_bar, bar_index
                )
                for fill in fills:
                    ctx.account.process_fill(fill, bar_index)
                    self._strategy.on_fill(symbol, fill)

    def _build_results(self) -> dict:
        """Build combined results across all instruments."""
        per_instrument = {}
        all_trades: list[Trade] = []
        total_commissions = 0.0
        total_slippage = 0.0
        total_margin_calls = 0
        total_forced_liqs = 0

        for symbol, ctx in self._instruments.items():
            per_instrument[symbol] = {
                "trades": ctx.account.trades,
                "equity_curve": ctx.account.equity_curve,
                "realized_pnl": ctx.account.realized_pnl,
                "unrealized_pnl": ctx.account.unrealized_pnl,
                "total_commissions": ctx.account.total_commissions,
                "total_slippage_dollars": ctx.account.total_slippage_dollars,
                "position": ctx.account.position,
                "total_margin_calls": ctx.account.total_margin_calls,
                "total_forced_liquidations": ctx.account.total_forced_liquidations,
            }
            all_trades.extend(ctx.account.trades)
            total_commissions += ctx.account.total_commissions
            total_slippage += ctx.account.total_slippage_dollars
            total_margin_calls += ctx.account.total_margin_calls
            total_forced_liqs += ctx.account.total_forced_liquidations

        # Merge equity curves: pick the longest one for timestamps
        longest_eq = max(
            (ctx.account.equity_curve for ctx in self._instruments.values()),
            key=len,
            default=[],
        )

        self._sync_equity()

        return {
            "strategy_name": self._strategy.name,
            "instruments": list(self._instruments.keys()),
            "per_instrument": per_instrument,
            "all_trades": sorted(all_trades, key=lambda t: t.entry_time),
            "equity_curve": longest_eq,
            "initial_capital": self._initial_capital,
            "final_equity": self._equity,
            "realized_pnl": sum(
                ctx.account.realized_pnl for ctx in self._instruments.values()
            ),
            "unrealized_pnl": sum(
                ctx.account.unrealized_pnl for ctx in self._instruments.values()
            ),
            "total_commissions": total_commissions,
            "total_slippage_dollars": total_slippage,
            "total_margin_calls": total_margin_calls,
            "total_forced_liquidations": total_forced_liqs,
        }
