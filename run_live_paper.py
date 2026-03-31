#!/usr/bin/env python3
"""Live paper trading with Databento real-time data.

Streams 1-minute OHLCV bars from Databento's live API for NQ, ES, RTY, YM
and feeds them into the V3 contrarian mean-reversion strategy with a
simulated $100,000 paper trading account.

Usage:
  export DATABENTO_API_KEY=your_key_here
  python run_live_paper.py --instruments NQ ES RTY YM --signal-only YM

Requires:
  pip install databento
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import databento as db

from engine.account import AccountTracker, Trade
from engine.config import SlippageConfig, SlippageModel
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
from engine.multi_state import InstrumentState, MultiInstrumentState
from engine.session_handler import SessionContext, SessionHandler
from strategies.contrarian_reversion_v3 import ContraMeanReversionV3

logger = logging.getLogger(__name__)

# Databento fixed-precision price scale
_DBN_PRICE_SCALE = 1e-9

# CME Globex dataset
DATASET = "GLBX.MDP3"


@dataclass
class _LiveInstrumentContext:
    """Per-instrument state for live engine."""

    symbol: str
    spec: ContractSpec
    account: AccountTracker
    executor: ExecutionSimulator
    session: SessionHandler
    pending_orders: list[Order] = field(default_factory=list)
    new_orders: list[Order] = field(default_factory=list)
    last_bar: Optional[Bar] = None


class LivePaperEngine:
    """Live paper trading engine that mirrors MultiInstrumentEngine.

    Receives real-time bars and feeds them through the same strategy
    interface used in backtesting, with simulated order execution.
    """

    def __init__(
        self,
        strategy: ContraMeanReversionV3,
        instruments: list[str],
        initial_capital: float = 100_000.0,
        max_position_size: int = 20,
        registry: ContractRegistry | None = None,
        log_dir: str = "live_paper_logs",
    ) -> None:
        self._strategy = strategy
        self._instrument_symbols = sorted(instruments)
        self._registry = registry or ContractRegistry()
        self._initial_capital = initial_capital
        self._equity = initial_capital
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Per-instrument contexts
        self._instruments: dict[str, _LiveInstrumentContext] = {}
        for symbol in self._instrument_symbols:
            spec = self._registry.get(symbol)
            account = AccountTracker(
                initial_capital=initial_capital,
                contract_spec=spec,
                max_position_size=max_position_size,
                enforce_daily_settlement=True,
            )
            executor = ExecutionSimulator(
                contract_spec=spec,
                slippage_config=SlippageConfig(
                    model=SlippageModel.FIXED, fixed_ticks=0.0
                ),
            )
            session = SessionHandler(spec)
            self._instruments[symbol] = _LiveInstrumentContext(
                symbol=symbol,
                spec=spec,
                account=account,
                executor=executor,
                session=session,
            )

        self._bar_index = 0
        self._pending_bars: dict[str, Bar] = {}
        self._trade_log: list[dict] = []
        self._running = False

        # Initialize strategy
        self._strategy.on_init(self._instrument_symbols)

    def process_bar_group(self, bars: dict[str, Bar]) -> None:
        """Process a synchronized group of bars (one per instrument).

        This mirrors MultiInstrumentEngine._process_multi_bar().
        """
        self._bar_index += 1
        bar_index = self._bar_index

        # 1. Process pending orders (fill against this bar)
        for symbol, bar in bars.items():
            ctx = self._instruments[symbol]
            ctx.last_bar = bar

            fills = ctx.executor.process_orders(
                ctx.pending_orders, bar, bar_index
            )
            for fill in fills:
                trades = ctx.account.process_fill(fill, bar_index)
                self._strategy.on_fill(symbol, fill)
                self._log_fill(symbol, fill)
                for trade in trades:
                    self._log_trade(symbol, trade)

            # Remove filled/cancelled
            ctx.pending_orders = [
                o for o in ctx.pending_orders
                if o.status in (OrderStatus.PENDING, OrderStatus.TRIGGERED)
            ]

            # IOC expiry
            for o in ctx.pending_orders[:]:
                if (
                    o.time_in_force == TimeInForce.IOC
                    and bar_index > o.submitted_bar_index
                ):
                    o.status = OrderStatus.CANCELLED
                    ctx.pending_orders.remove(o)

        # 2. Update session contexts and unrealized PnL
        session_ctx = None
        for symbol, bar in bars.items():
            ctx = self._instruments[symbol]
            sess = ctx.session.compute_context(bar, bar_index)

            # Expire DAY orders
            ctx.executor.expire_day_orders(
                ctx.pending_orders, sess.session_close_bar
            )
            ctx.pending_orders = [
                o for o in ctx.pending_orders
                if o.status in (OrderStatus.PENDING, OrderStatus.TRIGGERED)
            ]

            # Daily settlement
            if sess.is_daily_settlement:
                ctx.account.perform_daily_settlement(bar.close, bar.timestamp)

            ctx.account.update_unrealized_pnl(bar.close)

            if session_ctx is None:
                session_ctx = sess

        if session_ctx is None:
            return

        # 3. Sync equity
        self._sync_equity()

        # 4. Build state
        state = self._build_state(bars, bar_index, session_ctx)

        # 5. Clear new orders
        for ctx in self._instruments.values():
            ctx.new_orders = []

        # 6. Submit order function
        def submit_order(symbol: str, order: Order) -> str | None:
            return self._submit_order(symbol, order, bar_index)

        # 7. Call strategy
        self._strategy.on_bar(state, submit_order)

        # 8. Collect new orders
        for ctx in self._instruments.values():
            ctx.pending_orders.extend(ctx.new_orders)

        # 9. Record equity
        for symbol, bar in bars.items():
            ctx = self._instruments[symbol]
            settlement = (
                bar.close if session_ctx.is_daily_settlement else None
            )
            ctx.account.record_equity_point(bar.timestamp, settlement)

    def _submit_order(
        self, symbol: str, order: Order, bar_index: int
    ) -> str | None:
        if symbol not in self._instruments:
            logger.warning("Unknown instrument '%s' in order", symbol)
            return None

        ctx = self._instruments[symbol]

        error = ctx.executor.validate_and_prepare_order(order, bar_index)
        if error:
            logger.warning("Order rejected for %s: %s", symbol, error)
            order.status = OrderStatus.REJECTED
            return None

        ok, reason = ctx.account.check_margin_for_order(
            order.side, order.quantity, is_intraday=True
        )
        if not ok:
            logger.warning("Order rejected for %s: %s", symbol, reason)
            order.status = OrderStatus.REJECTED
            return None

        ctx.new_orders.append(order)
        logger.info(
            "ORDER SUBMITTED: %s %s %d @ %s [%s]",
            symbol, order.side.value, order.quantity,
            order.order_type.value, order.order_id,
        )
        return order.order_id

    def _sync_equity(self) -> None:
        total_realized = sum(
            ctx.account.realized_pnl for ctx in self._instruments.values()
        )
        total_unrealized = sum(
            ctx.account.unrealized_pnl for ctx in self._instruments.values()
        )
        self._equity = self._initial_capital + total_realized + total_unrealized

    def _build_state(
        self, bars: dict[str, Bar], bar_index: int, session_ctx: SessionContext
    ) -> MultiInstrumentState:
        instrument_states: dict[str, InstrumentState] = {}

        for symbol, ctx in self._instruments.items():
            bar = bars.get(symbol, ctx.last_bar)
            if bar is None:
                continue

            sess = ctx.session.compute_context(bar, bar_index)
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
            timestamp=next(iter(bars.values())).timestamp,
            bar_index=bar_index,
            instruments=instrument_states,
            active_symbols=tuple(bars.keys()),
            equity=self._equity,
            available_margin=max(0.0, self._equity - total_margin),
            total_margin_used=total_margin,
            total_unrealized_pnl=total_unrealized,
            total_realized_pnl=total_realized,
            total_trade_count=total_trades,
            session=session_ctx,
        )

    def _log_fill(self, symbol: str, fill: Fill) -> None:
        entry = {
            "type": "fill",
            "time": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "side": fill.side.value,
            "quantity": fill.quantity,
            "price": fill.fill_price,
            "commission": fill.commission,
            "order_id": fill.order_id,
        }
        self._trade_log.append(entry)
        self._append_log(entry)

    def _log_trade(self, symbol: str, trade: Trade) -> None:
        entry = {
            "type": "trade",
            "time": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "side": trade.side,
            "quantity": trade.quantity,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "gross_pnl": trade.gross_pnl,
            "net_pnl": trade.net_pnl,
            "commissions": trade.commissions,
        }
        self._trade_log.append(entry)
        self._append_log(entry)

    def _append_log(self, entry: dict) -> None:
        log_file = self._log_dir / "live_trades.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def print_status(self) -> None:
        """Print current account status."""
        self._sync_equity()
        total_realized = sum(
            ctx.account.realized_pnl for ctx in self._instruments.values()
        )
        total_unrealized = sum(
            ctx.account.unrealized_pnl for ctx in self._instruments.values()
        )
        total_commissions = sum(
            ctx.account.total_commissions for ctx in self._instruments.values()
        )
        total_trades = sum(
            len(ctx.account.trades) for ctx in self._instruments.values()
        )

        print("\n" + "=" * 65)
        print(f"  LIVE PAPER ACCOUNT STATUS — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("=" * 65)
        print(f"  Equity:       ${self._equity:>12,.2f}")
        print(f"  Initial:      ${self._initial_capital:>12,.2f}")
        print(f"  Realized PnL: ${total_realized:>12,.2f}")
        print(f"  Unrealized:   ${total_unrealized:>12,.2f}")
        print(f"  Commissions:  ${total_commissions:>12,.2f}")
        print(f"  Total Trades: {total_trades:>12d}")
        print(f"  Bars:         {self._bar_index:>12d}")
        print()
        print("  Positions:")
        for symbol, ctx in sorted(self._instruments.items()):
            pos = ctx.account.position
            if pos.is_flat:
                print(f"    {symbol}: FLAT")
            else:
                direction = "LONG" if pos.direction == 1 else "SHORT"
                pnl = ctx.account.unrealized_pnl
                last_price = ctx.last_bar.close if ctx.last_bar else 0
                print(
                    f"    {symbol}: {direction} {pos.quantity} "
                    f"@ {pos.avg_entry_price:.2f}  "
                    f"last={last_price:.2f}  "
                    f"uPnL=${pnl:+,.2f}"
                )
        print("=" * 65)


def _extract_root_symbol(raw_symbol: str) -> str | None:
    """Extract root symbol from Databento instrument (e.g., 'NQM5' -> 'NQ')."""
    registry = ContractRegistry()
    try:
        return registry.extract_root_symbol(raw_symbol)
    except Exception:
        # Fallback: try common patterns
        for root in ["NQ", "ES", "RTY", "YM"]:
            if raw_symbol.startswith(root):
                return root
        return None


def run_live(
    api_key: str,
    instruments: list[str],
    strategy: ContraMeanReversionV3,
    capital: float = 100_000.0,
    max_contracts: int = 20,
    status_interval: int = 60,
) -> None:
    """Connect to Databento live API and run paper trading."""

    engine = LivePaperEngine(
        strategy=strategy,
        instruments=instruments,
        initial_capital=capital,
        max_position_size=max_contracts,
    )

    # Build Databento symbol subscriptions — use continuous front month
    stype = "continuous"
    db_symbols = [f"{sym}.FUT" for sym in instruments]

    print("=" * 65)
    print("  LIVE PAPER TRADING — Databento Real-Time Feed")
    print("=" * 65)
    print(f"  Dataset:     {DATASET}")
    print(f"  Instruments: {instruments}")
    print(f"  DB Symbols:  {db_symbols}")
    print(f"  Schema:      ohlcv-1m")
    print(f"  Capital:     ${capital:,.0f}")
    print(f"  Max Size:    {max_contracts} contracts/instrument")
    print(f"  Status every {status_interval}s")
    print("=" * 65)
    print()

    # Track last status print time
    last_status_time = time.time()
    bar_buffer: dict[str, Bar] = {}
    last_ts: datetime | None = None

    # Graceful shutdown
    shutdown_event = threading.Event()

    def handle_signal(signum, frame):
        print("\n\nShutting down gracefully...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("Connecting to Databento live API...")

    try:
        live_client = db.Live(key=api_key)
        live_client.subscribe(
            dataset=DATASET,
            schema="ohlcv-1m",
            symbols=db_symbols,
            stype_in=stype,
        )

        print("Connected! Waiting for data...\n")

        for record in live_client:
            if shutdown_event.is_set():
                break

            # Skip non-OHLCV records (e.g., symbol mapping, system events)
            if not hasattr(record, "open"):
                # Handle symbol mapping messages
                if hasattr(record, "stype_in_symbol"):
                    logger.info(
                        "Symbol mapping: %s -> instrument_id=%s",
                        getattr(record, "stype_in_symbol", "?"),
                        getattr(record, "instrument_id", "?"),
                    )
                continue

            # Extract timestamp
            ts = record.ts_event
            if isinstance(ts, int):
                ts_dt = datetime.fromtimestamp(ts / 1_000_000_000, tz=timezone.utc)
            else:
                ts_dt = ts

            # Extract symbol — use the hd (header) for instrument_id
            raw_symbol = getattr(record, "symbol", None)
            if raw_symbol is None:
                # Try to get symbol from the pretty_* fields or use instrument_id
                raw_symbol = str(getattr(record, "instrument_id", "UNKNOWN"))

            # Map to root symbol
            root = _extract_root_symbol(raw_symbol)
            if root is None or root not in instruments:
                continue

            # Build Bar from OHLCV record
            # Databento prices are in fixed-precision (1e-9 scale)
            o = record.open * _DBN_PRICE_SCALE
            h = record.high * _DBN_PRICE_SCALE
            l = record.low * _DBN_PRICE_SCALE
            c = record.close * _DBN_PRICE_SCALE

            # Sanity check — skip invalid bars
            if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                continue

            vol = int(record.volume)

            bar = Bar(
                timestamp=ts_dt,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=vol,
                symbol=raw_symbol,
                duration_ns=60_000_000_000,  # 1 minute
            )

            # Buffer bars and process when we have all instruments at same timestamp
            # Use minute-level grouping
            bar_minute = ts_dt.replace(second=0, microsecond=0)

            if last_ts is not None and bar_minute != last_ts and len(bar_buffer) > 0:
                # New minute started — process buffered bars
                if len(bar_buffer) >= 2:  # Need at least 2 instruments
                    engine.process_bar_group(dict(bar_buffer))

                    # Periodic status
                    now = time.time()
                    if now - last_status_time >= status_interval:
                        engine.print_status()
                        last_status_time = now

                bar_buffer.clear()

            bar_buffer[root] = bar
            last_ts = bar_minute

        # Process any remaining bars
        if len(bar_buffer) >= 2:
            engine.process_bar_group(dict(bar_buffer))

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error("Live feed error: %s", e, exc_info=True)
        print(f"\nERROR: {e}")
    finally:
        print("\n\nFinal status:")
        engine.print_status()
        strategy.on_end()

        # Save final state
        state_file = engine._log_dir / "final_state.json"
        final = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equity": engine._equity,
            "initial_capital": engine._initial_capital,
            "bars_processed": engine._bar_index,
            "positions": {},
            "per_instrument": {},
        }
        for sym, ctx in engine._instruments.items():
            pos = ctx.account.position
            final["positions"][sym] = {
                "direction": pos.direction,
                "quantity": pos.quantity,
                "avg_entry": pos.avg_entry_price,
            }
            final["per_instrument"][sym] = {
                "realized_pnl": ctx.account.realized_pnl,
                "unrealized_pnl": ctx.account.unrealized_pnl,
                "trades": len(ctx.account.trades),
                "commissions": ctx.account.total_commissions,
            }

        with open(state_file, "w") as f:
            json.dump(final, f, indent=2)
        print(f"\nState saved to {state_file}")
        print(f"Trade log at {engine._log_dir / 'live_trades.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live Paper Trading with Databento Real-Time Data",
    )

    parser.add_argument(
        "--instruments", nargs="+", default=["NQ", "ES", "RTY", "YM"],
        help="Instrument root symbols (default: NQ ES RTY YM)",
    )

    # Strategy params (defaults match best backtest results)
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--z-entry", type=float, default=2.0)
    parser.add_argument("--z-exit", type=float, default=0.3)
    parser.add_argument("--confirm-bars", type=int, default=2)
    parser.add_argument("--stop-multiple", type=float, default=4.0)
    parser.add_argument("--momentum-threshold", type=float, default=0.75)
    parser.add_argument("--min-hold-bars", type=int, default=3)
    parser.add_argument("--skip-first-minutes", type=int, default=0)

    # V3 params
    parser.add_argument("--z-spread", type=float, default=2.0)
    parser.add_argument("--volume-spike", type=float, default=1.0)
    parser.add_argument("--per-leg-exit", action="store_true", default=True)
    parser.add_argument("--no-per-leg-exit", dest="per_leg_exit", action="store_false")
    parser.add_argument("--time-weight", action="store_true", default=True)
    parser.add_argument("--no-time-weight", dest="time_weight", action="store_false")

    # V3.1 params
    parser.add_argument("--z-widening", action="store_true", default=False)
    parser.add_argument("--no-z-widening", dest="z_widening", action="store_false")
    parser.add_argument("--leg-stop-atr", type=float, default=3.0)
    parser.add_argument("--asymmetric-exit", action="store_true", default=True)
    parser.add_argument("--no-asymmetric-exit", dest="asymmetric_exit", action="store_false")
    parser.add_argument("--vol-regime", action="store_true", default=True)
    parser.add_argument("--no-vol-regime", dest="vol_regime", action="store_false")
    parser.add_argument("--vol-regime-mult", type=float, default=1.5)

    # Signal-only
    parser.add_argument("--signal-only", nargs="+", default=["YM"],
                        help="Signal-only instruments (default: YM)")

    # Paper account
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--max-contracts", type=int, default=20)

    # Live feed
    parser.add_argument("--api-key", type=str, default=None,
                        help="Databento API key (or set DATABENTO_API_KEY env var)")
    parser.add_argument("--status-interval", type=int, default=60,
                        help="Print status every N seconds (default: 60)")

    # Logging
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # API key
    api_key = args.api_key or os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: Databento API key required.")
        print("  Set DATABENTO_API_KEY environment variable or use --api-key")
        sys.exit(1)

    instruments = [s.upper() for s in args.instruments]
    signal_only = [s.upper() for s in args.signal_only]

    strategy = ContraMeanReversionV3(
        lookback_bars=args.lookback,
        z_entry_threshold=args.z_entry,
        z_exit_threshold=args.z_exit,
        confirm_bars=args.confirm_bars,
        stop_multiple=args.stop_multiple,
        momentum_threshold=args.momentum_threshold,
        min_hold_bars=args.min_hold_bars,
        skip_first_minutes=args.skip_first_minutes,
        z_spread_threshold=args.z_spread,
        volume_spike_multiple=args.volume_spike,
        per_leg_exit=args.per_leg_exit,
        time_weight_enabled=args.time_weight,
        require_z_widening=args.z_widening,
        leg_stop_atr_multiple=args.leg_stop_atr,
        asymmetric_exit=args.asymmetric_exit,
        vol_regime_filter=args.vol_regime,
        vol_regime_multiple=args.vol_regime_mult,
        signal_only_symbols=signal_only,
    )

    run_live(
        api_key=api_key,
        instruments=instruments,
        strategy=strategy,
        capital=args.capital,
        max_contracts=args.max_contracts,
        status_interval=args.status_interval,
    )


if __name__ == "__main__":
    main()
