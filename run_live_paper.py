#!/usr/bin/env python3
"""Live paper trading with Databento real-time data + Telegram alerts.

Streams 1-minute OHLCV bars from Databento's live API for NQ, ES, RTY, YM
and feeds them into the V3 contrarian mean-reversion strategy with a
simulated $100,000 paper trading account.

Usage:
  1. Fill in .env with your API keys
  2. python run_live_paper.py --instruments NQ ES RTY YM --signal-only YM

Requires:
  pip install databento python-dotenv requests
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

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

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent / ".env")

# Databento fixed-precision price scale
_DBN_PRICE_SCALE = 1e-9

# CME Globex dataset
DATASET = "GLBX.MDP3"


# ------------------------------------------------------------------ #
# Telegram notifier
# ------------------------------------------------------------------ #

class TelegramNotifier:
    """Sends trade alerts to a Telegram chat via bot API.

    Also polls for incoming commands like /acc, /pos, /trades.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)
        self._base_url = f"https://api.telegram.org/bot{bot_token}"
        self._last_update_id = 0
        self._engine: Optional["LivePaperEngine"] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if self._enabled:
            logger.info("Telegram notifications enabled (chat_id=%s)", chat_id)
        else:
            logger.warning(
                "Telegram notifications disabled — "
                "set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def send(self, message: str) -> None:
        """Send a message (non-blocking, fire-and-forget)."""
        if not self._enabled:
            return
        t = threading.Thread(target=self._send_sync, args=(message,), daemon=True)
        t.start()

    def _send_sync(self, message: str) -> None:
        try:
            resp = requests.post(
                f"{self._base_url}/sendMessage",
                json={
                    "chat_id": self._chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            if not resp.ok:
                logger.warning("Telegram send failed: %s", resp.text)
        except Exception as e:
            logger.warning("Telegram send error: %s", e)

    def set_engine(self, engine: "LivePaperEngine") -> None:
        """Attach the engine so commands can query account state."""
        self._engine = engine

    def start_polling(self) -> None:
        """Start background thread that polls for Telegram commands."""
        if not self._enabled:
            return
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True
        )
        self._poll_thread.start()
        logger.info("Telegram command polling started")

    def stop_polling(self) -> None:
        """Stop the polling thread."""
        self._stop_event.set()

    def _poll_loop(self) -> None:
        """Long-poll for incoming Telegram messages."""
        while not self._stop_event.is_set():
            try:
                resp = requests.get(
                    f"{self._base_url}/getUpdates",
                    params={
                        "offset": self._last_update_id + 1,
                        "timeout": 10,
                    },
                    timeout=15,
                )
                if not resp.ok:
                    time.sleep(5)
                    continue

                data = resp.json()
                for update in data.get("result", []):
                    self._last_update_id = update["update_id"]
                    msg = update.get("message", {})
                    text = msg.get("text", "").strip()
                    chat_id = str(msg.get("chat", {}).get("id", ""))

                    # Only respond to our chat
                    if chat_id != self._chat_id:
                        continue

                    self._handle_command(text)

            except Exception as e:
                logger.debug("Telegram poll error: %s", e)
                time.sleep(5)

    def _handle_command(self, text: str) -> None:
        """Route incoming commands."""
        cmd = text.lower().split()[0] if text else ""
        if cmd == "/acc":
            self._cmd_account()
        elif cmd == "/pos":
            self._cmd_positions()
        elif cmd == "/trades":
            self._cmd_trades()
        elif cmd == "/help":
            self._cmd_help()
        elif cmd.startswith("/"):
            self.send(
                "Unknown command. Send /help for available commands."
            )

    def _cmd_help(self) -> None:
        msg = (
            "<b>Available Commands:</b>\n"
            "/acc — Account summary (equity, PnL, win rate)\n"
            "/pos — Current open positions\n"
            "/trades — Last 5 completed trades\n"
            "/help — Show this message"
        )
        self.send(msg)

    def _cmd_account(self) -> None:
        if self._engine is None:
            self.send("Engine not ready yet.")
            return

        eng = self._engine
        eng._sync_equity()

        total_realized = sum(
            ctx.account.realized_pnl for ctx in eng._instruments.values()
        )
        total_unrealized = sum(
            ctx.account.unrealized_pnl for ctx in eng._instruments.values()
        )
        total_commissions = sum(
            ctx.account.total_commissions for ctx in eng._instruments.values()
        )
        all_trades = []
        for ctx in eng._instruments.values():
            all_trades.extend(ctx.account.trades)
        total_trades = len(all_trades)
        winners = sum(1 for t in all_trades if t.net_pnl > 0)
        win_rate = (winners / total_trades * 100) if total_trades else 0
        net_pnl = total_realized - total_commissions

        msg = (
            f"\U0001f4ca <b>ACCOUNT SUMMARY</b>\n"
            f"\n"
            f"Equity:       <b>${eng._equity:,.2f}</b>\n"
            f"Initial:      ${eng._initial_capital:,.0f}\n"
            f"Realized PnL: ${total_realized:+,.2f}\n"
            f"Unrealized:   ${total_unrealized:+,.2f}\n"
            f"Commissions:  ${total_commissions:,.2f}\n"
            f"\n"
            f"Trades: {total_trades}\n"
            f"Win Rate: {win_rate:.1f}%\n"
            f"Bars: {eng._bar_index}"
        )
        self.send(msg)

    def _cmd_positions(self) -> None:
        if self._engine is None:
            self.send("Engine not ready yet.")
            return

        eng = self._engine
        eng._sync_equity()
        lines = ["\U0001f4cb <b>POSITIONS</b>\n"]
        any_open = False

        for symbol, ctx in sorted(eng._instruments.items()):
            pos = ctx.account.position
            if pos.is_flat:
                lines.append(f"  {symbol}: FLAT")
            else:
                any_open = True
                direction = "LONG" if pos.direction == 1 else "SHORT"
                pnl = ctx.account.unrealized_pnl
                last_price = ctx.last_bar.close if ctx.last_bar else 0
                lines.append(
                    f"  {symbol}: <b>{direction} {pos.quantity}</b> "
                    f"@ {pos.avg_entry_price:,.2f}\n"
                    f"    Last: {last_price:,.2f}  uPnL: ${pnl:+,.2f}"
                )

        if not any_open:
            lines.append("\nAll positions flat.")

        self.send("\n".join(lines))

    def _cmd_trades(self) -> None:
        if self._engine is None:
            self.send("Engine not ready yet.")
            return

        eng = self._engine
        all_trades: list[tuple[str, Trade]] = []
        for sym, ctx in eng._instruments.items():
            for t in ctx.account.trades:
                all_trades.append((sym, t))

        all_trades.sort(key=lambda x: x[1].exit_time, reverse=True)
        recent = all_trades[:5]

        if not recent:
            self.send("No completed trades yet.")
            return

        lines = ["\U0001f4dd <b>LAST 5 TRADES</b>\n"]
        for sym, t in recent:
            emoji = "\u2705" if t.net_pnl >= 0 else "\u274c"
            lines.append(
                f"{emoji} <b>{sym}</b> {t.side.upper()}\n"
                f"  {t.entry_price:,.2f} -> {t.exit_price:,.2f}\n"
                f"  Net: ${t.net_pnl:+,.2f}  |  {t.hold_duration_bars} bars"
            )

        self.send("\n".join(lines))

    def send_fill(self, symbol: str, fill: Fill, equity: float) -> None:
        """Notify on order fill."""
        side_emoji = "\U0001f7e2" if fill.side == OrderSide.BUY else "\U0001f534"
        msg = (
            f"{side_emoji} <b>FILL: {symbol}</b>\n"
            f"  {fill.side.value.upper()} {fill.quantity} @ {fill.fill_price:,.2f}\n"
            f"  Commission: ${fill.commission:.2f}\n"
            f"  Equity: ${equity:,.2f}"
        )
        self.send(msg)

    def send_trade_closed(
        self, symbol: str, trade: Trade, equity: float,
        total_trades: int, win_rate: float,
    ) -> None:
        """Notify on completed round-trip trade with stats."""
        pnl_emoji = "\u2705" if trade.net_pnl >= 0 else "\u274c"
        side_label = trade.side.upper()

        msg = (
            f"{pnl_emoji} <b>TRADE CLOSED: {symbol}</b>\n"
            f"  Side: {side_label}\n"
            f"  Entry: {trade.entry_price:,.2f}\n"
            f"  Exit:  {trade.exit_price:,.2f}\n"
            f"  Qty:   {trade.quantity}\n"
            f"  Gross: ${trade.gross_pnl:+,.2f}\n"
            f"  Net:   ${trade.net_pnl:+,.2f}\n"
            f"  Commissions: ${trade.commissions:.2f}\n"
            f"  Hold:  {trade.hold_duration_bars} bars\n"
            f"\n"
            f"<b>Account:</b>\n"
            f"  Equity: ${equity:,.2f}\n"
            f"  Trades: {total_trades}\n"
            f"  Win Rate: {win_rate:.1f}%"
        )
        self.send(msg)

    def send_startup(
        self, instruments: list[str], signal_only: list[str], capital: float,
    ) -> None:
        """Notify that live paper trading has started."""
        msg = (
            f"\U0001f680 <b>LIVE PAPER TRADING STARTED</b>\n"
            f"  Instruments: {', '.join(instruments)}\n"
            f"  Signal-only: {', '.join(signal_only) if signal_only else 'None'}\n"
            f"  Capital: ${capital:,.0f}\n"
            f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        self.send(msg)

    def send_shutdown(self, equity: float, total_pnl: float, total_trades: int) -> None:
        """Notify that live paper trading has stopped."""
        msg = (
            f"\U0001f6d1 <b>LIVE PAPER TRADING STOPPED</b>\n"
            f"  Final Equity: ${equity:,.2f}\n"
            f"  Total PnL: ${total_pnl:+,.2f}\n"
            f"  Total Trades: {total_trades}\n"
            f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        self.send(msg)


# ------------------------------------------------------------------ #
# Live instrument context
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
# Live paper engine
# ------------------------------------------------------------------ #

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
        telegram: TelegramNotifier | None = None,
    ) -> None:
        self._strategy = strategy
        self._instrument_symbols = sorted(instruments)
        self._registry = registry or ContractRegistry()
        self._initial_capital = initial_capital
        self._equity = initial_capital
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._telegram = telegram or TelegramNotifier("", "")

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

                # Telegram: fill alert
                self._sync_equity()
                self._telegram.send_fill(symbol, fill, self._equity)

                for trade in trades:
                    self._log_trade(symbol, trade)

                    # Telegram: trade closed alert with stats
                    self._sync_equity()
                    total_trades = sum(
                        len(c.account.trades) for c in self._instruments.values()
                    )
                    all_trades = []
                    for c in self._instruments.values():
                        all_trades.extend(c.account.trades)
                    winners = sum(1 for t in all_trades if t.net_pnl > 0)
                    win_rate = (winners / len(all_trades) * 100) if all_trades else 0
                    self._telegram.send_trade_closed(
                        symbol, trade, self._equity, total_trades, win_rate,
                    )

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
            "hold_duration_bars": trade.hold_duration_bars,
        }
        self._trade_log.append(entry)
        self._append_log(entry)

    def _append_log(self, entry: dict) -> None:
        log_file = self._log_dir / "live_trades.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------ #
    # State persistence — save/restore for resume across restarts
    # ------------------------------------------------------------------ #

    def save_state(self) -> None:
        """Save full engine state to disk for resume."""
        self._sync_equity()
        state = {
            "version": 2,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "initial_capital": self._initial_capital,
            "equity": self._equity,
            "bar_index": self._bar_index,
            "instruments": {},
        }

        for sym, ctx in self._instruments.items():
            pos = ctx.account.position
            acct = ctx.account

            # Serialize completed trades
            trades_data = []
            for t in acct.trades:
                trades_data.append({
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "side": t.side,
                    "gross_pnl": t.gross_pnl,
                    "net_pnl": t.net_pnl,
                    "commissions": t.commissions,
                    "slippage_ticks": t.slippage_ticks,
                    "slippage_dollars": t.slippage_dollars,
                    "hold_duration_bars": t.hold_duration_bars,
                    "entry_bar_index": t.entry_bar_index,
                    "exit_bar_index": t.exit_bar_index,
                    "notional_value": t.notional_value,
                })

            # Serialize equity curve
            eq_curve = []
            for ts, eq, settle in acct.equity_curve:
                eq_curve.append({
                    "ts": ts.isoformat() if ts else None,
                    "equity": eq,
                    "settlement": settle,
                })

            # Pending entry (open position cost tracking)
            pending_entry = None
            if acct._pending_entry is not None:
                pe = acct._pending_entry
                pending_entry = {
                    "time": pe.time.isoformat(),
                    "price": pe.price,
                    "bar_index": pe.bar_index,
                    "quantity": pe.quantity,
                    "side": pe.side,
                    "total_slippage_ticks": pe.total_slippage_ticks,
                    "total_slippage_dollars": pe.total_slippage_dollars,
                    "total_commissions": pe.total_commissions,
                }

            state["instruments"][sym] = {
                "position": {
                    "direction": pos.direction,
                    "quantity": pos.quantity,
                    "avg_entry_price": pos.avg_entry_price,
                },
                "realized_pnl": acct.realized_pnl,
                "unrealized_pnl": acct.unrealized_pnl,
                "total_commissions": acct.total_commissions,
                "total_slippage_dollars": acct.total_slippage_dollars,
                "margin_used": acct.margin_used,
                "last_settlement_price": acct._last_settlement_price,
                "daily_pnl": acct._daily_pnl,
                "total_margin_calls": acct.total_margin_calls,
                "total_forced_liquidations": acct.total_forced_liquidations,
                "trades": trades_data,
                "equity_curve": eq_curve,
                "pending_entry": pending_entry,
            }

        state_file = self._log_dir / "engine_state.json"
        tmp_file = self._log_dir / "engine_state.json.tmp"
        with open(tmp_file, "w") as f:
            json.dump(state, f, indent=2)
        tmp_file.rename(state_file)
        logger.info("State saved to %s", state_file)

    def restore_state(self) -> bool:
        """Restore engine state from disk. Returns True if restored."""
        state_file = self._log_dir / "engine_state.json"
        if not state_file.exists():
            return False

        try:
            with open(state_file) as f:
                state = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Failed to load state: %s", e)
            return False

        if state.get("version", 1) < 2:
            logger.warning("Old state format, cannot resume")
            return False

        self._bar_index = state["bar_index"]
        self._initial_capital = state["initial_capital"]

        for sym, inst_state in state["instruments"].items():
            if sym not in self._instruments:
                continue

            ctx = self._instruments[sym]
            acct = ctx.account

            # Restore position
            pos_data = inst_state["position"]
            acct.position.direction = pos_data["direction"]
            acct.position.quantity = pos_data["quantity"]
            acct.position.avg_entry_price = pos_data["avg_entry_price"]

            # Restore account state
            acct.realized_pnl = inst_state["realized_pnl"]
            acct.unrealized_pnl = inst_state["unrealized_pnl"]
            acct.total_commissions = inst_state["total_commissions"]
            acct.total_slippage_dollars = inst_state["total_slippage_dollars"]
            acct.margin_used = inst_state["margin_used"]
            acct._last_settlement_price = inst_state["last_settlement_price"]
            acct._daily_pnl = inst_state["daily_pnl"]
            acct.total_margin_calls = inst_state["total_margin_calls"]
            acct.total_forced_liquidations = inst_state["total_forced_liquidations"]

            # Restore trades
            acct.trades = []
            for td in inst_state["trades"]:
                acct.trades.append(Trade(
                    entry_time=datetime.fromisoformat(td["entry_time"]),
                    exit_time=datetime.fromisoformat(td["exit_time"]),
                    entry_price=td["entry_price"],
                    exit_price=td["exit_price"],
                    quantity=td["quantity"],
                    side=td["side"],
                    gross_pnl=td["gross_pnl"],
                    net_pnl=td["net_pnl"],
                    commissions=td["commissions"],
                    slippage_ticks=td["slippage_ticks"],
                    slippage_dollars=td["slippage_dollars"],
                    hold_duration_bars=td["hold_duration_bars"],
                    entry_bar_index=td["entry_bar_index"],
                    exit_bar_index=td["exit_bar_index"],
                    notional_value=td["notional_value"],
                ))

            # Restore equity curve
            acct.equity_curve = []
            for ec in inst_state["equity_curve"]:
                ts = datetime.fromisoformat(ec["ts"]) if ec["ts"] else None
                acct.equity_curve.append((ts, ec["equity"], ec["settlement"]))

            # Restore pending entry
            pe_data = inst_state.get("pending_entry")
            if pe_data:
                from engine.account import _PendingEntry
                acct._pending_entry = _PendingEntry(
                    time=datetime.fromisoformat(pe_data["time"]),
                    price=pe_data["price"],
                    bar_index=pe_data["bar_index"],
                    quantity=pe_data["quantity"],
                    side=pe_data["side"],
                    total_slippage_ticks=pe_data["total_slippage_ticks"],
                    total_slippage_dollars=pe_data["total_slippage_dollars"],
                    total_commissions=pe_data["total_commissions"],
                )
            else:
                acct._pending_entry = None

        self._sync_equity()

        saved_time = state.get("timestamp", "unknown")
        total_trades = sum(
            len(ctx.account.trades) for ctx in self._instruments.values()
        )
        logger.info(
            "State restored from %s — equity=$%.2f, bars=%d, trades=%d",
            saved_time, self._equity, self._bar_index, total_trades,
        )
        print(f"\n  RESUMED from saved state ({saved_time})")
        print(f"  Equity: ${self._equity:,.2f}  |  Bars: {self._bar_index}  |  Trades: {total_trades}")
        print()
        return True

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
    signal_only: list[str],
    strategy: ContraMeanReversionV3,
    capital: float = 100_000.0,
    max_contracts: int = 20,
    status_interval: int = 60,
    telegram: TelegramNotifier | None = None,
    resume: bool = False,
) -> None:
    """Connect to Databento live API and run paper trading."""

    tg = telegram or TelegramNotifier("", "")

    engine = LivePaperEngine(
        strategy=strategy,
        instruments=instruments,
        initial_capital=capital,
        max_position_size=max_contracts,
        telegram=tg,
    )

    # Resume from saved state if requested
    if resume:
        if not engine.restore_state():
            print("No saved state found — starting fresh")

    # Let telegram commands query the engine, start listening
    tg.set_engine(engine)
    tg.start_polling()

    # Build Databento symbol subscriptions — use continuous front month
    stype = "continuous"
    db_symbols = [f"{sym}.c.0" for sym in instruments]

    print("=" * 65)
    print("  LIVE PAPER TRADING — Databento Real-Time Feed")
    print("=" * 65)
    print(f"  Dataset:     {DATASET}")
    print(f"  Instruments: {instruments}")
    if signal_only:
        print(f"  Signal-only: {signal_only}")
    print(f"  DB Symbols:  {db_symbols}")
    print(f"  Schema:      ohlcv-1m")
    print(f"  Capital:     ${capital:,.0f}")
    print(f"  Max Size:    {max_contracts} contracts/instrument")
    print(f"  Telegram:    {'ON' if tg.enabled else 'OFF'}")
    print(f"  Status every {status_interval}s")
    print("=" * 65)
    print()

    # Telegram startup alert
    tg.send_startup(instruments, signal_only, capital)

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

            # Extract symbol — try multiple attributes
            raw_symbol = None
            for attr in ("pretty_symbol", "symbol", "stype_in_symbol"):
                val = getattr(record, attr, None)
                if val and isinstance(val, str) and val != "":
                    raw_symbol = val
                    break

            # Fall back to instrument_id -> symbology lookup
            if raw_symbol is None:
                iid = getattr(record, "instrument_id", None)
                if iid is not None and hasattr(live_client, "symbology_map"):
                    raw_symbol = live_client.symbology_map.get(iid)
                if raw_symbol is None:
                    raw_symbol = str(iid) if iid is not None else "UNKNOWN"

            # Map to root symbol
            root = _extract_root_symbol(raw_symbol)
            if root is None or root not in instruments:
                logger.debug("Skipping record: raw_symbol=%s, root=%s", raw_symbol, root)
                continue

            # Build Bar from OHLCV record
            # Databento prices: if int, they're fixed-precision (1e-9 scale)
            # If already float, use as-is
            raw_o = record.open
            if isinstance(raw_o, int) and raw_o > 1_000_000:
                o = raw_o * _DBN_PRICE_SCALE
                h = record.high * _DBN_PRICE_SCALE
                l = record.low * _DBN_PRICE_SCALE
                c = record.close * _DBN_PRICE_SCALE
            else:
                o = float(raw_o)
                h = float(record.high)
                l = float(record.low)
                c = float(record.close)

            if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                continue

            logger.debug("BAR: %s root=%s O=%.2f H=%.2f L=%.2f C=%.2f V=%s",
                         raw_symbol, root, o, h, l, c, record.volume)

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
            bar_minute = ts_dt.replace(second=0, microsecond=0)

            if last_ts is not None and bar_minute != last_ts and len(bar_buffer) > 0:
                # New minute started — process buffered bars
                if len(bar_buffer) >= 2:
                    engine.process_bar_group(dict(bar_buffer))

                    # Print first 5 bars to confirm data is flowing
                    if engine._bar_index <= 5:
                        ts_str = last_ts.strftime("%H:%M") if last_ts else "?"
                        parts = [f"{s} {bar_buffer[s].close:,.2f}" for s in sorted(bar_buffer)]
                        print(f"  [{engine._bar_index}/5] {ts_str} | {' | '.join(parts)}")
                        if engine._bar_index == 5:
                            print("  Data confirmed OK — running silently now.\n")

                    # Periodic status + auto-save
                    now = time.time()
                    if now - last_status_time >= status_interval:
                        engine.print_status()
                        engine.save_state()
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
        tg.stop_polling()
        print("\n\nFinal status:")
        engine.print_status()
        strategy.on_end()

        # Telegram shutdown alert
        engine._sync_equity()
        total_realized = sum(
            ctx.account.realized_pnl for ctx in engine._instruments.values()
        )
        total_trades = sum(
            len(ctx.account.trades) for ctx in engine._instruments.values()
        )
        tg.send_shutdown(engine._equity, total_realized, total_trades)

        # Save state for resume
        engine.save_state()
        print(f"\nState saved — restart with --resume to continue")
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

    # Daily profit cap
    parser.add_argument("--daily-profit-cap", type=float, default=1000.0,
                        help="Stop trading after this daily profit (default: 1000)")

    # Paper account
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--max-contracts", type=int, default=20)

    # Live feed
    parser.add_argument("--status-interval", type=int, default=60,
                        help="Print status every N seconds (default: 60)")

    # Resume
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved state in live_paper_logs/")

    # Logging
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # API keys from .env
    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key:
        print("ERROR: DATABENTO_API_KEY not set in .env")
        sys.exit(1)

    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    telegram = TelegramNotifier(tg_token, tg_chat_id)

    if not telegram.enabled:
        print("WARNING: Telegram not configured — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        print()

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
        daily_profit_cap=args.daily_profit_cap,
    )

    run_live(
        api_key=api_key,
        instruments=instruments,
        signal_only=signal_only,
        strategy=strategy,
        capital=args.capital,
        max_contracts=args.max_contracts,
        status_interval=args.status_interval,
        telegram=telegram,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
