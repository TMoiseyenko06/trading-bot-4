"""Intraday contrarian mean-reversion basket strategy.

Computes cross-sectional deviations from the equally-weighted market return
over a rolling signal window, volatility-normalizes, and fades outliers.
Always dollar-neutral. All instruments entered and exited as a group.
No overnight positions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from engine.execution import Fill, Order, OrderSide, OrderType, TimeInForce
from engine.multi_state import MultiInstrumentState
from strategies.multi_base import MultiInstrumentStrategy, MultiSubmitFn

logger = logging.getLogger(__name__)


@dataclass
class _InstrumentData:
    """Per-instrument rolling state for signal calculation."""

    # Signal window tracking
    window_open_price: Optional[float] = None
    window_volume: int = 0
    prior_window_volume: int = 0

    # Intraday ATR state (session-anchored)
    session_highs: list[float] = field(default_factory=list)
    session_lows: list[float] = field(default_factory=list)
    session_closes: list[float] = field(default_factory=list)
    prev_close: Optional[float] = None
    atr_value: Optional[float] = None

    # Current computed values
    return_pct: float = 0.0
    deviation: float = 0.0
    raw_weight: float = 0.0
    normalized_weight: float = 0.0
    volume_change: float = 0.0  # ln(vol_current / vol_prior)

    # Entry state (when in a trade)
    entry_deviation: float = 0.0
    entry_price: float = 0.0

    def reset_session(self) -> None:
        """Reset session-anchored data at session open."""
        self.session_highs.clear()
        self.session_lows.clear()
        self.session_closes.clear()
        self.prev_close = None
        self.atr_value = None
        self.window_open_price = None
        self.window_volume = 0
        self.prior_window_volume = 0

    def reset_window(self, open_price: float) -> None:
        """Reset for a new signal window."""
        self.prior_window_volume = max(1, self.window_volume)
        self.window_open_price = open_price
        self.window_volume = 0

    def update_atr(self, high: float, low: float, close: float) -> None:
        """Incrementally update intraday ATR using all session bars."""
        self.session_highs.append(high)
        self.session_lows.append(low)
        self.session_closes.append(close)

        n = len(self.session_closes)
        if n < 2:
            self.atr_value = high - low if high - low > 0 else None
            self.prev_close = close
            return

        # Compute true range for this bar
        tr = max(
            high - low,
            abs(high - self.prev_close) if self.prev_close else high - low,
            abs(low - self.prev_close) if self.prev_close else high - low,
        )
        self.prev_close = close

        # Simple running ATR over all session bars
        if self.atr_value is None:
            self.atr_value = tr
        else:
            # Wilder's smoothing with session length as period
            period = min(n, 14)
            self.atr_value = (
                self.atr_value * (period - 1) + tr
            ) / period


class ContraMeanReversionStrategy(MultiInstrumentStrategy):
    """Intraday contrarian mean-reversion across US equity index futures.

    Parameters:
        signal_window_minutes: Length of the signal computation window (default 30)
        theta: Deviation threshold for entry. If None, calibrates to 1 std dev
               of observed deviations during the first session.
        atr_period: Not used directly — ATR is session-anchored by default.
        stop_multiple: Exit if deviation widens beyond this * sigma (default 2.0)
        max_position_pct: Max position per instrument as fraction of capital (default 0.10)
        session_cutoff_minutes: Minutes before RTH close to flatten (default 15)
    """

    def __init__(
        self,
        name: str = "Contra_MeanRev",
        signal_window_minutes: int = 30,
        theta: Optional[float] = None,
        stop_multiple: float = 2.0,
        max_position_pct: float = 0.10,
        session_cutoff_minutes: int = 15,
    ) -> None:
        super().__init__(name)
        self._window_minutes = signal_window_minutes
        self._theta = theta
        self._theta_auto = theta is None
        self._stop_multiple = stop_multiple
        self._max_pos_pct = max_position_pct
        self._cutoff_minutes = session_cutoff_minutes

        # Runtime state
        self._symbols: list[str] = []
        self._data: dict[str, _InstrumentData] = {}
        self._in_trade: bool = False
        self._bars_in_window: int = 0
        self._bars_per_window: int = 0
        self._session_active: bool = False
        self._past_cutoff: bool = False
        self._session_pnl: float = 0.0

        # Theta calibration
        self._calibration_deviations: list[float] = []
        self._calibrated: bool = False

        # Display state
        self._display: dict = {}

    def on_init(self, symbols: list[str]) -> None:
        self._symbols = sorted(symbols)
        for sym in self._symbols:
            self._data[sym] = _InstrumentData()

        logger.info(
            "ContraMeanRev initialized: instruments=%s, window=%dm, "
            "theta=%s, stop=%.1fx, cutoff=%dm",
            self._symbols,
            self._window_minutes,
            self._theta if self._theta else "auto-calibrate",
            self._stop_multiple,
            self._cutoff_minutes,
        )

    def on_bar(
        self, state: MultiInstrumentState, submit_order: MultiSubmitFn
    ) -> None:
        # Need all instruments to have data
        if not state.has_all_instruments(self._symbols):
            return

        session = state.session

        # Handle session transitions
        if session.session_open_bar:
            self._on_session_open(state)

        if not session.is_rth:
            return

        # Infer bars per window from bar duration
        if self._bars_per_window == 0:
            any_bar = state.instruments[self._symbols[0]].bar
            bar_minutes = max(1, any_bar.duration_ns // 60_000_000_000)
            self._bars_per_window = max(1, self._window_minutes // bar_minutes)

        # Update per-instrument ATR and volume
        for sym in self._symbols:
            inst = state.instruments[sym]
            bar = inst.bar
            data = self._data[sym]
            data.update_atr(bar.high, bar.low, bar.close)
            data.window_volume += bar.volume

            # Initialize window open if needed
            if data.window_open_price is None:
                data.window_open_price = bar.open

        self._bars_in_window += 1

        # Check session cutoff
        if session.minutes_to_session_close <= self._cutoff_minutes:
            if not self._past_cutoff:
                self._past_cutoff = True
                logger.info(
                    "[%s] SESSION CUTOFF — %d min to close, flattening all",
                    self.name,
                    int(session.minutes_to_session_close),
                )
            if self._in_trade:
                self._exit_all(state, submit_order, reason="SESSION_CUTOFF")
            return

        # Continuously monitor exits if in a trade
        if self._in_trade:
            self._compute_signals(state)
            self._check_exits(state, submit_order)
            self._update_display(state)
            return

        # Signal window complete?
        if self._bars_in_window >= self._bars_per_window:
            self._compute_signals(state)
            self._update_display(state)

            # Calibrate theta if needed
            if self._theta_auto and not self._calibrated:
                self._collect_calibration_data()
                # Reset window
                self._reset_window(state)
                return

            # Check filters and enter
            if self._check_filters():
                self._enter_basket(state, submit_order)

            # Reset window
            self._reset_window(state)

    def on_fill(self, symbol: str, fill: Fill) -> None:
        logger.debug(
            "  [%s] FILL %s: %s %d @ %.2f (slip: %.1f ticks, comm: $%.2f)",
            self.name,
            symbol,
            fill.side.value,
            fill.quantity,
            fill.fill_price,
            fill.slippage_ticks,
            fill.commission,
        )

    def on_end(self) -> None:
        logger.info(
            "[%s] Strategy complete. Total session PnL tracked: $%.2f",
            self.name,
            self._session_pnl,
        )

    # ------------------------------------------------------------------ #
    # Signal computation
    # ------------------------------------------------------------------ #

    def _compute_signals(self, state: MultiInstrumentState) -> None:
        """Compute R_i, R_m, d_i, sigma_i, w_i for all instruments."""
        n = len(self._symbols)

        # Step 1-2: Compute per-instrument returns
        for sym in self._symbols:
            data = self._data[sym]
            bar = state.instruments[sym].bar
            if data.window_open_price and data.window_open_price != 0:
                data.return_pct = (
                    (bar.close - data.window_open_price) / data.window_open_price
                )
            else:
                data.return_pct = 0.0

        # Step 3: Market return (equally weighted)
        r_m = sum(self._data[s].return_pct for s in self._symbols) / n

        # Step 4: Deviations
        for sym in self._symbols:
            self._data[sym].deviation = self._data[sym].return_pct - r_m

        # Step 5: ATR already computed incrementally

        # Step 6: Raw weights = -1 * (d_i / sigma_i)
        for sym in self._symbols:
            data = self._data[sym]
            sigma = data.atr_value
            if sigma and sigma > 0:
                # Normalize deviation by ATR expressed as % of price
                bar = state.instruments[sym].bar
                sigma_pct = sigma / bar.close if bar.close > 0 else 1.0
                data.raw_weight = -1.0 * (data.deviation / sigma_pct)
            else:
                data.raw_weight = 0.0

        # Step 7: Normalize so sum(|w_i|) == 1
        total_abs = sum(abs(self._data[s].raw_weight) for s in self._symbols)
        if total_abs > 0:
            for sym in self._symbols:
                self._data[sym].normalized_weight = (
                    self._data[sym].raw_weight / total_abs
                )
        else:
            for sym in self._symbols:
                self._data[sym].normalized_weight = 0.0

        # Step 8: Verify dollar neutrality
        weight_sum = sum(self._data[s].normalized_weight for s in self._symbols)
        if abs(weight_sum) > 1e-10:
            logger.error(
                "DOLLAR NEUTRALITY VIOLATED: sum(w_i) = %.10f", weight_sum
            )

        # Volume filter: v_i = ln(vol_current / vol_prior)
        for sym in self._symbols:
            data = self._data[sym]
            if data.prior_window_volume > 0 and data.window_volume > 0:
                data.volume_change = math.log(
                    data.window_volume / data.prior_window_volume
                )
            else:
                data.volume_change = 0.0

        # Store R_m for display
        self._display["R_m"] = r_m

    # ------------------------------------------------------------------ #
    # Filters
    # ------------------------------------------------------------------ #

    def _check_filters(self) -> bool:
        """Check deviation threshold filter AND volume filter."""
        if self._theta is None:
            return False

        # FILTER 1: Deviation threshold
        max_dev = max(abs(self._data[s].deviation) for s in self._symbols)
        if max_dev <= self._theta:
            return False

        # FILTER 2: Volume — instrument with largest |d_i| must have v_i > 0
        largest_dev_sym = max(
            self._symbols, key=lambda s: abs(self._data[s].deviation)
        )
        if self._data[largest_dev_sym].volume_change <= 0:
            logger.debug(
                "Volume filter blocked entry: %s v_i=%.4f",
                largest_dev_sym,
                self._data[largest_dev_sym].volume_change,
            )
            return False

        return True

    # ------------------------------------------------------------------ #
    # Entry
    # ------------------------------------------------------------------ #

    def _enter_basket(
        self, state: MultiInstrumentState, submit_order: MultiSubmitFn
    ) -> None:
        """Enter all instruments simultaneously based on normalized weights."""
        if self._in_trade:
            return

        for sym in self._symbols:
            data = self._data[sym]
            w = data.normalized_weight
            if abs(w) < 1e-10:
                continue

            bar = state.instruments[sym].bar
            spec = None
            # Compute position size from weight
            # weight magnitude * capital = target notional
            # contracts = notional / (price * point_value)
            from engine.contract_registry import ContractRegistry

            registry = ContractRegistry()
            spec = registry.get(sym)

            target_notional = abs(w) * state.equity
            # Cap at max_position_pct of capital
            max_notional = self._max_pos_pct * state.equity
            target_notional = min(target_notional, max_notional)

            price_notional = bar.close * spec.point_value
            if price_notional <= 0:
                continue

            contracts = max(1, int(target_notional / price_notional))

            side = OrderSide.BUY if w > 0 else OrderSide.SELL
            order = Order(
                side=side,
                quantity=contracts,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
            )

            result = submit_order(sym, order)
            if result:
                data.entry_deviation = data.deviation
                data.entry_price = bar.close
                logger.info(
                    "  [%s] ENTRY %s: %s %d contracts, w=%.4f, d_i=%.6f",
                    self.name,
                    sym,
                    side.value,
                    contracts,
                    w,
                    data.deviation,
                )

        self._in_trade = True

    # ------------------------------------------------------------------ #
    # Exit checks
    # ------------------------------------------------------------------ #

    def _check_exits(
        self, state: MultiInstrumentState, submit_order: MultiSubmitFn
    ) -> None:
        """Monitor all three exit conditions continuously."""
        if not self._in_trade:
            return

        # TARGET EXIT: all |d_i| < theta/2 for held positions
        if self._theta is not None:
            half_theta = self._theta / 2.0
            all_reverted = all(
                abs(self._data[s].deviation) < half_theta
                for s in self._symbols
                if state.instruments[s].position_quantity > 0
            )
            if all_reverted:
                self._exit_all(state, submit_order, reason="TARGET_REVERT")
                return

        # STOP LOSS: any position's deviation widens beyond stop_multiple * sigma
        for sym in self._symbols:
            inst = state.instruments[sym]
            if inst.position_quantity == 0:
                continue

            data = self._data[sym]
            sigma = data.atr_value
            if sigma is None or sigma <= 0:
                continue

            bar = inst.bar
            sigma_pct = sigma / bar.close if bar.close > 0 else 1.0

            # Deviation widening: entry_deviation had one sign, check if
            # current deviation has moved further against us
            dev_change = abs(data.deviation) - abs(data.entry_deviation)
            if dev_change > self._stop_multiple * sigma_pct:
                logger.warning(
                    "  [%s] STOP LOSS triggered on %s: "
                    "dev_change=%.6f > %.1f * sigma=%.6f",
                    self.name,
                    sym,
                    dev_change,
                    self._stop_multiple,
                    sigma_pct,
                )
                self._exit_all(state, submit_order, reason="STOP_LOSS")
                return

    def _exit_all(
        self,
        state: MultiInstrumentState,
        submit_order: MultiSubmitFn,
        reason: str,
    ) -> None:
        """Close ALL positions in ALL instruments simultaneously."""
        if not self._in_trade:
            return

        for sym in self._symbols:
            inst = state.instruments[sym]
            if inst.position_quantity == 0:
                continue

            side = (
                OrderSide.SELL if inst.position_direction == 1 else OrderSide.BUY
            )
            order = Order(
                side=side,
                quantity=inst.position_quantity,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
            )
            result = submit_order(sym, order)
            if result:
                logger.info(
                    "  [%s] EXIT %s: %s %d contracts [%s]",
                    self.name,
                    sym,
                    side.value,
                    inst.position_quantity,
                    reason,
                )

        self._in_trade = False

    # ------------------------------------------------------------------ #
    # Session management
    # ------------------------------------------------------------------ #

    def _on_session_open(self, state: MultiInstrumentState) -> None:
        """Reset all session-anchored state."""
        for sym in self._symbols:
            self._data[sym].reset_session()
            if sym in state.instruments:
                bar = state.instruments[sym].bar
                self._data[sym].window_open_price = bar.open

        self._bars_in_window = 0
        self._session_active = True
        self._past_cutoff = False
        self._in_trade = False
        self._session_pnl = 0.0

        # After first session of calibration, compute theta
        if self._theta_auto and not self._calibrated and self._calibration_deviations:
            self._finalize_calibration()

        logger.info("[%s] Session open — state reset", self.name)

    def _reset_window(self, state: MultiInstrumentState) -> None:
        """Reset the signal window for the next computation period."""
        for sym in self._symbols:
            bar = state.instruments[sym].bar
            self._data[sym].reset_window(bar.close)
        self._bars_in_window = 0

    # ------------------------------------------------------------------ #
    # Theta calibration
    # ------------------------------------------------------------------ #

    def _collect_calibration_data(self) -> None:
        """Collect deviation samples for auto-calibrating theta."""
        for sym in self._symbols:
            self._calibration_deviations.append(abs(self._data[sym].deviation))

    def _finalize_calibration(self) -> None:
        """Set theta to 1 standard deviation of observed |d_i|."""
        if not self._calibration_deviations:
            self._theta = 0.001  # Fallback
            self._calibrated = True
            return

        n = len(self._calibration_deviations)
        mean = sum(self._calibration_deviations) / n
        variance = sum(
            (x - mean) ** 2 for x in self._calibration_deviations
        ) / max(1, n - 1)
        std = math.sqrt(variance)

        self._theta = std
        self._calibrated = True
        logger.info(
            "[%s] Theta calibrated: %.6f (from %d samples, mean=%.6f, std=%.6f)",
            self.name,
            self._theta,
            n,
            mean,
            std,
        )

    # ------------------------------------------------------------------ #
    # Display / monitoring
    # ------------------------------------------------------------------ #

    def _update_display(self, state: MultiInstrumentState) -> None:
        """Update monitoring display data."""
        self._display["in_trade"] = self._in_trade
        self._display["theta"] = self._theta
        self._display["calibrated"] = self._calibrated
        self._display["session_pnl"] = state.total_realized_pnl

        for sym in self._symbols:
            data = self._data[sym]
            inst = state.instruments.get(sym)
            self._display[f"{sym}_d_i"] = data.deviation
            self._display[f"{sym}_R_i"] = data.return_pct
            self._display[f"{sym}_w_i"] = data.normalized_weight
            self._display[f"{sym}_sigma"] = data.atr_value
            self._display[f"{sym}_v_i"] = data.volume_change

            if inst and inst.position_quantity > 0:
                self._display[f"{sym}_entry_d_i"] = data.entry_deviation
                self._display[f"{sym}_pos"] = (
                    inst.position_direction * inst.position_quantity
                )

                # Stop loss proximity warning
                if data.atr_value and data.atr_value > 0:
                    bar = inst.bar
                    sigma_pct = data.atr_value / bar.close if bar.close > 0 else 1
                    dev_change = abs(data.deviation) - abs(data.entry_deviation)
                    stop_dist = self._stop_multiple * sigma_pct
                    if dev_change > stop_dist * 0.75:
                        self._display[f"{sym}_STOP_WARNING"] = (
                            f"APPROACHING STOP: {dev_change:.6f} / {stop_dist:.6f}"
                        )
                    elif f"{sym}_STOP_WARNING" in self._display:
                        del self._display[f"{sym}_STOP_WARNING"]

        # Session cutoff warning
        if state.session.minutes_to_session_close <= self._cutoff_minutes * 2:
            self._display["CUTOFF_WARNING"] = (
                f"{state.session.minutes_to_session_close:.0f} min to close"
            )

    def get_display(self) -> dict:
        """Get current monitoring display data (for external consumers)."""
        return dict(self._display)

    def print_status(self) -> None:
        """Print current status to stdout."""
        d = self._display
        print(f"\n--- {self.name} Status ---")
        print(f"  R_m = {d.get('R_m', 0):.6f}")
        print(f"  Theta = {d.get('theta', 'N/A')}")
        print(f"  In Trade = {d.get('in_trade', False)}")
        print(f"  Session PnL = ${d.get('session_pnl', 0):,.2f}")
        print()
        for sym in self._symbols:
            line = f"  {sym}: d_i={d.get(f'{sym}_d_i', 0):.6f}"
            line += f"  R_i={d.get(f'{sym}_R_i', 0):.6f}"
            line += f"  w_i={d.get(f'{sym}_w_i', 0):.4f}"
            sigma = d.get(f'{sym}_sigma')
            line += f"  σ={sigma:.4f}" if sigma else "  σ=N/A"
            pos = d.get(f"{sym}_pos")
            if pos:
                line += f"  POS={pos:+d}"
                line += f"  entry_d={d.get(f'{sym}_entry_d_i', 0):.6f}"
            warn = d.get(f"{sym}_STOP_WARNING")
            if warn:
                line += f"  ⚠ {warn}"
            print(line)

        cutoff = d.get("CUTOFF_WARNING")
        if cutoff:
            print(f"  ⚠ CUTOFF: {cutoff}")
        print()
