"""Intraday contrarian mean-reversion basket strategy (v2).

Computes cross-sectional deviations from the equally-weighted market return
using a rolling z-score (not fixed windows), volatility-normalizes, and
fades outliers. Includes momentum regime filter and confirmation bars.

Always dollar-neutral. All instruments entered and exited as a group.
No overnight positions.
"""

from __future__ import annotations

import logging
import math
from collections import deque
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

    # Rolling price/return history
    close_history: deque = field(default_factory=lambda: deque(maxlen=200))
    return_history: deque = field(default_factory=lambda: deque(maxlen=200))
    deviation_history: deque = field(default_factory=lambda: deque(maxlen=200))
    volume_history: deque = field(default_factory=lambda: deque(maxlen=200))

    # Current bar values
    current_close: float = 0.0
    current_return: float = 0.0  # 1-bar return
    prev_close: float = 0.0

    # Rolling statistics
    deviation: float = 0.0        # d_i = R_i - R_m (current bar)
    z_score: float = 0.0          # z_i = d_i / std(d_i) over lookback
    rolling_vol: float = 0.0      # std of returns over lookback
    raw_weight: float = 0.0
    normalized_weight: float = 0.0

    # Confirmation tracking
    signal_direction: int = 0     # -1 or +1 when confirming, 0 otherwise
    confirm_count: int = 0        # consecutive bars signal has persisted

    # Entry state
    entry_deviation: float = 0.0
    entry_z_score: float = 0.0
    entry_price: float = 0.0

    # ATR for stop loss (Wilder's smoothing)
    atr_value: Optional[float] = None
    atr_prev_close: Optional[float] = None

    def reset_session(self) -> None:
        """Reset session-anchored data at session open."""
        self.close_history.clear()
        self.return_history.clear()
        self.deviation_history.clear()
        self.volume_history.clear()
        self.current_close = 0.0
        self.current_return = 0.0
        self.prev_close = 0.0
        self.deviation = 0.0
        self.z_score = 0.0
        self.rolling_vol = 0.0
        self.raw_weight = 0.0
        self.normalized_weight = 0.0
        self.signal_direction = 0
        self.confirm_count = 0
        self.atr_value = None
        self.atr_prev_close = None

    def update_atr(self, high: float, low: float, close: float) -> None:
        """Incrementally update ATR using Wilder's smoothing (14-period)."""
        if self.atr_prev_close is None:
            self.atr_value = high - low if high > low else 0.001
            self.atr_prev_close = close
            return

        tr = max(
            high - low,
            abs(high - self.atr_prev_close),
            abs(low - self.atr_prev_close),
        )
        self.atr_prev_close = close

        if self.atr_value is None:
            self.atr_value = tr
        else:
            self.atr_value = (self.atr_value * 13 + tr) / 14


class ContraMeanReversionStrategy(MultiInstrumentStrategy):
    """Intraday contrarian mean-reversion across US equity index futures (v2).

    Key improvements over v1:
    - Rolling z-score signal (continuous, not fixed-window reset)
    - Momentum regime filter (skip trending markets)
    - Confirmation bars (deviation must persist before entry)
    - Rolling return volatility for cross-sectional normalization
    """

    def __init__(
        self,
        name: str = "Contra_MeanRev",
        signal_window_minutes: int = 60,
        theta: Optional[float] = None,
        theta_multiplier: float = 1.5,
        stop_multiple: float = 3.0,
        target_fraction: float = 0.4,
        max_position_pct: float = 0.10,
        session_cutoff_minutes: int = 15,
        min_hold_bars: int = 5,
        skip_first_minutes: int = 30,
        lookback_bars: int = 60,
        z_entry_threshold: float = 2.0,
        z_exit_threshold: float = 0.5,
        confirm_bars: int = 3,
        momentum_filter_window: int = 20,
        momentum_threshold: float = 0.7,
    ) -> None:
        super().__init__(name)
        self._window_minutes = signal_window_minutes
        self._theta = theta
        self._theta_auto = theta is None
        self._theta_multiplier = theta_multiplier
        self._stop_multiple = stop_multiple
        self._target_fraction = target_fraction
        self._max_pos_pct = max_position_pct
        self._cutoff_minutes = session_cutoff_minutes
        self._min_hold_bars = min_hold_bars
        self._skip_first_minutes = skip_first_minutes
        self._lookback_bars = lookback_bars
        self._z_entry = z_entry_threshold
        self._z_exit = z_exit_threshold
        self._confirm_bars = confirm_bars
        self._momentum_window = momentum_filter_window
        self._momentum_threshold = momentum_threshold

        # Runtime state
        self._symbols: list[str] = []
        self._data: dict[str, _InstrumentData] = {}
        self._in_trade: bool = False
        self._trade_entry_bar: int = 0
        self._session_active: bool = False
        self._past_cutoff: bool = False
        self._session_bars: int = 0
        self._skip_first_bars: int = 0

        # Rolling market return history for momentum filter
        self._market_return_history: deque = deque(maxlen=200)

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
            "ContraMeanRev v2: instruments=%s, lookback=%d bars, "
            "z_entry=%.1f, z_exit=%.1f, confirm=%d bars, "
            "stop=%.1fx, theta_mult=%.1f, momentum_thresh=%.1f, "
            "skip_first=%dm, min_hold=%d bars",
            self._symbols, self._lookback_bars,
            self._z_entry, self._z_exit, self._confirm_bars,
            self._stop_multiple, self._theta_multiplier,
            self._momentum_threshold,
            self._skip_first_minutes, self._min_hold_bars,
        )

    def on_bar(
        self, state: MultiInstrumentState, submit_order: MultiSubmitFn
    ) -> None:
        if not state.has_all_instruments(self._symbols):
            return

        session = state.session

        if session.session_open_bar:
            self._on_session_open(state)

        if not session.is_rth:
            return

        self._session_bars += 1

        # Compute skip threshold on first bar
        if self._skip_first_bars == 0 and self._skip_first_minutes > 0:
            any_bar = state.instruments[self._symbols[0]].bar
            bar_minutes = max(1, any_bar.duration_ns // 60_000_000_000)
            self._skip_first_bars = max(1, self._skip_first_minutes // bar_minutes)

        # Skip opening noise
        if self._skip_first_minutes > 0 and self._session_bars <= self._skip_first_bars:
            # Still update prices so we have history when we start trading
            for sym in self._symbols:
                bar = state.instruments[sym].bar
                data = self._data[sym]
                data.current_close = bar.close
                data.update_atr(bar.high, bar.low, bar.close)
                if data.prev_close > 0:
                    data.current_return = (bar.close - data.prev_close) / data.prev_close
                data.prev_close = bar.close
            return

        # --- Core signal computation (every bar) ---
        self._update_returns(state)
        self._compute_deviations(state)
        self._compute_z_scores()
        self._compute_weights()

        # Session cutoff
        if session.minutes_to_session_close <= self._cutoff_minutes:
            if not self._past_cutoff:
                self._past_cutoff = True
                logger.info("[%s] SESSION CUTOFF — flattening all", self.name)
            if self._in_trade:
                self._exit_all(state, submit_order, reason="SESSION_CUTOFF")
            return

        # If in trade, check exits continuously
        if self._in_trade:
            self._check_exits(state, submit_order)
            return

        # Not in trade — check for entry signal
        if self._session_bars >= self._lookback_bars:
            if self._check_entry_signal(state):
                self._enter_basket(state, submit_order)

    def on_fill(self, symbol: str, fill: Fill) -> None:
        logger.debug(
            "  [%s] FILL %s: %s %d @ %.2f",
            self.name, symbol, fill.side.value,
            fill.quantity, fill.fill_price,
        )

    def on_end(self) -> None:
        logger.info("[%s] Strategy complete.", self.name)

    # ------------------------------------------------------------------ #
    # Signal computation (runs every bar)
    # ------------------------------------------------------------------ #

    def _update_returns(self, state: MultiInstrumentState) -> None:
        """Update 1-bar returns and price history for all instruments."""
        for sym in self._symbols:
            bar = state.instruments[sym].bar
            data = self._data[sym]

            data.current_close = bar.close
            data.update_atr(bar.high, bar.low, bar.close)
            data.volume_history.append(bar.volume)

            if data.prev_close > 0:
                data.current_return = (bar.close - data.prev_close) / data.prev_close
            else:
                data.current_return = 0.0

            data.close_history.append(bar.close)
            data.return_history.append(data.current_return)
            data.prev_close = bar.close

    def _compute_deviations(self, state: MultiInstrumentState) -> None:
        """Compute cross-sectional deviation d_i = R_i - R_m for this bar."""
        n = len(self._symbols)
        r_m = sum(self._data[s].current_return for s in self._symbols) / n

        self._market_return_history.append(r_m)

        for sym in self._symbols:
            data = self._data[sym]
            data.deviation = data.current_return - r_m
            data.deviation_history.append(data.deviation)

        self._display["R_m"] = r_m

    def _compute_z_scores(self) -> None:
        """Compute rolling z-score of deviations for each instrument."""
        lb = self._lookback_bars

        for sym in self._symbols:
            data = self._data[sym]
            devs = data.deviation_history

            if len(devs) < lb:
                data.z_score = 0.0
                data.rolling_vol = 0.0
                continue

            # Rolling mean and std of deviations over lookback
            recent = list(devs)[-lb:]
            mean_d = sum(recent) / lb
            var_d = sum((x - mean_d) ** 2 for x in recent) / lb
            std_d = math.sqrt(var_d) if var_d > 0 else 1e-10

            data.z_score = (data.deviation - mean_d) / std_d

            # Also compute rolling return vol for weighting
            rets = list(data.return_history)[-lb:]
            mean_r = sum(rets) / lb
            var_r = sum((x - mean_r) ** 2 for x in rets) / lb
            data.rolling_vol = math.sqrt(var_r) if var_r > 0 else 1e-10

    def _compute_weights(self) -> None:
        """Compute dollar-neutral weights from z-scores."""
        # Raw weight: fade the z-score (buy underperformers, sell overperformers)
        # Normalize by rolling vol to equalize risk contribution
        for sym in self._symbols:
            data = self._data[sym]
            if data.rolling_vol > 0:
                data.raw_weight = -data.z_score / data.rolling_vol
            else:
                data.raw_weight = 0.0

        # Demean to force dollar neutrality
        n = len(self._symbols)
        mean_w = sum(self._data[s].raw_weight for s in self._symbols) / n
        for sym in self._symbols:
            self._data[sym].raw_weight -= mean_w

        # Normalize so sum(|w_i|) == 1
        total_abs = sum(abs(self._data[s].raw_weight) for s in self._symbols)
        if total_abs > 0:
            for sym in self._symbols:
                self._data[sym].normalized_weight = self._data[sym].raw_weight / total_abs
        else:
            for sym in self._symbols:
                self._data[sym].normalized_weight = 0.0

    # ------------------------------------------------------------------ #
    # Entry logic
    # ------------------------------------------------------------------ #

    def _check_entry_signal(self, state: MultiInstrumentState) -> bool:
        """Check if entry conditions are met: z-score threshold + confirmation + regime."""

        # 1. Z-score threshold: at least one instrument must exceed z_entry
        max_abs_z = max(abs(self._data[s].z_score) for s in self._symbols)
        if max_abs_z < self._z_entry:
            # Reset confirmation counter
            for sym in self._symbols:
                self._data[sym].confirm_count = 0
                self._data[sym].signal_direction = 0
            return False

        # 2. Momentum regime filter: skip if market is trending
        if self._is_trending():
            for sym in self._symbols:
                self._data[sym].confirm_count = 0
                self._data[sym].signal_direction = 0
            return False

        # 3. Confirmation: the z-score signal must persist for N consecutive bars
        #    Direction = sign of the strongest z-score instrument's weight
        strongest_sym = max(self._symbols, key=lambda s: abs(self._data[s].z_score))
        current_dir = -1 if self._data[strongest_sym].z_score > 0 else 1  # fade it

        all_confirmed = True
        for sym in self._symbols:
            data = self._data[sym]
            # Check if this instrument's signal direction is consistent
            sym_dir = -1 if data.z_score > 0 else 1 if data.z_score < 0 else 0

            if sym_dir == data.signal_direction and sym_dir != 0:
                data.confirm_count += 1
            else:
                data.signal_direction = sym_dir
                data.confirm_count = 1

        # The strongest instrument must have confirmed
        if self._data[strongest_sym].confirm_count < self._confirm_bars:
            return False

        # 4. Volume confirmation on the strongest deviator
        data = self._data[strongest_sym]
        if len(data.volume_history) >= 2:
            recent_vol = list(data.volume_history)[-1]
            prev_vol = list(data.volume_history)[-2]
            if prev_vol > 0 and recent_vol > 0:
                if math.log(recent_vol / prev_vol) <= 0:
                    return False

        return True

    def _is_trending(self) -> bool:
        """Check if the market is in a momentum regime (all instruments trending same way)."""
        if len(self._market_return_history) < self._momentum_window:
            return False

        recent_market = list(self._market_return_history)[-self._momentum_window:]
        positive = sum(1 for r in recent_market if r > 0)
        fraction = positive / self._momentum_window

        # If > threshold of bars are positive or negative, market is trending
        if fraction > self._momentum_threshold or fraction < (1 - self._momentum_threshold):
            return True

        # Also check if all instruments are moving same direction consistently
        n = len(self._symbols)
        same_dir = 0
        for sym in self._symbols:
            rets = list(self._data[sym].return_history)
            if len(rets) >= self._momentum_window:
                recent = rets[-self._momentum_window:]
                pos_frac = sum(1 for r in recent if r > 0) / self._momentum_window
                if pos_frac > self._momentum_threshold or pos_frac < (1 - self._momentum_threshold):
                    same_dir += 1

        # If most instruments are trending, skip
        if same_dir >= n * 0.75:
            return True

        return False

    def _enter_basket(
        self, state: MultiInstrumentState, submit_order: MultiSubmitFn
    ) -> None:
        """Enter all instruments simultaneously based on normalized weights."""
        if self._in_trade:
            return

        from engine.contract_registry import ContractRegistry
        registry = ContractRegistry()

        entered = False
        for sym in self._symbols:
            data = self._data[sym]
            w = data.normalized_weight
            if abs(w) < 1e-10:
                continue

            bar = state.instruments[sym].bar
            spec = registry.get(sym)

            target_notional = abs(w) * state.equity
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
                entered = True
                data.entry_deviation = data.deviation
                data.entry_z_score = data.z_score
                data.entry_price = bar.close
                logger.info(
                    "  [%s] ENTRY %s: %s %d contracts, w=%.4f, z=%.2f, d_i=%.6f",
                    self.name, sym, side.value, contracts, w, data.z_score, data.deviation,
                )

        if entered:
            self._in_trade = True
            self._trade_entry_bar = state.bar_index

    # ------------------------------------------------------------------ #
    # Exit logic
    # ------------------------------------------------------------------ #

    def _check_exits(
        self, state: MultiInstrumentState, submit_order: MultiSubmitFn
    ) -> None:
        """Monitor exit conditions continuously."""
        if not self._in_trade:
            return

        # Minimum hold period
        bars_held = state.bar_index - self._trade_entry_bar
        if self._min_hold_bars > 0 and bars_held < self._min_hold_bars:
            return

        # TARGET EXIT: z-scores have reverted (all below z_exit threshold)
        all_reverted = all(
            abs(self._data[s].z_score) < self._z_exit
            for s in self._symbols
            if state.instruments[s].position_quantity > 0
        )
        if all_reverted:
            self._exit_all(state, submit_order, reason="TARGET_REVERT")
            return

        # STOP LOSS: z-score has widened beyond stop_multiple of entry z-score
        for sym in self._symbols:
            inst = state.instruments[sym]
            if inst.position_quantity == 0:
                continue

            data = self._data[sym]

            # Z-score stop: if current z is stop_multiple * entry z in the wrong direction
            if abs(data.entry_z_score) > 0:
                z_ratio = abs(data.z_score) / abs(data.entry_z_score)
                if z_ratio > self._stop_multiple and abs(data.z_score) > self._z_entry:
                    logger.warning(
                        "  [%s] Z-SCORE STOP on %s: z=%.2f (entry=%.2f, ratio=%.1fx)",
                        self.name, sym, data.z_score, data.entry_z_score, z_ratio,
                    )
                    self._exit_all(state, submit_order, reason="STOP_LOSS")
                    return

            # ATR-based hard stop as backup
            if data.atr_value and data.atr_value > 0:
                bar = inst.bar
                price_move = abs(bar.close - data.entry_price)
                if price_move > self._stop_multiple * data.atr_value:
                    logger.warning(
                        "  [%s] ATR STOP on %s: move=%.2f > %.1f * ATR=%.2f",
                        self.name, sym, price_move, self._stop_multiple, data.atr_value,
                    )
                    self._exit_all(state, submit_order, reason="ATR_STOP")
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

            side = OrderSide.SELL if inst.position_direction == 1 else OrderSide.BUY
            order = Order(
                side=side,
                quantity=inst.position_quantity,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
            )
            result = submit_order(sym, order)
            if result:
                logger.info(
                    "  [%s] EXIT %s: %s %d contracts [%s] z=%.2f",
                    self.name, sym, side.value, inst.position_quantity,
                    reason, self._data[sym].z_score,
                )

        self._in_trade = False
        # Reset confirmation state
        for sym in self._symbols:
            self._data[sym].confirm_count = 0
            self._data[sym].signal_direction = 0

    # ------------------------------------------------------------------ #
    # Session management
    # ------------------------------------------------------------------ #

    def _on_session_open(self, state: MultiInstrumentState) -> None:
        """Reset all session-anchored state."""
        for sym in self._symbols:
            self._data[sym].reset_session()

        self._session_active = True
        self._past_cutoff = False
        self._in_trade = False
        self._session_bars = 0
        self._skip_first_bars = 0
        self._market_return_history.clear()

        # Calibrate theta from prior sessions if needed
        if self._theta_auto and not self._calibrated and self._calibration_deviations:
            self._finalize_calibration()

        logger.info("[%s] Session open — state reset", self.name)

    # ------------------------------------------------------------------ #
    # Theta calibration (kept for backward compat, mostly unused in v2)
    # ------------------------------------------------------------------ #

    def _finalize_calibration(self) -> None:
        if not self._calibration_deviations:
            self._theta = 0.001
            self._calibrated = True
            return

        n = len(self._calibration_deviations)
        mean = sum(self._calibration_deviations) / n
        variance = sum((x - mean) ** 2 for x in self._calibration_deviations) / max(1, n - 1)
        std = math.sqrt(variance)
        self._theta = std * self._theta_multiplier
        self._calibrated = True
        logger.info(
            "[%s] Theta calibrated: %.6f (std=%.6f x %.1f, %d samples)",
            self.name, self._theta, std, self._theta_multiplier, n,
        )

    # ------------------------------------------------------------------ #
    # Display / monitoring
    # ------------------------------------------------------------------ #

    def get_display(self) -> dict:
        return dict(self._display)

    def print_status(self) -> None:
        d = self._display
        print(f"\n--- {self.name} Status ---")
        print(f"  R_m = {d.get('R_m', 0):.6f}")
        print(f"  In Trade = {self._in_trade}")
        print()
        for sym in self._symbols:
            data = self._data[sym]
            line = f"  {sym}: z={data.z_score:+.2f}  d_i={data.deviation:.6f}"
            line += f"  w_i={data.normalized_weight:.4f}"
            line += f"  vol={data.rolling_vol:.6f}"
            line += f"  confirm={data.confirm_count}/{self._confirm_bars}"
            print(line)
        print()
