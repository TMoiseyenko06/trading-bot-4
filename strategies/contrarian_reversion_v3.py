"""Intraday contrarian mean-reversion basket strategy (v3).

Improvements over v2 based on grid search analysis:
1. Spread-based signal: require cross-sectional z-score spread threshold
2. Adaptive exit tightening: progressively tighten exit threshold as trade ages
3. Relative volume filter: rolling average volume spike detection
4. Per-leg profit taking: exit individual legs when they revert
5. Intraday time weighting: scale entry threshold by time-of-day
6. Z-score widening: require divergence to still be accelerating at entry
7. Per-leg ATR stop: tighter ATR-based stop on individual legs
8. Asymmetric exit: let winners ride to full reversion, cut losers fast
9. Volatility regime filter: skip entries when intraday vol is elevated
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
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
    current_return: float = 0.0
    prev_close: float = 0.0

    # Rolling statistics
    deviation: float = 0.0
    z_score: float = 0.0
    prev_z_score: float = 0.0  # for z-widening check
    rolling_vol: float = 0.0
    raw_weight: float = 0.0
    normalized_weight: float = 0.0

    # Confirmation tracking
    signal_direction: int = 0
    confirm_count: int = 0

    # Entry state
    entry_deviation: float = 0.0
    entry_z_score: float = 0.0
    entry_price: float = 0.0
    entry_weight: float = 0.0

    # ATR for stop loss
    atr_value: Optional[float] = None
    atr_prev_close: Optional[float] = None

    # Per-leg tracking
    leg_exited: bool = False

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
        self.prev_z_score = 0.0
        self.rolling_vol = 0.0
        self.raw_weight = 0.0
        self.normalized_weight = 0.0
        self.signal_direction = 0
        self.confirm_count = 0
        self.atr_value = None
        self.atr_prev_close = None
        self.leg_exited = False
        self.entry_weight = 0.0

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


class ContraMeanReversionV3(MultiInstrumentStrategy):
    """Intraday contrarian mean-reversion across US equity index futures (v3)."""

    def __init__(
        self,
        name: str = "Contra_MeanRev_v3",
        signal_window_minutes: int = 60,
        stop_multiple: float = 4.0,
        max_position_pct: float = 0.10,
        session_cutoff_minutes: int = 15,
        min_hold_bars: int = 3,
        skip_first_minutes: int = 0,
        lookback_bars: int = 60,
        z_entry_threshold: float = 2.0,
        z_exit_threshold: float = 0.5,
        confirm_bars: int = 2,
        momentum_filter_window: int = 20,
        momentum_threshold: float = 0.75,
        # v3 parameters
        z_spread_threshold: float = 2.0,
        exit_tighten_bars: int = 20,
        exit_tighten_rate: float = 0.02,
        volume_spike_multiple: float = 1.3,
        volume_avg_window: int = 20,
        per_leg_exit: bool = True,
        time_weight_enabled: bool = False,
        time_weight_peak_hour: float = 2.0,
        # v3.1 — win rate improvements
        require_z_widening: bool = False,
        leg_stop_atr_multiple: float = 3.0,
        asymmetric_exit: bool = False,
        vol_regime_filter: bool = False,
        vol_regime_multiple: float = 1.5,
        signal_only_symbols: list[str] | None = None,  # instruments used for signal but not traded
    ) -> None:
        super().__init__(name)
        self._window_minutes = signal_window_minutes
        self._stop_multiple = stop_multiple
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

        # v3 parameters
        self._z_spread_threshold = z_spread_threshold
        self._exit_tighten_bars = exit_tighten_bars
        self._exit_tighten_rate = exit_tighten_rate
        self._volume_spike_mult = volume_spike_multiple
        self._volume_avg_window = volume_avg_window
        self._per_leg_exit = per_leg_exit
        self._time_weight_enabled = time_weight_enabled
        self._time_weight_peak_hour = time_weight_peak_hour

        # v3.1 win rate parameters
        self._require_z_widening = require_z_widening
        self._leg_stop_atr = leg_stop_atr_multiple
        self._asymmetric_exit = asymmetric_exit
        self._vol_regime_filter = vol_regime_filter
        self._vol_regime_multiple = vol_regime_multiple
        self._signal_only: set[str] = set(s.upper() for s in (signal_only_symbols or []))

        # Runtime state
        self._symbols: list[str] = []
        self._data: dict[str, _InstrumentData] = {}
        self._in_trade: bool = False
        self._trade_entry_bar: int = 0
        self._session_active: bool = False
        self._past_cutoff: bool = False
        self._session_bars: int = 0
        self._skip_first_bars: int = 0
        self._active_legs: int = 0

        # Rolling market return history for momentum filter
        self._market_return_history: deque = deque(maxlen=200)

        # Rolling market volatility history for vol regime filter
        self._market_vol_history: deque = deque(maxlen=200)

        # Display state
        self._display: dict = {}

    def on_init(self, symbols: list[str]) -> None:
        self._symbols = sorted(symbols)
        for sym in self._symbols:
            self._data[sym] = _InstrumentData()

        logger.info(
            "ContraMeanRev v3: instruments=%s, lookback=%d, "
            "z_entry=%.1f, z_exit=%.1f, z_spread=%.1f, confirm=%d, "
            "stop=%.1fx, momentum=%.2f, skip=%dm, hold=%d, "
            "vol_spike=%.1f, per_leg=%s, time_wt=%s, "
            "z_widen=%s, leg_atr=%.1f, asym=%s, vol_regime=%s/%.1f",
            self._symbols, self._lookback_bars,
            self._z_entry, self._z_exit, self._z_spread_threshold,
            self._confirm_bars,
            self._stop_multiple, self._momentum_threshold,
            self._skip_first_minutes, self._min_hold_bars,
            self._volume_spike_mult, self._per_leg_exit,
            self._time_weight_enabled,
            self._require_z_widening, self._leg_stop_atr,
            self._asymmetric_exit, self._vol_regime_filter,
            self._vol_regime_multiple,
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

        # Skip opening noise — but still build history
        if self._skip_first_minutes > 0 and self._session_bars <= self._skip_first_bars:
            for sym in self._symbols:
                bar = state.instruments[sym].bar
                data = self._data[sym]
                data.current_close = bar.close
                data.update_atr(bar.high, bar.low, bar.close)
                data.volume_history.append(bar.volume)
                if data.prev_close > 0:
                    data.current_return = (bar.close - data.prev_close) / data.prev_close
                data.prev_close = bar.close
            return

        # --- Core signal computation (every bar) ---
        self._update_returns(state)
        self._compute_deviations(state)
        self._compute_z_scores()
        self._compute_weights()
        self._update_market_vol()

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
    # Signal computation
    # ------------------------------------------------------------------ #

    def _update_returns(self, state: MultiInstrumentState) -> None:
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
        n = len(self._symbols)
        r_m = sum(self._data[s].current_return for s in self._symbols) / n

        self._market_return_history.append(r_m)

        for sym in self._symbols:
            data = self._data[sym]
            data.deviation = data.current_return - r_m
            data.deviation_history.append(data.deviation)

        self._display["R_m"] = r_m

    def _compute_z_scores(self) -> None:
        lb = self._lookback_bars

        for sym in self._symbols:
            data = self._data[sym]
            devs = data.deviation_history

            # Save previous z-score before computing new one
            data.prev_z_score = data.z_score

            if len(devs) < lb:
                data.z_score = 0.0
                data.rolling_vol = 0.0
                continue

            recent = list(devs)[-lb:]
            mean_d = sum(recent) / lb
            var_d = sum((x - mean_d) ** 2 for x in recent) / lb
            std_d = math.sqrt(var_d) if var_d > 0 else 1e-10

            data.z_score = (data.deviation - mean_d) / std_d

            rets = list(data.return_history)[-lb:]
            mean_r = sum(rets) / lb
            var_r = sum((x - mean_r) ** 2 for x in rets) / lb
            data.rolling_vol = math.sqrt(var_r) if var_r > 0 else 1e-10

    def _compute_weights(self) -> None:
        for sym in self._symbols:
            data = self._data[sym]
            if data.rolling_vol > 0:
                data.raw_weight = -data.z_score / data.rolling_vol
            else:
                data.raw_weight = 0.0

        n = len(self._symbols)
        mean_w = sum(self._data[s].raw_weight for s in self._symbols) / n
        for sym in self._symbols:
            self._data[sym].raw_weight -= mean_w

        total_abs = sum(abs(self._data[s].raw_weight) for s in self._symbols)
        if total_abs > 0:
            for sym in self._symbols:
                self._data[sym].normalized_weight = self._data[sym].raw_weight / total_abs
        else:
            for sym in self._symbols:
                self._data[sym].normalized_weight = 0.0

    def _update_market_vol(self) -> None:
        """Compute rolling market return volatility for vol regime filter."""
        if len(self._market_return_history) < 20:
            return
        recent = list(self._market_return_history)[-20:]
        mean_r = sum(recent) / 20
        var_r = sum((x - mean_r) ** 2 for x in recent) / 20
        self._market_vol_history.append(math.sqrt(var_r) if var_r > 0 else 0.0)

    # ------------------------------------------------------------------ #
    # Entry logic
    # ------------------------------------------------------------------ #

    def _check_entry_signal(self, state: MultiInstrumentState) -> bool:
        z_scores = {s: self._data[s].z_score for s in self._symbols}

        # 1. Z-score threshold: at least one instrument must exceed z_entry
        max_abs_z = max(abs(z) for z in z_scores.values())
        if max_abs_z < self._z_entry:
            self._reset_confirmations()
            return False

        # 2. Spread-based signal
        z_max = max(z_scores.values())
        z_min = min(z_scores.values())
        z_spread = z_max - z_min
        if z_spread < self._z_spread_threshold:
            self._reset_confirmations()
            return False

        # 3. Momentum regime filter
        if self._is_trending():
            self._reset_confirmations()
            return False

        # 4. Volatility regime filter — skip when intraday vol is elevated
        if self._vol_regime_filter and self._is_high_vol_regime():
            self._reset_confirmations()
            return False

        # 5. Z-score widening — require divergence to be accelerating
        if self._require_z_widening:
            strongest_sym = max(self._symbols, key=lambda s: abs(self._data[s].z_score))
            d = self._data[strongest_sym]
            if abs(d.z_score) <= abs(d.prev_z_score):
                # Divergence is contracting, not widening — skip
                self._reset_confirmations()
                return False

        # 6. Confirmation bars
        strongest_sym = max(self._symbols, key=lambda s: abs(self._data[s].z_score))

        for sym in self._symbols:
            data = self._data[sym]
            sym_dir = -1 if data.z_score > 0 else 1 if data.z_score < 0 else 0

            if sym_dir == data.signal_direction and sym_dir != 0:
                data.confirm_count += 1
            else:
                data.signal_direction = sym_dir
                data.confirm_count = 1

        if self._data[strongest_sym].confirm_count < self._confirm_bars:
            return False

        # 7. Volume spike filter
        if not self._check_volume_spike(strongest_sym):
            return False

        # 8. Time-of-day weighting
        if self._time_weight_enabled:
            time_scale = self._get_time_weight()
            effective_z_entry = self._z_entry / time_scale
            if max_abs_z < effective_z_entry:
                return False

        return True

    def _check_volume_spike(self, sym: str) -> bool:
        data = self._data[sym]
        vols = data.volume_history

        if len(vols) < self._volume_avg_window + 1:
            return True

        recent_vols = list(vols)[-(self._volume_avg_window + 1):-1]
        avg_vol = sum(recent_vols) / len(recent_vols)

        if avg_vol <= 0:
            return True

        current_vol = list(vols)[-1]
        return current_vol >= avg_vol * self._volume_spike_mult

    def _get_time_weight(self) -> float:
        hours_in = self._session_bars / 60.0
        peak = self._time_weight_peak_hour
        sigma = 1.5
        weight = 0.8 + 0.4 * math.exp(-0.5 * ((hours_in - peak) / sigma) ** 2)
        return weight

    def _is_trending(self) -> bool:
        if len(self._market_return_history) < self._momentum_window:
            return False

        recent_market = list(self._market_return_history)[-self._momentum_window:]
        positive = sum(1 for r in recent_market if r > 0)
        fraction = positive / self._momentum_window

        if fraction > self._momentum_threshold or fraction < (1 - self._momentum_threshold):
            return True

        n = len(self._symbols)
        same_dir = 0
        for sym in self._symbols:
            rets = list(self._data[sym].return_history)
            if len(rets) >= self._momentum_window:
                recent = rets[-self._momentum_window:]
                pos_frac = sum(1 for r in recent if r > 0) / self._momentum_window
                if pos_frac > self._momentum_threshold or pos_frac < (1 - self._momentum_threshold):
                    same_dir += 1

        if same_dir >= n * 0.75:
            return True

        return False

    def _is_high_vol_regime(self) -> bool:
        """Check if current market vol is elevated vs recent history."""
        if len(self._market_vol_history) < 30:
            return False

        current_vol = self._market_vol_history[-1]
        if current_vol <= 0:
            return False

        # Compare current vol to median of recent vol history
        sorted_vols = sorted(self._market_vol_history)
        median_vol = sorted_vols[len(sorted_vols) // 2]

        if median_vol <= 0:
            return False

        return current_vol > self._vol_regime_multiple * median_vol

    def _reset_confirmations(self) -> None:
        for sym in self._symbols:
            self._data[sym].confirm_count = 0
            self._data[sym].signal_direction = 0

    def _enter_basket(
        self, state: MultiInstrumentState, submit_order: MultiSubmitFn
    ) -> None:
        if self._in_trade:
            return

        from engine.contract_registry import ContractRegistry
        registry = ContractRegistry()

        entered = 0
        for sym in self._symbols:
            if sym in self._signal_only:
                continue

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
                entered += 1
                data.entry_deviation = data.deviation
                data.entry_z_score = data.z_score
                data.entry_price = bar.close
                data.entry_weight = w
                data.leg_exited = False
                logger.info(
                    "  [%s] ENTRY %s: %s %d contracts, w=%.4f, z=%.2f, d_i=%.6f",
                    self.name, sym, side.value, contracts, w, data.z_score, data.deviation,
                )

        if entered > 0:
            self._in_trade = True
            self._trade_entry_bar = state.bar_index
            self._active_legs = entered

    # ------------------------------------------------------------------ #
    # Exit logic
    # ------------------------------------------------------------------ #

    def _check_exits(
        self, state: MultiInstrumentState, submit_order: MultiSubmitFn
    ) -> None:
        if not self._in_trade:
            return

        bars_held = state.bar_index - self._trade_entry_bar
        if self._min_hold_bars > 0 and bars_held < self._min_hold_bars:
            return

        # Adaptive exit threshold — tightens as trade ages
        effective_z_exit = self._z_exit
        if bars_held > self._exit_tighten_bars:
            extra_bars = bars_held - self._exit_tighten_bars
            effective_z_exit = self._z_exit + extra_bars * self._exit_tighten_rate
            effective_z_exit = min(effective_z_exit, self._z_entry * 0.8)

        # Per-leg exit mode
        if self._per_leg_exit:
            self._check_per_leg_exits(state, submit_order, effective_z_exit)
            return

        # All-or-nothing exit mode
        all_reverted = all(
            abs(self._data[s].z_score) < effective_z_exit
            for s in self._symbols
            if s not in self._signal_only and state.instruments[s].position_quantity > 0
        )
        if all_reverted:
            self._exit_all(state, submit_order, reason="TARGET_REVERT")
            return

        # Stop loss checks
        for sym in self._symbols:
            if sym in self._signal_only:
                continue

            inst = state.instruments[sym]
            if inst.position_quantity == 0:
                continue

            data = self._data[sym]

            if abs(data.entry_z_score) > 0:
                z_ratio = abs(data.z_score) / abs(data.entry_z_score)
                if z_ratio > self._stop_multiple and abs(data.z_score) > self._z_entry:
                    self._exit_all(state, submit_order, reason="STOP_LOSS")
                    return

            if data.atr_value and data.atr_value > 0:
                bar = inst.bar
                price_move = abs(bar.close - data.entry_price)
                if price_move > self._stop_multiple * data.atr_value:
                    self._exit_all(state, submit_order, reason="ATR_STOP")
                    return

    def _check_per_leg_exits(
        self,
        state: MultiInstrumentState,
        submit_order: MultiSubmitFn,
        effective_z_exit: float,
    ) -> None:
        """Per-leg exit with optional asymmetric handling."""
        for sym in self._symbols:
            if sym in self._signal_only:
                continue

            inst = state.instruments[sym]
            data = self._data[sym]

            if inst.position_quantity == 0 or data.leg_exited:
                continue

            bar = inst.bar

            # Determine if this leg is currently winning or losing
            if inst.position_direction == 1:  # long
                leg_pnl_sign = bar.close - data.entry_price
            else:  # short
                leg_pnl_sign = data.entry_price - bar.close

            if self._asymmetric_exit:
                # --- Asymmetric exit logic ---
                if leg_pnl_sign > 0:
                    # WINNING leg: let it ride to full reversion (z near 0)
                    # Only exit when z-score crosses through zero (full mean reversion)
                    if abs(data.z_score) < 0.1:
                        self._exit_leg(state, submit_order, sym, reason="WINNER_FULL_REVERT")
                        continue
                else:
                    # LOSING leg: cut fast with tighter ATR stop
                    if data.atr_value and data.atr_value > 0:
                        price_move = abs(bar.close - data.entry_price)
                        if price_move > self._leg_stop_atr * data.atr_value:
                            self._exit_leg(state, submit_order, sym, reason="LOSER_ATR_CUT")
                            continue
                    # Also check z-score stop for losers (tighter: use z_exit not stop_multiple)
                    if abs(data.entry_z_score) > 0:
                        z_ratio = abs(data.z_score) / abs(data.entry_z_score)
                        if z_ratio > self._stop_multiple and abs(data.z_score) > self._z_entry:
                            self._exit_leg(state, submit_order, sym, reason="LOSER_Z_STOP")
                            continue
            else:
                # --- Standard per-leg exit ---
                # Target: z-score reverted
                if abs(data.z_score) < effective_z_exit:
                    self._exit_leg(state, submit_order, sym, reason="LEG_REVERT")
                    continue

                # Per-leg ATR stop
                if data.atr_value and data.atr_value > 0:
                    price_move = abs(bar.close - data.entry_price)
                    if price_move > self._leg_stop_atr * data.atr_value:
                        self._exit_leg(state, submit_order, sym, reason="LEG_ATR_STOP")
                        continue

                # Z-score stop
                if abs(data.entry_z_score) > 0:
                    z_ratio = abs(data.z_score) / abs(data.entry_z_score)
                    if z_ratio > self._stop_multiple and abs(data.z_score) > self._z_entry:
                        self._exit_leg(state, submit_order, sym, reason="LEG_Z_STOP")
                        continue

        # Check if all legs are closed
        any_open = any(
            state.instruments[s].position_quantity > 0
            for s in self._symbols
            if s not in self._signal_only
        )
        if not any_open:
            self._in_trade = False
            self._reset_confirmations()

    def _exit_leg(
        self,
        state: MultiInstrumentState,
        submit_order: MultiSubmitFn,
        sym: str,
        reason: str,
    ) -> None:
        inst = state.instruments[sym]
        if inst.position_quantity == 0:
            return

        side = OrderSide.SELL if inst.position_direction == 1 else OrderSide.BUY
        order = Order(
            side=side,
            quantity=inst.position_quantity,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
        )
        result = submit_order(sym, order)
        if result:
            self._data[sym].leg_exited = True
            self._active_legs -= 1
            logger.info(
                "  [%s] EXIT LEG %s: %s %d contracts [%s] z=%.2f (remaining: %d)",
                self.name, sym, side.value, inst.position_quantity,
                reason, self._data[sym].z_score, self._active_legs,
            )

    def _exit_all(
        self,
        state: MultiInstrumentState,
        submit_order: MultiSubmitFn,
        reason: str,
    ) -> None:
        if not self._in_trade:
            return

        for sym in self._symbols:
            if sym in self._signal_only:
                continue

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
        self._active_legs = 0
        self._reset_confirmations()

    # ------------------------------------------------------------------ #
    # Session management
    # ------------------------------------------------------------------ #

    def _on_session_open(self, state: MultiInstrumentState) -> None:
        for sym in self._symbols:
            self._data[sym].reset_session()

        self._session_active = True
        self._past_cutoff = False
        self._in_trade = False
        self._session_bars = 0
        self._skip_first_bars = 0
        self._active_legs = 0
        self._market_return_history.clear()
        self._market_vol_history.clear()

        logger.info("[%s] Session open — state reset", self.name)

    # ------------------------------------------------------------------ #
    # Display
    # ------------------------------------------------------------------ #

    def get_display(self) -> dict:
        return dict(self._display)

    def print_status(self) -> None:
        d = self._display
        print(f"\n--- {self.name} Status ---")
        print(f"  R_m = {d.get('R_m', 0):.6f}")
        print(f"  In Trade = {self._in_trade} (legs: {self._active_legs})")
        print()
        for sym in self._symbols:
            data = self._data[sym]
            line = f"  {sym}: z={data.z_score:+.2f}  d_i={data.deviation:.6f}"
            line += f"  w_i={data.normalized_weight:.4f}"
            line += f"  vol={data.rolling_vol:.6f}"
            line += f"  confirm={data.confirm_count}/{self._confirm_bars}"
            if data.leg_exited:
                line += "  [EXITED]"
            print(line)
        print()
