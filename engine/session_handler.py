"""Session and calendar awareness for futures trading sessions."""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta, timezone
from typing import Optional

from engine.contract_registry import ContractSpec, SessionHours
from engine.data_feed import Bar

logger = logging.getLogger(__name__)

# Central Time offset from UTC (standard: -6, daylight: -5)
# We use a simplified approach: CT = UTC-6 for CST, UTC-5 for CDT
# DST transition: second Sunday of March to first Sunday of November
_CT_STANDARD_OFFSET = timedelta(hours=-6)
_CT_DAYLIGHT_OFFSET = timedelta(hours=-5)


def _utc_to_ct(dt: datetime) -> datetime:
    """Convert a UTC datetime to Central Time (handles DST)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    # Simple DST check for US Central Time
    year = dt.year
    # Second Sunday of March
    march_1 = datetime(year, 3, 1, tzinfo=timezone.utc)
    march_second_sun = march_1 + timedelta(days=(6 - march_1.weekday()) % 7 + 7)
    dst_start = march_second_sun.replace(hour=8)  # 2 AM CT = 8 AM UTC

    # First Sunday of November
    nov_1 = datetime(year, 11, 1, tzinfo=timezone.utc)
    nov_first_sun = nov_1 + timedelta(days=(6 - nov_1.weekday()) % 7)
    dst_end = nov_first_sun.replace(hour=7)  # 2 AM CDT = 7 AM UTC

    if dst_start <= dt < dst_end:
        return dt + _CT_DAYLIGHT_OFFSET
    return dt + _CT_STANDARD_OFFSET


class SessionContext:
    """Session context exposed to strategies via MarketState."""

    def __init__(self) -> None:
        self.is_rth: bool = False
        self.is_eth: bool = False
        self.session_open_bar: bool = False
        self.session_close_bar: bool = False
        self.is_daily_settlement: bool = False
        self.minutes_to_session_close: float = 0.0
        self.overnight_gap: Optional[float] = None
        self.contract_roll_pending: bool = False


class SessionHandler:
    """Determines session context for each bar based on contract spec."""

    def __init__(self, contract_spec: ContractSpec) -> None:
        self._spec = contract_spec
        self._hours = contract_spec.session_hours
        self._last_session_close_price: Optional[float] = None
        self._last_settlement_price: Optional[float] = None
        self._prev_is_rth: bool = False
        self._prev_symbol: Optional[str] = None
        self._bars_since_roll_warning: int = 0
        self._rth_bar_count: int = 0

    def compute_context(self, bar: Bar, bar_index: int) -> SessionContext:
        """Compute session context for the given bar."""
        ctx = SessionContext()
        ct = _utc_to_ct(bar.timestamp)
        ct_time = (ct.hour, ct.minute)
        day_of_week = ct.weekday()  # 0=Monday, 6=Sunday

        rth_open = self._hours.rth_open_time
        rth_close = self._hours.rth_close_time

        # Determine RTH/ETH
        ctx.is_rth = self._is_in_rth(ct_time, day_of_week)
        ctx.is_eth = not ctx.is_rth and self._is_in_globex(ct_time, day_of_week)

        # Session open: first RTH bar
        if ctx.is_rth and not self._prev_is_rth:
            ctx.session_open_bar = True
            self._rth_bar_count = 0
            # Compute overnight gap
            if self._last_session_close_price is not None:
                ctx.overnight_gap = bar.open - self._last_session_close_price

        self._rth_bar_count += 1

        # Session close detection: check if next bar would be outside RTH
        # We approximate by checking if we're near RTH close time
        close_minutes = rth_close[0] * 60 + rth_close[1]
        current_minutes = ct.hour * 60 + ct.minute
        bar_duration_minutes = max(1, bar.duration_ns // 60_000_000_000)

        if ctx.is_rth:
            ctx.minutes_to_session_close = max(0.0, close_minutes - current_minutes)
            if ctx.minutes_to_session_close <= bar_duration_minutes:
                ctx.session_close_bar = True
                self._last_session_close_price = bar.close

        # Daily settlement: typically at RTH close
        if ctx.session_close_bar:
            ctx.is_daily_settlement = True
            self._last_settlement_price = bar.close

        # Contract roll detection
        if bar.symbol and self._prev_symbol is not None:
            if bar.symbol != self._prev_symbol:
                ctx.contract_roll_pending = True
                logger.info(
                    "Contract roll detected: %s -> %s at bar %d",
                    self._prev_symbol,
                    bar.symbol,
                    bar_index,
                )
        self._prev_symbol = bar.symbol

        self._prev_is_rth = ctx.is_rth
        return ctx

    def _is_in_rth(
        self, ct_time: tuple[int, int], day_of_week: int
    ) -> bool:
        """Check if the given CT time falls within RTH."""
        # RTH only on weekdays (Mon-Fri = 0-4)
        if day_of_week > 4:
            return False

        rth_open = self._hours.rth_open_time
        rth_close = self._hours.rth_close_time
        current = ct_time[0] * 60 + ct_time[1]
        open_min = rth_open[0] * 60 + rth_open[1]
        close_min = rth_close[0] * 60 + rth_close[1]

        return open_min <= current < close_min

    def _is_in_globex(
        self, ct_time: tuple[int, int], day_of_week: int
    ) -> bool:
        """Check if the given CT time falls within Globex session."""
        # Simplified: Globex runs Sun 5 PM CT through Fri 4 PM CT
        # with daily halt at 4:15-4:30 PM CT and 5 PM reset
        current = ct_time[0] * 60 + ct_time[1]

        # Check daily halt
        if self._hours.daily_halt_start and self._hours.daily_halt_end:
            halt_start = (
                self._hours.daily_halt_start[0] * 60
                + self._hours.daily_halt_start[1]
            )
            halt_end = (
                self._hours.daily_halt_end[0] * 60
                + self._hours.daily_halt_end[1]
            )
            if halt_start <= current < halt_end:
                return False

        # Weekend check: Saturday is always closed
        if day_of_week == 5:
            return False

        # Sunday: only after 5 PM CT
        if day_of_week == 6:
            return current >= 17 * 60

        # Friday: only before 4 PM CT
        if day_of_week == 4:
            close_min = (
                self._hours.globex_close_time[0] * 60
                + self._hours.globex_close_time[1]
            )
            return current < close_min

        # Mon-Thu: all times except halt
        return True

    @property
    def last_settlement_price(self) -> Optional[float]:
        return self._last_settlement_price
