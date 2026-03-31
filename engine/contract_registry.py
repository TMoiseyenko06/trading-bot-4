"""Contract specification registry for CME-style futures contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SessionHours:
    """Trading session hours definition in CT (Central Time).

    Times are stored as (hour, minute) tuples in 24-hour format CT.
    """

    globex_open_day: int  # day of week, 6=Sunday
    globex_open_time: tuple[int, int]  # (hour, minute) CT
    globex_close_day: int  # day of week, 4=Friday
    globex_close_time: tuple[int, int]  # (hour, minute) CT
    rth_open_time: tuple[int, int]  # (hour, minute) CT
    rth_close_time: tuple[int, int]  # (hour, minute) CT
    daily_halt_start: Optional[tuple[int, int]] = None  # (hour, minute) CT
    daily_halt_end: Optional[tuple[int, int]] = None  # (hour, minute) CT
    daily_reset_time: Optional[tuple[int, int]] = None  # (hour, minute) CT


@dataclass(frozen=True)
class ContractSpec:
    """Specification for a single futures contract product."""

    symbol: str  # Root symbol (e.g., "NQ", "ES")
    exchange: str
    tick_size: float  # Minimum price increment
    point_value: float  # Dollar value per full point move
    contract_multiplier: float  # Usually same as point_value
    commission_per_side: float  # USD per contract per side
    initial_margin: float  # Initial margin per contract
    maintenance_margin: float  # Maintenance margin per contract
    intraday_margin: float  # Intraday margin (often lower)
    session_hours: SessionHours
    description: str = ""

    @property
    def tick_value(self) -> float:
        """Dollar value of one tick move."""
        return self.tick_size * self.point_value

    def snap_price_to_tick(self, price: float) -> float:
        """Round a price to the nearest valid tick increment."""
        ticks = round(price / self.tick_size)
        return round(ticks * self.tick_size, 10)

    def is_valid_tick(self, price: float) -> bool:
        """Check if a price falls on a valid tick boundary."""
        remainder = price / self.tick_size
        return abs(remainder - round(remainder)) < 1e-9


# --------------------------------------------------------------------------- #
# Pre-populated session definitions
# --------------------------------------------------------------------------- #

_CME_EQUITY_SESSION = SessionHours(
    globex_open_day=6,
    globex_open_time=(17, 0),
    globex_close_day=4,
    globex_close_time=(16, 0),
    rth_open_time=(8, 30),
    rth_close_time=(15, 15),
    daily_halt_start=(16, 15),
    daily_halt_end=(16, 30),
    daily_reset_time=(17, 0),
)

_CME_ENERGY_SESSION = SessionHours(
    globex_open_day=6,
    globex_open_time=(17, 0),
    globex_close_day=4,
    globex_close_time=(16, 0),
    rth_open_time=(8, 0),
    rth_close_time=(13, 30),
)

_COMEX_METALS_SESSION = SessionHours(
    globex_open_day=6,
    globex_open_time=(17, 0),
    globex_close_day=4,
    globex_close_time=(16, 0),
    rth_open_time=(7, 20),
    rth_close_time=(12, 30),
)

_CBOT_RATES_SESSION = SessionHours(
    globex_open_day=6,
    globex_open_time=(17, 0),
    globex_close_day=4,
    globex_close_time=(16, 0),
    rth_open_time=(7, 20),
    rth_close_time=(13, 0),
)


class ContractRegistry:
    """Central registry mapping root symbols to their contract specifications."""

    def __init__(self) -> None:
        self._specs: dict[str, ContractSpec] = {}
        self._populate_defaults()

    def _populate_defaults(self) -> None:
        defaults = [
            ContractSpec(
                symbol="NQ",
                exchange="CME",
                tick_size=0.25,
                point_value=20.0,
                contract_multiplier=20.0,
                commission_per_side=1.40,
                initial_margin=18_700.0,
                maintenance_margin=17_000.0,
                intraday_margin=9_350.0,
                session_hours=_CME_EQUITY_SESSION,
                description="E-mini NASDAQ-100 Futures",
            ),
            ContractSpec(
                symbol="MNQ",
                exchange="CME",
                tick_size=0.25,
                point_value=2.0,
                contract_multiplier=2.0,
                commission_per_side=0.62,
                initial_margin=1_870.0,
                maintenance_margin=1_700.0,
                intraday_margin=935.0,
                session_hours=_CME_EQUITY_SESSION,
                description="Micro E-mini NASDAQ-100 Futures",
            ),
            ContractSpec(
                symbol="ES",
                exchange="CME",
                tick_size=0.25,
                point_value=50.0,
                contract_multiplier=50.0,
                commission_per_side=1.40,
                initial_margin=12_650.0,
                maintenance_margin=11_500.0,
                intraday_margin=6_325.0,
                session_hours=_CME_EQUITY_SESSION,
                description="E-mini S&P 500 Futures",
            ),
            ContractSpec(
                symbol="MES",
                exchange="CME",
                tick_size=0.25,
                point_value=5.0,
                contract_multiplier=5.0,
                commission_per_side=0.62,
                initial_margin=1_265.0,
                maintenance_margin=1_150.0,
                intraday_margin=632.5,
                session_hours=_CME_EQUITY_SESSION,
                description="Micro E-mini S&P 500 Futures",
            ),
            ContractSpec(
                symbol="RTY",
                exchange="CME",
                tick_size=0.10,
                point_value=50.0,
                contract_multiplier=50.0,
                commission_per_side=1.40,
                initial_margin=7_150.0,
                maintenance_margin=6_500.0,
                intraday_margin=3_575.0,
                session_hours=_CME_EQUITY_SESSION,
                description="E-mini Russell 2000 Futures",
            ),
            ContractSpec(
                symbol="M2K",
                exchange="CME",
                tick_size=0.10,
                point_value=5.0,
                contract_multiplier=5.0,
                commission_per_side=0.62,
                initial_margin=715.0,
                maintenance_margin=650.0,
                intraday_margin=357.5,
                session_hours=_CME_EQUITY_SESSION,
                description="Micro E-mini Russell 2000 Futures",
            ),
            ContractSpec(
                symbol="YM",
                exchange="CME",
                tick_size=1.0,
                point_value=5.0,
                contract_multiplier=5.0,
                commission_per_side=1.40,
                initial_margin=9_900.0,
                maintenance_margin=9_000.0,
                intraday_margin=4_950.0,
                session_hours=_CME_EQUITY_SESSION,
                description="E-mini Dow Futures",
            ),
            ContractSpec(
                symbol="MYM",
                exchange="CME",
                tick_size=1.0,
                point_value=0.50,
                contract_multiplier=0.50,
                commission_per_side=0.62,
                initial_margin=990.0,
                maintenance_margin=900.0,
                intraday_margin=495.0,
                session_hours=_CME_EQUITY_SESSION,
                description="Micro E-mini Dow Futures",
            ),
            ContractSpec(
                symbol="CL",
                exchange="CME",
                tick_size=0.01,
                point_value=1000.0,
                contract_multiplier=1000.0,
                commission_per_side=2.25,
                initial_margin=6_600.0,
                maintenance_margin=6_000.0,
                intraday_margin=3_300.0,
                session_hours=_CME_ENERGY_SESSION,
                description="Crude Oil Futures",
            ),
            ContractSpec(
                symbol="GC",
                exchange="COMEX",
                tick_size=0.10,
                point_value=100.0,
                contract_multiplier=100.0,
                commission_per_side=2.25,
                initial_margin=10_000.0,
                maintenance_margin=9_100.0,
                intraday_margin=5_000.0,
                session_hours=_COMEX_METALS_SESSION,
                description="Gold Futures",
            ),
            ContractSpec(
                symbol="ZB",
                exchange="CBOT",
                tick_size=1 / 32,
                point_value=1000.0,
                contract_multiplier=1000.0,
                commission_per_side=1.40,
                initial_margin=4_400.0,
                maintenance_margin=4_000.0,
                intraday_margin=2_200.0,
                session_hours=_CBOT_RATES_SESSION,
                description="30-Year U.S. Treasury Bond Futures",
            ),
            ContractSpec(
                symbol="ZN",
                exchange="CBOT",
                tick_size=1 / 64,
                point_value=1000.0,
                contract_multiplier=1000.0,
                commission_per_side=1.40,
                initial_margin=2_200.0,
                maintenance_margin=2_000.0,
                intraday_margin=1_100.0,
                session_hours=_CBOT_RATES_SESSION,
                description="10-Year U.S. Treasury Note Futures",
            ),
            ContractSpec(
                symbol="ZT",
                exchange="CBOT",
                tick_size=1 / 128,
                point_value=2000.0,
                contract_multiplier=2000.0,
                commission_per_side=1.40,
                initial_margin=760.0,
                maintenance_margin=690.0,
                intraday_margin=380.0,
                session_hours=_CBOT_RATES_SESSION,
                description="2-Year U.S. Treasury Note Futures",
            ),
            ContractSpec(
                symbol="ZF",
                exchange="CBOT",
                tick_size=1 / 128,
                point_value=1000.0,
                contract_multiplier=1000.0,
                commission_per_side=1.40,
                initial_margin=1_100.0,
                maintenance_margin=1_000.0,
                intraday_margin=550.0,
                session_hours=_CBOT_RATES_SESSION,
                description="5-Year U.S. Treasury Note Futures",
            ),
        ]
        for spec in defaults:
            self._specs[spec.symbol] = spec

    def get(self, symbol: str) -> ContractSpec:
        """Look up a contract spec by root symbol.

        Accepts full contract identifiers like 'NQM5' and extracts the root.
        """
        # Direct match first
        if symbol in self._specs:
            return self._specs[symbol]
        root = self.extract_root(symbol, known_roots=set(self._specs.keys()))
        if root not in self._specs:
            raise KeyError(
                f"No contract spec found for root symbol '{root}' "
                f"(from '{symbol}'). Available: {list(self._specs.keys())}"
            )
        return self._specs[root]

    def register(self, spec: ContractSpec) -> None:
        """Register a custom contract specification."""
        self._specs[spec.symbol] = spec

    @staticmethod
    def extract_root(
        symbol: str, known_roots: set[str] | None = None
    ) -> str:
        """Extract the root symbol from a full futures identifier.

        Examples: 'NQM5' -> 'NQ', 'ESZ25' -> 'ES', 'NQ' -> 'NQ',
                  'M2KZ5' -> 'M2K', 'RTYM5' -> 'RTY'
        """
        month_codes = "FGHJKMNQUVXZ"

        # Try known roots first (longest match wins)
        if known_roots:
            for length in range(len(symbol), 0, -1):
                candidate = symbol[:length]
                if candidate in known_roots:
                    return candidate

        # Try splitting at each position: find shortest prefix where
        # the remainder is month_code + year digits
        for i in range(1, len(symbol) + 1):
            prefix = symbol[:i]
            rest = symbol[i:]
            if rest and rest[0] in month_codes:
                after_month = rest[1:]
                if after_month and after_month[0].isdigit():
                    return prefix

        # No month+year suffix found — return the whole symbol as root
        return symbol

    def list_symbols(self) -> list[str]:
        """List all registered root symbols."""
        return sorted(self._specs.keys())
