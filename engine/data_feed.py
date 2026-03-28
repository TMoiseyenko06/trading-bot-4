"""Data feed: strict forward-only iterator over Databento .dbn files."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional

import databento as db

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Bar:
    """Normalized OHLCV bar."""

    timestamp: datetime  # Bar close timestamp (UTC)
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str  # Full instrument id with expiry (e.g. "NQM5")
    duration_ns: int  # Bar duration in nanoseconds

    @property
    def duration_seconds(self) -> float:
        return self.duration_ns / 1_000_000_000

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2.0


# Fixed-price field scaling factor used by Databento
_DBN_PRICE_SCALE = 1e-9


def _to_price(raw: int) -> float:
    """Convert a Databento fixed-precision integer price to float."""
    return raw * _DBN_PRICE_SCALE


def _is_outright_contract(name: str) -> bool:
    """Check if a contract name is an outright future (not a spread)."""
    # Calendar spreads contain "-" (e.g., "ESM8-ESH9")
    # Butterfly spreads contain ":BF" (e.g., "YM:BF U5-Z5-H6")
    if "-" in name or ":" in name:
        return False
    return True


def build_instrument_map(metadata: object) -> dict[int, str]:
    """Build instrument_id -> contract_symbol mapping from DBN metadata.

    Parses metadata.mappings which is a dict of:
        contract_name -> [{'start_date', 'end_date', 'symbol': instrument_id_str}]

    Only includes outright contracts (no spreads).
    """
    id_to_symbol: dict[int, str] = {}

    mappings = getattr(metadata, "mappings", None)
    if not mappings or not isinstance(mappings, dict):
        logger.warning("No mappings dict found in metadata")
        return id_to_symbol

    for contract_name, intervals in mappings.items():
        # Skip calendar spreads and butterfly spreads
        if not _is_outright_contract(contract_name):
            continue

        for interval in intervals:
            if isinstance(interval, dict):
                iid_str = interval.get("symbol", "")
            else:
                iid_str = getattr(interval, "symbol", "")

            try:
                iid = int(iid_str)
                id_to_symbol[iid] = contract_name
            except (ValueError, TypeError):
                continue

    logger.info(
        "Built instrument map: %d outright contracts mapped", len(id_to_symbol)
    )
    return id_to_symbol


class DataFeed:
    """Strict forward-only iterator over a .dbn file.

    Yields one Bar at a time in timestamp order. Never loads the full dataset
    into memory. No random access. No backward seeking.

    Resolves instrument_id to contract symbol using the file's metadata
    mappings when the symbol field is not directly available on records.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._store: Optional[db.DBNStore] = None

    def __iter__(self) -> Iterator[Bar]:
        self._store = db.DBNStore.from_file(self._path)

        # Build instrument_id -> symbol mapping from metadata
        id_map = build_instrument_map(self._store.metadata)

        for record in self._store:
            bar = self._normalize(record, id_map)
            if bar is not None:
                yield bar

    @staticmethod
    def _normalize(
        record: object, id_map: dict[int, str]
    ) -> Optional[Bar]:
        """Convert a Databento OHLCV record into our Bar dataclass."""
        try:
            ts_event = record.ts_event  # type: ignore[attr-defined]
            if isinstance(ts_event, int):
                timestamp = datetime.fromtimestamp(
                    ts_event / 1_000_000_000, tz=timezone.utc
                )
            else:
                timestamp = ts_event

            # Resolve symbol: try record attribute first, then metadata map
            symbol = getattr(record, "symbol", None) or ""
            if not symbol:
                instrument_id = getattr(record, "instrument_id", None)
                if instrument_id is not None and instrument_id in id_map:
                    symbol = id_map[instrument_id]
                else:
                    # Skip records we can't identify
                    return None

            # Skip spread contracts
            if not _is_outright_contract(symbol):
                return None

            # Resolve prices: try pretty_ attributes first (already float),
            # fall back to raw int * scale
            open_price = getattr(record, "pretty_open", None)
            if open_price is None:
                open_price = _to_price(record.open)  # type: ignore[attr-defined]

            high_price = getattr(record, "pretty_high", None)
            if high_price is None:
                high_price = _to_price(record.high)  # type: ignore[attr-defined]

            low_price = getattr(record, "pretty_low", None)
            if low_price is None:
                low_price = _to_price(record.low)  # type: ignore[attr-defined]

            close_price = getattr(record, "pretty_close", None)
            if close_price is None:
                close_price = _to_price(record.close)  # type: ignore[attr-defined]

            duration_ns = _infer_duration_ns(record)

            return Bar(
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=record.volume,  # type: ignore[attr-defined]
                symbol=symbol,
                duration_ns=duration_ns,
            )
        except AttributeError as exc:
            logger.warning("Skipping record that lacks OHLCV fields: %s", exc)
            return None


def _infer_duration_ns(record: object) -> int:
    """Infer bar duration in nanoseconds from the record's rtype."""
    rtype = getattr(record, "rtype", None)
    # Databento rtype values for OHLCV schemas
    _RTYPE_DURATIONS = {
        32: 1_000_000_000,            # ohlcv-1s
        33: 60_000_000_000,           # ohlcv-1m
        34: 3_600_000_000_000,        # ohlcv-1h
        35: 86_400_000_000_000,       # ohlcv-1d
    }
    if rtype in _RTYPE_DURATIONS:
        return _RTYPE_DURATIONS[rtype]
    # Fallback: check for a duration field or default to 60s
    dur = getattr(record, "duration", None)
    if dur is not None:
        return int(dur)
    return 60_000_000_000  # default 1-minute
