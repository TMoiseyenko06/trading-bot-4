"""Data feed: strict forward-only iterator over Databento .dbn files."""

from __future__ import annotations

import logging
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


class DataFeed:
    """Strict forward-only iterator over a .dbn file.

    Yields one Bar at a time in timestamp order. Never loads the full dataset
    into memory. No random access. No backward seeking.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._store: Optional[db.DBNStore] = None

    def __iter__(self) -> Iterator[Bar]:
        self._store = db.DBNStore.from_file(self._path)
        for record in self._store:
            bar = self._normalize(record)
            if bar is not None:
                yield bar

    @staticmethod
    def _normalize(record: object) -> Optional[Bar]:
        """Convert a Databento OHLCV record into our Bar dataclass."""
        # Databento OHLCV bars have: open, high, low, close, volume,
        # ts_event (bar close), and rtype-based duration
        try:
            ts_event = record.ts_event  # type: ignore[attr-defined]
            if isinstance(ts_event, int):
                timestamp = datetime.fromtimestamp(
                    ts_event / 1_000_000_000, tz=timezone.utc
                )
            else:
                timestamp = ts_event

            # Extract symbol from the record's instrument metadata
            # Databento records carry the raw symbol via the `symbol` field
            # on pretty-printed records or via the parent store's symbology
            symbol = getattr(record, "symbol", None) or ""

            # Determine bar duration from the schema / rtype
            # Common Databento OHLCV rtypes and their durations:
            #   ohlcv-1s -> 1s, ohlcv-1m -> 60s, ohlcv-1h -> 3600s,
            #   ohlcv-1d -> 86400s
            duration_ns = _infer_duration_ns(record)

            return Bar(
                timestamp=timestamp,
                open=_to_price(record.open),  # type: ignore[attr-defined]
                high=_to_price(record.high),  # type: ignore[attr-defined]
                low=_to_price(record.low),  # type: ignore[attr-defined]
                close=_to_price(record.close),  # type: ignore[attr-defined]
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
