"""Synchronized multi-instrument data feed.

Supports two modes:
  1. Single .dbn file containing multiple instruments (records distinguished
     by the symbol field on each bar)
  2. Multiple .dbn files, one per instrument, merged by timestamp

In both cases, bars are grouped by timestamp and keyed by root symbol.
Forward-only, no random access.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

from engine.contract_registry import ContractRegistry
from engine.data_feed import Bar, DataFeed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiBar:
    """A synchronized group of bars at the same timestamp."""

    timestamp: datetime
    bars: dict[str, Bar]  # root_symbol -> Bar
    bar_index: int

    def has(self, symbol: str) -> bool:
        return symbol in self.bars

    def get(self, symbol: str) -> Bar | None:
        return self.bars.get(symbol)


class MultiInstrumentFeed:
    """Multi-instrument data feed with timestamp synchronization.

    Can be constructed from either:
      - A single .dbn file containing all instruments (from_single_file)
      - Multiple .dbn files, one per instrument (from_multiple_files)
    """

    def __init__(self) -> None:
        self._single_path: str | None = None
        self._multi_paths: dict[str, str] | None = None
        self._instruments: list[str] | None = None
        self._registry = ContractRegistry()

    @classmethod
    def from_single_file(
        cls, path: str, instruments: list[str] | None = None
    ) -> "MultiInstrumentFeed":
        """Create feed from a single .dbn file with multiple instruments.

        Args:
            path: Path to .dbn file containing bars for all instruments.
            instruments: Optional filter — only yield bars for these root
                         symbols. If None, include all symbols found.
        """
        feed = cls()
        feed._single_path = path
        feed._instruments = instruments
        return feed

    @classmethod
    def from_multiple_files(cls, dbn_paths: dict[str, str]) -> "MultiInstrumentFeed":
        """Create feed from multiple .dbn files, one per instrument.

        Args:
            dbn_paths: Mapping of root_symbol -> dbn file path.
                       e.g. {"NQ": "data/NQ_1m.dbn", "ES": "data/ES_1m.dbn"}
        """
        feed = cls()
        feed._multi_paths = dbn_paths
        return feed

    def __iter__(self) -> Iterator[MultiBar]:
        if self._single_path is not None:
            yield from self._iter_single_file()
        elif self._multi_paths is not None:
            yield from self._iter_multiple_files()
        else:
            raise RuntimeError(
                "Feed not configured. Use from_single_file or from_multiple_files."
            )

    def _iter_single_file(self) -> Iterator[MultiBar]:
        """Iterate a single .dbn file, grouping bars by timestamp and root symbol."""
        assert self._single_path is not None
        data_feed = DataFeed(self._single_path)

        allowed_roots: set[str] | None = None
        if self._instruments:
            allowed_roots = set(self._instruments)

        bar_index = 0
        current_ts: datetime | None = None
        bars_at_ts: dict[str, Bar] = {}

        for bar in data_feed:
            # Extract root symbol from the bar's symbol field
            root = self._registry.extract_root(
                bar.symbol, known_roots=set(self._registry.list_symbols())
            )

            # Filter to requested instruments
            if allowed_roots and root not in allowed_roots:
                continue

            if current_ts is None:
                current_ts = bar.timestamp

            # If we've moved to a new timestamp, yield the previous group
            if bar.timestamp != current_ts:
                if bars_at_ts:
                    yield MultiBar(
                        timestamp=current_ts,
                        bars=bars_at_ts,
                        bar_index=bar_index,
                    )
                    bar_index += 1
                current_ts = bar.timestamp
                bars_at_ts = {}

            bars_at_ts[root] = bar

        # Yield the final group
        if bars_at_ts and current_ts is not None:
            yield MultiBar(
                timestamp=current_ts,
                bars=bars_at_ts,
                bar_index=bar_index,
            )

    def _iter_multiple_files(self) -> Iterator[MultiBar]:
        """Merge multiple .dbn files by timestamp using a min-heap."""
        assert self._multi_paths is not None

        iterators: dict[str, Iterator[Bar]] = {}
        for symbol, path in self._multi_paths.items():
            iterators[symbol] = iter(DataFeed(path))

        heap: list[tuple[datetime, int, str, Bar]] = []
        seq = 0

        for symbol, it in iterators.items():
            try:
                bar = next(it)
                heapq.heappush(heap, (bar.timestamp, seq, symbol, bar))
                seq += 1
            except StopIteration:
                logger.warning("Empty data feed for %s", symbol)

        bar_index = 0
        while heap:
            current_ts = heap[0][0]

            bars_at_ts: dict[str, Bar] = {}
            while heap and heap[0][0] == current_ts:
                _, _, symbol, bar = heapq.heappop(heap)
                bars_at_ts[symbol] = bar

                try:
                    next_bar = next(iterators[symbol])
                    heapq.heappush(
                        heap, (next_bar.timestamp, seq, symbol, next_bar)
                    )
                    seq += 1
                except StopIteration:
                    pass

            yield MultiBar(
                timestamp=current_ts,
                bars=bars_at_ts,
                bar_index=bar_index,
            )
            bar_index += 1
