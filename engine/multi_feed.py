"""Synchronized multi-instrument data feed.

Merges multiple .dbn files into a single forward-only timestamp-ordered
stream, grouping bars by timestamp so all instruments are available at
each time step.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

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
    """Merge multiple .dbn files into a synchronized bar stream.

    Bars from all instruments are grouped by timestamp. At each timestamp,
    the yielded MultiBar contains all instruments that have a bar at that
    exact time. Forward-only, no random access.
    """

    def __init__(self, dbn_paths: dict[str, str]) -> None:
        """
        Args:
            dbn_paths: Mapping of root_symbol -> dbn file path.
                       e.g. {"NQ": "data/NQ_1m.dbn", "ES": "data/ES_1m.dbn"}
        """
        self._dbn_paths = dbn_paths

    def __iter__(self) -> Iterator[MultiBar]:
        # Create iterators for each instrument
        iterators: dict[str, Iterator[Bar]] = {}
        for symbol, path in self._dbn_paths.items():
            iterators[symbol] = iter(DataFeed(path))

        # Use a min-heap to merge by timestamp
        # Heap entries: (timestamp, sequence_counter, symbol, bar)
        heap: list[tuple[datetime, int, str, Bar]] = []
        seq = 0

        # Seed the heap with the first bar from each iterator
        for symbol, it in iterators.items():
            try:
                bar = next(it)
                heapq.heappush(heap, (bar.timestamp, seq, symbol, bar))
                seq += 1
            except StopIteration:
                logger.warning("Empty data feed for %s", symbol)

        bar_index = 0
        while heap:
            # Peek at the earliest timestamp
            current_ts = heap[0][0]

            # Collect all bars at this timestamp
            bars_at_ts: dict[str, Bar] = {}
            while heap and heap[0][0] == current_ts:
                _, _, symbol, bar = heapq.heappop(heap)
                bars_at_ts[symbol] = bar

                # Advance that instrument's iterator
                try:
                    next_bar = next(iterators[symbol])
                    heapq.heappush(heap, (next_bar.timestamp, seq, symbol, next_bar))
                    seq += 1
                except StopIteration:
                    pass  # This instrument's data is exhausted

            yield MultiBar(
                timestamp=current_ts,
                bars=bars_at_ts,
                bar_index=bar_index,
            )
            bar_index += 1
