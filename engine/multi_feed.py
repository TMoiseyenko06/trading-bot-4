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
        """Iterate a single .dbn file, grouping bars by timestamp and root symbol.

        When multiple contracts exist for the same root at the same timestamp
        (front + back month during roll periods), uses cumulative session
        volume to pick the front month. This prevents flip-flopping between
        contracts on individual bars where volume leadership is noisy.
        """
        assert self._single_path is not None
        data_feed = DataFeed(self._single_path)

        allowed_roots: set[str] | None = None
        known_roots = set(self._registry.list_symbols())
        if self._instruments:
            allowed_roots = set(self._instruments)

        bar_index = 0
        current_ts: datetime | None = None
        # At each timestamp, collect ALL bars per root (may have multiple contracts)
        candidates_at_ts: dict[str, list[Bar]] = {}

        # Track cumulative volume per contract symbol to determine front month
        cum_volume: dict[str, int] = {}
        # Track current front month per root for stability
        current_front: dict[str, str] = {}

        for bar in data_feed:
            root = self._registry.extract_root(bar.symbol, known_roots=known_roots)

            if allowed_roots and root not in allowed_roots:
                continue

            if current_ts is None:
                current_ts = bar.timestamp

            # New timestamp: resolve previous group and yield
            if bar.timestamp != current_ts:
                resolved = self._resolve_front_months(
                    candidates_at_ts, cum_volume, current_front
                )
                if resolved:
                    yield MultiBar(
                        timestamp=current_ts,
                        bars=resolved,
                        bar_index=bar_index,
                    )
                    bar_index += 1
                current_ts = bar.timestamp
                candidates_at_ts = {}

            # Accumulate candidates
            if root not in candidates_at_ts:
                candidates_at_ts[root] = []
            candidates_at_ts[root].append(bar)

            # Track cumulative volume per contract
            cum_volume[bar.symbol] = cum_volume.get(bar.symbol, 0) + bar.volume

        # Yield final group
        if candidates_at_ts and current_ts is not None:
            resolved = self._resolve_front_months(
                candidates_at_ts, cum_volume, current_front
            )
            if resolved:
                yield MultiBar(
                    timestamp=current_ts,
                    bars=resolved,
                    bar_index=bar_index,
                )

    @staticmethod
    def _resolve_front_months(
        candidates: dict[str, list[Bar]],
        cum_volume: dict[str, int],
        current_front: dict[str, str],
    ) -> dict[str, Bar]:
        """Pick the front-month contract for each root symbol.

        Uses cumulative volume to determine the front month. Once a contract
        takes the cumulative volume lead, it stays as front month until
        another contract surpasses it. This prevents noisy flip-flopping
        during roll periods.
        """
        result: dict[str, Bar] = {}
        for root, bars in candidates.items():
            if len(bars) == 1:
                chosen = bars[0]
            else:
                # Sort by cumulative volume (descending)
                bars.sort(
                    key=lambda b: cum_volume.get(b.symbol, 0), reverse=True
                )
                chosen = bars[0]

                # If we already have a front month for this root, prefer
                # sticking with it unless a different contract has pulled
                # clearly ahead in cumulative volume
                if root in current_front:
                    prev_sym = current_front[root]
                    prev_vol = cum_volume.get(prev_sym, 0)
                    new_vol = cum_volume.get(chosen.symbol, 0)
                    if chosen.symbol != prev_sym and new_vol < prev_vol * 1.1:
                        # Previous front month still has more cumulative volume
                        # (or within 10%), stick with it
                        for b in bars:
                            if b.symbol == prev_sym:
                                chosen = b
                                break

            current_front[root] = chosen.symbol
            result[root] = chosen
        return result

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
