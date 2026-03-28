#!/usr/bin/env python3
"""Inspect the first 20 bars of a .dbn file to see symbol format."""
import sys
from engine.data_feed import DataFeed

path = sys.argv[1] if len(sys.argv) > 1 else "data/multi.dbn"
feed = DataFeed(path)
count = 0
symbols_seen = set()
for bar in feed:
    symbols_seen.add(bar.symbol)
    if count < 20:
        print(
            f"  ts={bar.timestamp}  symbol={bar.symbol!r}  "
            f"O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} C={bar.close:.2f} V={bar.volume}"
        )
    count += 1

print(f"\nTotal bars: {count}")
print(f"Unique symbols: {symbols_seen}")
