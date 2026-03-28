#!/usr/bin/env python3
"""Inspect a .dbn file to see symbol mappings and record structure."""
import sys
import databento as db

path = sys.argv[1] if len(sys.argv) > 1 else "data/multi.dbn"
store = db.DBNStore.from_file(path)

print("=== Store metadata ===")
meta = store.metadata
print(f"  Schema: {meta.schema}")
print(f"  Start: {meta.start}")
print(f"  End: {meta.end}")
print(f"  Symbols: {meta.symbols}")
print(f"  Stype in: {meta.stype_in}")
print(f"  Stype out: {meta.stype_out}")

print("\n=== Symbol mappings ===")
if hasattr(meta, 'mappings') and meta.mappings:
    for m in meta.mappings:
        print(f"  {m}")
elif hasattr(meta, 'symbol_map') and meta.symbol_map:
    for k, v in meta.symbol_map.items():
        print(f"  {k} -> {v}")

print("\n=== Symbology resolution ===")
if hasattr(store, 'symbology_map'):
    print(f"  symbology_map: {store.symbology_map}")
if hasattr(store, 'symbol_map'):
    print(f"  symbol_map type: {type(store.symbol_map)}")
    try:
        for k, v in store.symbol_map.items():
            print(f"    {k} -> {v}")
    except Exception:
        print(f"    {store.symbol_map}")

print("\n=== First 10 records (raw attributes) ===")
store2 = db.DBNStore.from_file(path)
count = 0
for rec in store2:
    if count >= 10:
        break
    iid = getattr(rec, 'instrument_id', getattr(rec, 'hd', {None: None}))
    hd = getattr(rec, 'hd', None)
    sym = getattr(rec, 'symbol', None)
    pretty_sym = getattr(rec, 'pretty_symbol', None)
    print(f"  instrument_id={iid}  hd={hd}  symbol={sym!r}  pretty_symbol={pretty_sym!r}")
    if count == 0:
        print(f"  All attrs: {[a for a in dir(rec) if not a.startswith('_')]}")
    count += 1
