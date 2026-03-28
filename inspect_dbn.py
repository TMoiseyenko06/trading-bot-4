#!/usr/bin/env python3
"""Inspect .dbn metadata mappings to understand instrument_id -> symbol resolution."""
import sys
import databento as db

path = sys.argv[1] if len(sys.argv) > 1 else "data/multi.dbn"
store = db.DBNStore.from_file(path)
meta = store.metadata

print("=== Metadata ===")
print(f"  symbols: {meta.symbols}")
print(f"  stype_in: {meta.stype_in}")
print(f"  stype_out: {meta.stype_out}")

print("\n=== Mappings type and structure ===")
print(f"  type(mappings): {type(meta.mappings)}")
if isinstance(meta.mappings, dict):
    keys = list(meta.mappings.keys())[:5]
    print(f"  First 5 keys: {keys}")
    for k in keys[:2]:
        v = meta.mappings[k]
        print(f"  mappings[{k!r}] = {v}")
        print(f"    type: {type(v)}")
        if isinstance(v, list) and v:
            item = v[0]
            print(f"    first item: {item}")
            print(f"    item type: {type(item)}")
            print(f"    item attrs: {[a for a in dir(item) if not a.startswith('_')]}")
            for attr in dir(item):
                if not attr.startswith('_'):
                    print(f"      .{attr} = {getattr(item, attr, '?')}")
elif isinstance(meta.mappings, list):
    print(f"  len: {len(meta.mappings)}")
    if meta.mappings:
        item = meta.mappings[0]
        print(f"  first item: {item}")
        print(f"  item type: {type(item)}")
        print(f"  item attrs: {[a for a in dir(item) if not a.startswith('_')]}")
        for attr in dir(item):
            if not attr.startswith('_'):
                try:
                    print(f"    .{attr} = {getattr(item, attr)}")
                except Exception as e:
                    print(f"    .{attr} = ERROR: {e}")

print("\n=== Try to_df on first 20 rows ===")
try:
    store2 = db.DBNStore.from_file(path)
    df = store2.to_df()
    print(f"  columns: {list(df.columns)}")
    print(df.head(20).to_string())
except Exception as e:
    print(f"  to_df failed: {e}")
