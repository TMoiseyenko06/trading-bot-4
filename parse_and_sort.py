#!/usr/bin/env python3
"""Parse CSV from stdin, sort by net_pnl, print tables."""
import csv, sys, io

data = sys.stdin.read()
reader = csv.DictReader(io.StringIO(data))
rows = list(reader)
print(f"Loaded {len(rows)} results", file=sys.stderr)

rows.sort(key=lambda r: float(r["net_pnl"]), reverse=True)

def prt(rlist, title, limit=50):
    hdr = (f"{'#':>3}  {'LB':>3} {'Zin':>4} {'Zout':>4} {'Cf':>2} {'Stop':>4} {'Mom':>4} {'Hld':>3} {'Skp':>3}  "
           f"{'Net PnL':>12}  {'PF':>6}  {'WR%':>5}  {'Sharpe':>7}  {'MaxDD%':>6}  {'Trades':>6}")
    sep = "-" * len(hdr)
    print(f"\n{'='*len(hdr)}\n  {title}\n{'='*len(hdr)}")
    print(hdr); print(sep)
    for i, r in enumerate(rlist[:limit], 1):
        pf = float(r["profit_factor"])
        pf_s = f"{pf:.2f}" if pf < 100 else "inf"
        print(f"{i:>3}  {int(r['lookback_bars']):>3} {float(r['z_entry']):>4.1f} {float(r['z_exit']):>4.1f} "
              f"{int(r['confirm_bars']):>2} {float(r['stop_multiple']):>4.1f} {float(r['momentum_threshold']):>4.2f} "
              f"{int(r['min_hold_bars']):>3} {int(r['skip_first_minutes']):>3}  "
              f"${float(r['net_pnl']):>11,.0f}  {pf_s:>6}  {float(r['win_rate'])*100:>5.1f}  "
              f"{float(r['sharpe']):>7.2f}  {float(r['max_dd_pct'])*100:>5.1f}%  "
              f"{int(r['total_trades']):>6}")
    print(f"  ({len(rlist)} total in this category)")

prt(rows, "ALL RESULTS — TOP 50 BY NET PnL", 50)

robust = sorted([r for r in rows if int(r["total_trades"]) >= 200 and 1.0 < float(r["profit_factor"]) < 100],
                key=lambda r: float(r["net_pnl"]), reverse=True)
prt(robust, "ROBUST (200+ trades, 1.0 < PF < 100) — TOP 40 BY NET PnL", 40)

vrob = sorted([r for r in rows if int(r["total_trades"]) >= 1000 and 1.05 < float(r["profit_factor"]) < 100],
              key=lambda r: float(r["net_pnl"]), reverse=True)
prt(vrob, "VERY ROBUST (1000+ trades, PF > 1.05) — TOP 20 BY NET PnL", 20)
