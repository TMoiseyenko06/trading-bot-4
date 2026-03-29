#!/usr/bin/env python3
"""Sort grid search results by net PnL from CSV file."""
import csv
import sys

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results_grid/grid_search_results.csv"

    with open(path) as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} results\n")

    # Sort by net_pnl descending
    rows.sort(key=lambda r: float(r["net_pnl"]), reverse=True)

    hdr = (f"{'#':>3}  {'LB':>3} {'Zin':>4} {'Zout':>4} {'Cf':>2} {'Stop':>4} {'Mom':>4} {'Hld':>3} {'Skp':>3}  "
           f"{'Net PnL':>12}  {'PF':>6}  {'WR%':>5}  {'Sharpe':>7}  {'MaxDD%':>6}  {'Trades':>6}")
    sep = "-" * len(hdr)

    # ---- ALL RESULTS TOP 50 ----
    print("=" * len(hdr))
    print("  TOP 50 BY NET PnL (ALL)")
    print("=" * len(hdr))
    print(hdr)
    print(sep)
    for i, r in enumerate(rows[:50], 1):
        pf = float(r["profit_factor"])
        pf_s = f"{pf:.2f}" if pf < 100 else "inf"
        print(f"{i:>3}  {int(r['lookback_bars']):>3} {float(r['z_entry']):>4.1f} {float(r['z_exit']):>4.1f} "
              f"{int(r['confirm_bars']):>2} {float(r['stop_multiple']):>4.1f} {float(r['momentum_threshold']):>4.2f} "
              f"{int(r['min_hold_bars']):>3} {int(r['skip_first_minutes']):>3}  "
              f"${float(r['net_pnl']):>11,.0f}  {pf_s:>6}  {float(r['win_rate'])*100:>5.1f}  "
              f"{float(r['sharpe']):>7.2f}  {float(r['max_dd_pct'])*100:>5.1f}%  "
              f"{int(r['total_trades']):>6}")

    # ---- ROBUST: 200+ trades, PF > 1.0 ----
    robust = [r for r in rows if int(r["total_trades"]) >= 200 and float(r["profit_factor"]) > 1.0 and float(r["profit_factor"]) < 100]
    robust.sort(key=lambda r: float(r["net_pnl"]), reverse=True)

    print(f"\n{'=' * len(hdr)}")
    print(f"  TOP 40 BY NET PnL (200+ trades, PF>1.0, PF<100)")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print(sep)
    for i, r in enumerate(robust[:40], 1):
        pf = float(r["profit_factor"])
        pf_s = f"{pf:.2f}" if pf < 100 else "inf"
        print(f"{i:>3}  {int(r['lookback_bars']):>3} {float(r['z_entry']):>4.1f} {float(r['z_exit']):>4.1f} "
              f"{int(r['confirm_bars']):>2} {float(r['stop_multiple']):>4.1f} {float(r['momentum_threshold']):>4.2f} "
              f"{int(r['min_hold_bars']):>3} {int(r['skip_first_minutes']):>3}  "
              f"${float(r['net_pnl']):>11,.0f}  {pf_s:>6}  {float(r['win_rate'])*100:>5.1f}  "
              f"{float(r['sharpe']):>7.2f}  {float(r['max_dd_pct'])*100:>5.1f}%  "
              f"{int(r['total_trades']):>6}")
    print(f"\n  Total robust combos: {len(robust)}")

    # ---- MOST ROBUST: 1000+ trades, PF > 1.05 ----
    very_robust = [r for r in rows if int(r["total_trades"]) >= 1000 and float(r["profit_factor"]) > 1.05 and float(r["profit_factor"]) < 100]
    very_robust.sort(key=lambda r: float(r["net_pnl"]), reverse=True)

    print(f"\n{'=' * len(hdr)}")
    print(f"  TOP 20 BY NET PnL (1000+ trades, PF>1.05)")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print(sep)
    for i, r in enumerate(very_robust[:20], 1):
        pf = float(r["profit_factor"])
        pf_s = f"{pf:.2f}" if pf < 100 else "inf"
        print(f"{i:>3}  {int(r['lookback_bars']):>3} {float(r['z_entry']):>4.1f} {float(r['z_exit']):>4.1f} "
              f"{int(r['confirm_bars']):>2} {float(r['stop_multiple']):>4.1f} {float(r['momentum_threshold']):>4.2f} "
              f"{int(r['min_hold_bars']):>3} {int(r['skip_first_minutes']):>3}  "
              f"${float(r['net_pnl']):>11,.0f}  {pf_s:>6}  {float(r['win_rate'])*100:>5.1f}  "
              f"{float(r['sharpe']):>7.2f}  {float(r['max_dd_pct'])*100:>5.1f}%  "
              f"{int(r['total_trades']):>6}")
    print(f"\n  Total very robust combos: {len(very_robust)}")


if __name__ == "__main__":
    main()
