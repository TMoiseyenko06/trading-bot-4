#!/usr/bin/env python3
"""Analyze grid search results sorted by net PnL."""
import csv
import io
import sys

def main():
    with open("/dev/stdin" if len(sys.argv) < 2 else sys.argv[1]) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Sort by net_pnl descending
    rows.sort(key=lambda r: float(r["net_pnl"]), reverse=True)

    # Print header
    print(f"{'#':>3}  {'LB':>3} {'Zin':>4} {'Zout':>4} {'Conf':>4} {'Stop':>4} {'Mom':>5} {'Hold':>4} {'Skip':>4}  "
          f"{'Net PnL':>12}  {'PF':>6}  {'WR%':>5}  {'Sharpe':>7}  {'MaxDD%':>6}  {'Trades':>6}  {'AvgW':>8}  {'AvgL':>8}")
    print("-" * 130)

    for i, r in enumerate(rows[:50], 1):
        pf = float(r["profit_factor"])
        pf_str = f"{pf:.2f}" if pf < 100 else "inf"
        trades = int(r["total_trades"])
        print(
            f"{i:>3}  {int(r['lookback_bars']):>3} {float(r['z_entry']):>4.1f} {float(r['z_exit']):>4.1f} "
            f"{int(r['confirm_bars']):>4} {float(r['stop_multiple']):>4.1f} {float(r['momentum_threshold']):>5.2f} "
            f"{int(r['min_hold_bars']):>4} {int(r['skip_first_minutes']):>4}  "
            f"${float(r['net_pnl']):>11,.0f}  {pf_str:>6}  {float(r['win_rate'])*100:>5.1f}  "
            f"{float(r['sharpe']):>7.2f}  {float(r['max_dd_pct'])*100:>5.1f}%  "
            f"{trades:>6}  ${float(r['avg_winner']):>7,.0f}  ${float(r['avg_loser']):>7,.0f}"
        )

    # Summary stats
    print("\n" + "=" * 130)
    print("FILTERING: Combinations with 200+ trades and PF > 1.0:")
    print("=" * 130)

    robust = [r for r in rows if int(r["total_trades"]) >= 200 and float(r["profit_factor"]) > 1.0]
    robust.sort(key=lambda r: float(r["net_pnl"]), reverse=True)

    print(f"{'#':>3}  {'LB':>3} {'Zin':>4} {'Zout':>4} {'Conf':>4} {'Stop':>4} {'Mom':>5} {'Hold':>4} {'Skip':>4}  "
          f"{'Net PnL':>12}  {'PF':>6}  {'WR%':>5}  {'Sharpe':>7}  {'MaxDD%':>6}  {'Trades':>6}  {'AvgW':>8}  {'AvgL':>8}")
    print("-" * 130)

    for i, r in enumerate(robust[:40], 1):
        pf = float(r["profit_factor"])
        pf_str = f"{pf:.2f}" if pf < 100 else "inf"
        trades = int(r["total_trades"])
        print(
            f"{i:>3}  {int(r['lookback_bars']):>3} {float(r['z_entry']):>4.1f} {float(r['z_exit']):>4.1f} "
            f"{int(r['confirm_bars']):>4} {float(r['stop_multiple']):>4.1f} {float(r['momentum_threshold']):>5.2f} "
            f"{int(r['min_hold_bars']):>4} {int(r['skip_first_minutes']):>4}  "
            f"${float(r['net_pnl']):>11,.0f}  {pf_str:>6}  {float(r['win_rate'])*100:>5.1f}  "
            f"{float(r['sharpe']):>7.2f}  {float(r['max_dd_pct'])*100:>5.1f}%  "
            f"{trades:>6}  ${float(r['avg_winner']):>7,.0f}  ${float(r['avg_loser']):>7,.0f}"
        )

    print(f"\n  Total robust combinations (200+ trades, PF>1.0): {len(robust)}")

    # Also show top by Sharpe with 500+ trades
    print("\n" + "=" * 130)
    print("TOP BY SHARPE RATIO (500+ trades):")
    print("=" * 130)
    sharpe_list = [r for r in rows if int(r["total_trades"]) >= 500]
    sharpe_list.sort(key=lambda r: float(r["sharpe"]), reverse=True)

    print(f"{'#':>3}  {'LB':>3} {'Zin':>4} {'Zout':>4} {'Conf':>4} {'Stop':>4} {'Mom':>5} {'Hold':>4} {'Skip':>4}  "
          f"{'Net PnL':>12}  {'PF':>6}  {'WR%':>5}  {'Sharpe':>7}  {'MaxDD%':>6}  {'Trades':>6}  {'AvgW':>8}  {'AvgL':>8}")
    print("-" * 130)

    for i, r in enumerate(sharpe_list[:20], 1):
        pf = float(r["profit_factor"])
        pf_str = f"{pf:.2f}" if pf < 100 else "inf"
        trades = int(r["total_trades"])
        print(
            f"{i:>3}  {int(r['lookback_bars']):>3} {float(r['z_entry']):>4.1f} {float(r['z_exit']):>4.1f} "
            f"{int(r['confirm_bars']):>4} {float(r['stop_multiple']):>4.1f} {float(r['momentum_threshold']):>5.2f} "
            f"{int(r['min_hold_bars']):>4} {int(r['skip_first_minutes']):>4}  "
            f"${float(r['net_pnl']):>11,.0f}  {pf_str:>6}  {float(r['win_rate'])*100:>5.1f}  "
            f"{float(r['sharpe']):>7.2f}  {float(r['max_dd_pct'])*100:>5.1f}%  "
            f"{trades:>6}  ${float(r['avg_winner']):>7,.0f}  ${float(r['avg_loser']):>7,.0f}"
        )


if __name__ == "__main__":
    main()
