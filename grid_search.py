#!/usr/bin/env python3
"""Grid search over contrarian mean-reversion strategy parameters.

Runs all combinations of specified parameter ranges and outputs a sorted
results table to identify the best parameter sets.

Usage:
  python grid_search.py data/multi.dbn --instruments NQ ES RTY YM

Uses sensible default search ranges. Override any range via CLI flags.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

from engine.config import EngineConfig, SessionFilter, SlippageConfig, SlippageModel
from engine.contract_registry import ContractRegistry
from engine.multi_engine import MultiInstrumentEngine
from analytics.metrics import MetricsCalculator
from strategies.contrarian_reversion import ContraMeanReversionStrategy


@dataclass
class ParamSet:
    """One combination of strategy parameters."""

    window: int
    theta_multiplier: float
    stop_multiple: float
    target_fraction: float
    min_hold_bars: int
    skip_first_minutes: int


@dataclass
class SearchResult:
    """Results from one parameter combination."""

    params: ParamSet
    net_pnl: float
    profit_factor: float
    win_rate: float
    sharpe: float
    sortino: float
    calmar: float
    max_dd_pct: float
    max_dd_dollars: float
    total_trades: int
    avg_winner: float
    avg_loser: float
    expectancy: float
    best_day: float
    worst_day: float
    avg_hold_bars: float


def run_single(
    dbn_source,
    instruments: list[str],
    is_single_file: bool,
    params: ParamSet,
    capital: float,
    max_contracts: int,
    registry: ContractRegistry,
) -> SearchResult:
    """Run one backtest with a specific parameter set."""
    strategy = ContraMeanReversionStrategy(
        signal_window_minutes=params.window,
        theta=None,  # Always auto-calibrate
        theta_multiplier=params.theta_multiplier,
        stop_multiple=params.stop_multiple,
        target_fraction=params.target_fraction,
        max_position_pct=0.10,
        session_cutoff_minutes=15,
        min_hold_bars=params.min_hold_bars,
        skip_first_minutes=params.skip_first_minutes,
    )

    config = EngineConfig(
        instrument=instruments[0],
        initial_capital=capital,
        slippage=SlippageConfig(model=SlippageModel.FIXED, fixed_ticks=0.0),
        max_position_size=max_contracts,
        session_filter=SessionFilter.FULL_GLOBEX,
        enforce_daily_settlement=True,
        results_dir="results_grid",
    )

    engine = MultiInstrumentEngine(
        dbn_paths=dbn_source,
        strategy=strategy,
        config=config,
        instruments=instruments if is_single_file else None,
        registry=registry,
    )
    raw = engine.run()

    result = MetricsCalculator.compute(
        strategy_name=raw["strategy_name"],
        instrument="+".join(raw["instruments"]),
        initial_capital=raw["initial_capital"],
        final_equity=raw["final_equity"],
        realized_pnl=raw["realized_pnl"],
        unrealized_pnl=raw["unrealized_pnl"],
        total_commissions=raw["total_commissions"],
        total_slippage_dollars=raw["total_slippage_dollars"],
        trades=raw["all_trades"],
        equity_curve=raw["equity_curve"],
        total_margin_calls=raw["total_margin_calls"],
        total_forced_liquidations=raw["total_forced_liquidations"],
    )

    return SearchResult(
        params=params,
        net_pnl=result.net_pnl,
        profit_factor=result.profit_factor,
        win_rate=result.win_rate,
        sharpe=result.sharpe_ratio,
        sortino=result.sortino_ratio,
        calmar=result.calmar_ratio,
        max_dd_pct=result.max_drawdown_pct,
        max_dd_dollars=result.max_drawdown_dollars,
        total_trades=result.total_trades,
        avg_winner=result.avg_winner,
        avg_loser=result.avg_loser,
        expectancy=result.expectancy,
        best_day=result.best_day,
        worst_day=result.worst_day,
        avg_hold_bars=result.avg_hold_bars,
    )


def print_results_table(results: list[SearchResult]) -> None:
    """Print sorted results table."""
    # Sort by profit factor descending (filter out inf)
    results.sort(
        key=lambda r: r.profit_factor if r.profit_factor < 100 else -1,
        reverse=True,
    )

    header = (
        f"{'#':>3}  {'Win':>5}  {'ThtM':>4}  {'Stop':>4}  {'TgtF':>4}  "
        f"{'Hold':>4}  {'Skip':>4}  {'PF':>6}  {'WR%':>5}  {'Sharpe':>6}  "
        f"{'Sortino':>7}  {'NetPnL':>12}  {'MaxDD%':>6}  {'Trades':>6}  "
        f"{'AvgW':>8}  {'AvgL':>8}  {'Expect':>8}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print("  GRID SEARCH RESULTS (sorted by Profit Factor)")
    print(sep)
    print(header)
    print(sep)

    for i, r in enumerate(results, 1):
        p = r.params
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor < 100 else "inf"
        print(
            f"{i:>3}  {p.window:>5}  {p.theta_multiplier:>4.1f}  "
            f"{p.stop_multiple:>4.1f}  {p.target_fraction:>4.2f}  "
            f"{p.min_hold_bars:>4}  {p.skip_first_minutes:>4}  "
            f"{pf_str:>6}  {r.win_rate:>5.1f}  {r.sharpe:>6.2f}  "
            f"{r.sortino:>7.2f}  {r.net_pnl:>12,.0f}  {r.max_dd_pct:>5.1f}%  "
            f"{r.total_trades:>6}  {r.avg_winner:>8,.0f}  {r.avg_loser:>8,.0f}  "
            f"{r.expectancy:>8,.0f}"
        )

    print(sep)
    print(f"  Total combinations tested: {len(results)}")
    print(sep)


def save_results_csv(results: list[SearchResult], path: str) -> None:
    """Save results to CSV for further analysis."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "window", "theta_multiplier", "stop_multiple", "target_fraction",
            "min_hold_bars", "skip_first_minutes",
            "profit_factor", "win_rate", "sharpe", "sortino", "calmar",
            "net_pnl", "max_dd_pct", "max_dd_dollars", "total_trades",
            "avg_winner", "avg_loser", "expectancy", "avg_hold_bars",
            "best_day", "worst_day",
        ])
        for r in results:
            p = r.params
            writer.writerow([
                p.window, p.theta_multiplier, p.stop_multiple, p.target_fraction,
                p.min_hold_bars, p.skip_first_minutes,
                r.profit_factor, r.win_rate, r.sharpe, r.sortino, r.calmar,
                r.net_pnl, r.max_dd_pct, r.max_dd_dollars, r.total_trades,
                r.avg_winner, r.avg_loser, r.expectancy, r.avg_hold_bars,
                r.best_day, r.worst_day,
            ])
    print(f"\nResults saved to: {path}")


def _worker(args_tuple) -> SearchResult | str:
    """Worker function for multiprocessing. Returns SearchResult or error string."""
    dbn_source, instruments, params_dict, capital, max_contracts = args_tuple

    # Suppress logging in worker processes
    logging.disable(logging.CRITICAL)

    params = ParamSet(**params_dict)
    try:
        return run_single(
            dbn_source=dbn_source,
            instruments=instruments,
            is_single_file=True,
            params=params,
            capital=capital,
            max_contracts=max_contracts,
            registry=ContractRegistry(),
        )
    except Exception as e:
        return f"ERROR: {e}"


def parse_float_list(s: str) -> list[float]:
    """Parse comma-separated float values."""
    return [float(x.strip()) for x in s.split(",")]


def parse_int_list(s: str) -> list[int]:
    """Parse comma-separated int values."""
    return [int(x.strip()) for x in s.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search over contrarian strategy parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default search ranges:
  python grid_search.py data/multi.dbn --instruments NQ ES RTY YM

  # Custom parameter ranges (comma-separated values):
  python grid_search.py data/multi.dbn --instruments NQ ES RTY YM \\
      --windows 30,60,90 \\
      --theta-multipliers 1.0,1.5,2.0 \\
      --stop-multiples 2.0,3.0,4.0 \\
      --target-fractions 0.3,0.5 \\
      --min-hold 0,5,10 \\
      --skip-first 0,30,60
        """,
    )

    # Data
    parser.add_argument("dbn_file", help="Single .dbn file with all instruments")
    parser.add_argument(
        "--instruments", nargs="+", required=True,
        help="Instrument root symbols (e.g. NQ ES RTY YM)",
    )

    # Parameter ranges (comma-separated)
    parser.add_argument(
        "--windows", type=str, default="30,60,90,120",
        help="Signal window lengths in minutes (default: 30,60,90,120)",
    )
    parser.add_argument(
        "--theta-multipliers", type=str, default="1.0,1.5,2.0",
        help="Theta multipliers (default: 1.0,1.5,2.0)",
    )
    parser.add_argument(
        "--stop-multiples", type=str, default="2.0,3.0,4.0",
        help="Stop loss multiples (default: 2.0,3.0,4.0)",
    )
    parser.add_argument(
        "--target-fractions", type=str, default="0.3,0.5",
        help="Target profit fractions (default: 0.3,0.5)",
    )
    parser.add_argument(
        "--min-hold", type=str, default="0,5",
        help="Minimum hold bars (default: 0,5)",
    )
    parser.add_argument(
        "--skip-first", type=str, default="0,30",
        help="Skip first N minutes of session (default: 0,30)",
    )

    # Engine params
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--max-contracts", type=int, default=20)

    # Parallelism
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Number of parallel workers. 0=auto (all CPU cores), 1=sequential (default: 0)",
    )

    # Output
    parser.add_argument(
        "--output", type=str, default="results_grid/grid_search_results.csv",
        help="CSV output path (default: results_grid/grid_search_results.csv)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # Quiet logging for grid search — only show warnings
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # Parse parameter ranges
    windows = parse_int_list(args.windows)
    theta_mults = parse_float_list(args.theta_multipliers)
    stop_mults = parse_float_list(args.stop_multiples)
    target_fracs = parse_float_list(args.target_fractions)
    min_holds = parse_int_list(args.min_hold)
    skip_firsts = parse_int_list(args.skip_first)

    instruments = [s.upper() for s in args.instruments]

    # Build all combinations
    combos = list(itertools.product(
        windows, theta_mults, stop_mults, target_fracs, min_holds, skip_firsts
    ))
    total = len(combos)

    print("=" * 60)
    print("  GRID SEARCH — Contrarian Mean-Reversion")
    print("=" * 60)
    print(f"  Data: {args.dbn_file}")
    print(f"  Instruments: {instruments}")
    print(f"  Capital: ${args.capital:,.0f}")
    print()
    print(f"  Windows:           {windows}")
    print(f"  Theta multipliers: {theta_mults}")
    print(f"  Stop multiples:    {stop_mults}")
    print(f"  Target fractions:  {target_fracs}")
    print(f"  Min hold bars:     {min_holds}")
    print(f"  Skip first mins:   {skip_firsts}")
    print()
    print(f"  Total combinations: {total}")

    # Determine worker count
    num_workers = args.workers if args.workers > 0 else mp.cpu_count()
    sequential = num_workers == 1

    print(f"  Workers: {num_workers} {'(sequential)' if sequential else f'(parallel across {num_workers} CPU cores)'}")
    print("=" * 60)
    print()

    # Build worker arguments — use dicts since dataclasses may not pickle across processes
    worker_args = []
    for win, tm, sm, tf, mh, sf in combos:
        params_dict = dict(
            window=win,
            theta_multiplier=tm,
            stop_multiple=sm,
            target_fraction=tf,
            min_hold_bars=mh,
            skip_first_minutes=sf,
        )
        worker_args.append((
            args.dbn_file, instruments, params_dict,
            args.capital, args.max_contracts,
        ))

    results: list[SearchResult] = []
    start_time = time.time()

    if sequential:
        # Sequential mode — same as before, with progress
        registry = ContractRegistry()
        for i, (win, tm, sm, tf, mh, sf) in enumerate(combos, 1):
            params = ParamSet(
                window=win, theta_multiplier=tm, stop_multiple=sm,
                target_fraction=tf, min_hold_bars=mh, skip_first_minutes=sf,
            )
            elapsed = time.time() - start_time
            avg_per = elapsed / max(1, i - 1)
            remaining = avg_per * (total - i + 1)
            print(
                f"[{i}/{total}] win={win}, theta_m={tm:.1f}, stop={sm:.1f}, "
                f"tgt={tf:.2f}, hold={mh}, skip={sf}  "
                f"(~{remaining/60:.0f}m remaining)",
                end="", flush=True,
            )
            try:
                sr = run_single(
                    dbn_source=args.dbn_file, instruments=instruments,
                    is_single_file=True, params=params,
                    capital=args.capital, max_contracts=args.max_contracts,
                    registry=registry,
                )
                results.append(sr)
                pf = f"{sr.profit_factor:.2f}" if sr.profit_factor < 100 else "inf"
                print(f"  -> PF={pf}, WR={sr.win_rate:.1f}%, PnL=${sr.net_pnl:,.0f}, DD={sr.max_dd_pct:.1f}%")
            except Exception as e:
                print(f"  -> ERROR: {e}")
    else:
        # Parallel mode — use process pool
        completed = 0
        print(f"Launching {total} backtests across {num_workers} workers...\n")

        with mp.Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(_worker, worker_args):
                completed += 1
                elapsed = time.time() - start_time
                avg_per = elapsed / completed
                remaining = avg_per * (total - completed)

                if isinstance(result, str):
                    # Error
                    print(f"  [{completed}/{total}] {result}  (~{remaining/60:.0f}m remaining)")
                else:
                    results.append(result)
                    p = result.params
                    pf = f"{result.profit_factor:.2f}" if result.profit_factor < 100 else "inf"
                    print(
                        f"  [{completed}/{total}] win={p.window}, theta_m={p.theta_multiplier:.1f}, "
                        f"stop={p.stop_multiple:.1f}, tgt={p.target_fraction:.2f}, "
                        f"hold={p.min_hold_bars}, skip={p.skip_first_minutes}  "
                        f"-> PF={pf}, WR={result.win_rate:.1f}%, "
                        f"PnL=${result.net_pnl:,.0f}, DD={result.max_dd_pct:.1f}%  "
                        f"(~{remaining/60:.0f}m remaining)"
                    )

    total_time = time.time() - start_time
    print(f"\nGrid search complete in {total_time/60:.1f} minutes")

    # Print and save results
    if results:
        print_results_table(results)
        save_results_csv(results, args.output)


if __name__ == "__main__":
    main()
