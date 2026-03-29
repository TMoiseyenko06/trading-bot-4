#!/usr/bin/env python3
"""Grid search over V3 contrarian mean-reversion strategy parameters.

Focuses on the new V3 parameters while using optimal V2 base parameters
discovered from the 864-combination grid search:
  - lookback=60, z_entry=2.0, z_exit=0.5, confirm=2
  - stop=4.0, momentum=0.75, min_hold=3, skip_first=0

Usage:
  python grid_search_v3.py data/multi.dbn --instruments NQ ES RTY YM
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

from engine.config import EngineConfig, SessionFilter, SlippageConfig, SlippageModel
from engine.contract_registry import ContractRegistry
from engine.multi_engine import MultiInstrumentEngine
from analytics.metrics import MetricsCalculator
from strategies.contrarian_reversion_v3 import ContraMeanReversionV3


@dataclass
class V3ParamSet:
    """One combination of V3 strategy parameters."""
    # Base params (from V2 optimal)
    lookback_bars: int
    z_entry: float
    z_exit: float
    confirm_bars: int
    stop_multiple: float
    momentum_threshold: float
    min_hold_bars: int
    skip_first_minutes: int
    # V3 new params
    z_spread_threshold: float
    exit_tighten_bars: int
    exit_tighten_rate: float
    volume_spike_multiple: float
    per_leg_exit: bool
    time_weight_enabled: bool


@dataclass
class V3SearchResult:
    """Results from one V3 parameter combination."""
    params: V3ParamSet
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
    params: V3ParamSet,
    capital: float,
    max_contracts: int,
    registry: ContractRegistry,
) -> V3SearchResult:
    """Run one backtest with a specific V3 parameter set."""
    strategy = ContraMeanReversionV3(
        lookback_bars=params.lookback_bars,
        z_entry_threshold=params.z_entry,
        z_exit_threshold=params.z_exit,
        confirm_bars=params.confirm_bars,
        stop_multiple=params.stop_multiple,
        momentum_threshold=params.momentum_threshold,
        min_hold_bars=params.min_hold_bars,
        skip_first_minutes=params.skip_first_minutes,
        z_spread_threshold=params.z_spread_threshold,
        exit_tighten_bars=params.exit_tighten_bars,
        exit_tighten_rate=params.exit_tighten_rate,
        volume_spike_multiple=params.volume_spike_multiple,
        per_leg_exit=params.per_leg_exit,
        time_weight_enabled=params.time_weight_enabled,
    )

    config = EngineConfig(
        instrument=instruments[0],
        initial_capital=capital,
        slippage=SlippageConfig(model=SlippageModel.FIXED, fixed_ticks=0.0),
        max_position_size=max_contracts,
        session_filter=SessionFilter.FULL_GLOBEX,
        enforce_daily_settlement=True,
        results_dir="results_grid_v3",
    )

    engine = MultiInstrumentEngine(
        dbn_paths=dbn_source,
        strategy=strategy,
        config=config,
        instruments=instruments,
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

    return V3SearchResult(
        params=params,
        net_pnl=result.net_pnl,
        profit_factor=result.profit_factor,
        win_rate=result.win_rate,
        sharpe=result.sharpe_ratio,
        sortino=result.sortino_ratio,
        calmar=result.calmar_ratio,
        max_dd_pct=result.max_drawdown_pct,
        max_dd_dollars=result.max_drawdown_dollars,
        total_trades=result.num_trades,
        avg_winner=result.avg_winner,
        avg_loser=result.avg_loser,
        expectancy=result.expectancy,
        best_day=result.best_daily_pnl,
        worst_day=result.worst_daily_pnl,
        avg_hold_bars=result.avg_hold_time_bars,
    )


CSV_COLUMNS = [
    "lookback_bars", "z_entry", "z_exit", "confirm_bars",
    "stop_multiple", "momentum_threshold",
    "min_hold_bars", "skip_first_minutes",
    "z_spread_threshold", "exit_tighten_bars", "exit_tighten_rate",
    "volume_spike_multiple", "per_leg_exit", "time_weight_enabled",
    "profit_factor", "win_rate", "sharpe", "sortino", "calmar",
    "net_pnl", "max_dd_pct", "max_dd_dollars", "total_trades",
    "avg_winner", "avg_loser", "expectancy", "avg_hold_bars",
    "best_day", "worst_day",
]


def init_csv(path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(CSV_COLUMNS)


def append_result_csv(path: str, r: V3SearchResult) -> None:
    p = r.params
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            p.lookback_bars, p.z_entry, p.z_exit, p.confirm_bars,
            p.stop_multiple, p.momentum_threshold,
            p.min_hold_bars, p.skip_first_minutes,
            p.z_spread_threshold, p.exit_tighten_bars, p.exit_tighten_rate,
            p.volume_spike_multiple, p.per_leg_exit, p.time_weight_enabled,
            r.profit_factor, r.win_rate, r.sharpe, r.sortino, r.calmar,
            r.net_pnl, r.max_dd_pct, r.max_dd_dollars, r.total_trades,
            r.avg_winner, r.avg_loser, r.expectancy, r.avg_hold_bars,
            r.best_day, r.worst_day,
        ])


def _worker(args_tuple) -> V3SearchResult | str:
    dbn_source, instruments, params_dict, capital, max_contracts = args_tuple
    logging.disable(logging.CRITICAL)

    params = V3ParamSet(**params_dict)
    try:
        return run_single(
            dbn_source=dbn_source,
            instruments=instruments,
            params=params,
            capital=capital,
            max_contracts=max_contracts,
            registry=ContractRegistry(),
        )
    except Exception as e:
        return f"ERROR: {e}"


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",")]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V3 Grid search — tests new features with optimal V2 base params",
    )

    parser.add_argument("dbn_file", help="Single .dbn file with all instruments")
    parser.add_argument(
        "--instruments", nargs="+", required=True,
        help="Instrument root symbols (e.g. NQ ES RTY YM)",
    )

    # Base params — use V2 optimal values, allow override
    parser.add_argument("--lookbacks", type=str, default="60",
                        help="Base lookback bars (default: 60)")
    parser.add_argument("--z-entries", type=str, default="2.0",
                        help="Base z_entry (default: 2.0)")
    parser.add_argument("--z-exits", type=str, default="0.5",
                        help="Base z_exit (default: 0.5)")
    parser.add_argument("--confirm", type=str, default="2",
                        help="Base confirm bars (default: 2)")
    parser.add_argument("--stop-multiples", type=str, default="4.0",
                        help="Base stop multiple (default: 4.0)")
    parser.add_argument("--momentum-thresholds", type=str, default="0.75",
                        help="Base momentum threshold (default: 0.75)")
    parser.add_argument("--min-hold", type=str, default="3",
                        help="Base min hold bars (default: 3)")
    parser.add_argument("--skip-first", type=str, default="0",
                        help="Base skip first minutes (default: 0)")

    # V3 new parameter ranges
    parser.add_argument("--z-spreads", type=str, default="2.0,3.0,4.0",
                        help="Z-score spread thresholds (default: 2.0,3.0,4.0)")
    parser.add_argument("--exit-tighten-bars", type=str, default="15,25,40",
                        help="Bars before exit tightening starts (default: 15,25,40)")
    parser.add_argument("--exit-tighten-rates", type=str, default="0.01,0.02,0.04",
                        help="Exit tighten rate per bar (default: 0.01,0.02,0.04)")
    parser.add_argument("--volume-spikes", type=str, default="1.0,1.3,1.5",
                        help="Volume spike multiples (1.0=disabled) (default: 1.0,1.3,1.5)")
    parser.add_argument("--per-leg", type=str, default="True,False",
                        help="Per-leg exit mode (default: True,False)")
    parser.add_argument("--time-weight", type=str, default="True,False",
                        help="Time-of-day weighting (default: True,False)")

    # Engine params
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--max-contracts", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--output", type=str, default="results_grid_v3/grid_v3_results.csv",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # Parse parameter ranges
    lookbacks = parse_int_list(args.lookbacks)
    z_entries = parse_float_list(args.z_entries)
    z_exits = parse_float_list(args.z_exits)
    confirms = parse_int_list(args.confirm)
    stop_mults = parse_float_list(args.stop_multiples)
    mom_thresholds = parse_float_list(args.momentum_thresholds)
    min_holds = parse_int_list(args.min_hold)
    skip_firsts = parse_int_list(args.skip_first)

    z_spreads = parse_float_list(args.z_spreads)
    tighten_bars = parse_int_list(args.exit_tighten_bars)
    tighten_rates = parse_float_list(args.exit_tighten_rates)
    vol_spikes = parse_float_list(args.volume_spikes)
    per_legs = [x.strip().lower() == "true" for x in args.per_leg.split(",")]
    time_weights = [x.strip().lower() == "true" for x in args.time_weight.split(",")]

    instruments = [s.upper() for s in args.instruments]

    # Build all combinations
    combos = list(itertools.product(
        lookbacks, z_entries, z_exits, confirms,
        stop_mults, mom_thresholds, min_holds, skip_firsts,
        z_spreads, tighten_bars, tighten_rates, vol_spikes,
        per_legs, time_weights,
    ))
    total = len(combos)

    print("=" * 60)
    print("  GRID SEARCH v3 — Contrarian Mean-Reversion")
    print("=" * 60)
    print(f"  Data: {args.dbn_file}")
    print(f"  Instruments: {instruments}")
    print(f"  Capital: ${args.capital:,.0f}")
    print()
    print("  BASE PARAMS (from V2 optimal):")
    print(f"    Lookback bars:     {lookbacks}")
    print(f"    Z-entry:           {z_entries}")
    print(f"    Z-exit:            {z_exits}")
    print(f"    Confirm bars:      {confirms}")
    print(f"    Stop multiples:    {stop_mults}")
    print(f"    Momentum thresh:   {mom_thresholds}")
    print(f"    Min hold bars:     {min_holds}")
    print(f"    Skip first mins:   {skip_firsts}")
    print()
    print("  V3 NEW PARAMS:")
    print(f"    Z-spread thresh:   {z_spreads}")
    print(f"    Exit tighten bars: {list(tighten_bars)}")
    print(f"    Exit tighten rate: {tighten_rates}")
    print(f"    Volume spike mult: {vol_spikes}")
    print(f"    Per-leg exit:      {per_legs}")
    print(f"    Time weighting:    {time_weights}")
    print()
    print(f"  Total combinations: {total}")

    num_workers = args.workers if args.workers > 0 else mp.cpu_count()
    sequential = num_workers == 1

    print(f"  Workers: {num_workers} {'(sequential)' if sequential else f'(parallel)'}")
    print("=" * 60)
    print()

    # Build worker arguments
    worker_args = []
    for lb, ze, zx, cb, sm, mt, mh, sf, zsp, tb, tr, vs, pl, tw in combos:
        params_dict = dict(
            lookback_bars=lb, z_entry=ze, z_exit=zx, confirm_bars=cb,
            stop_multiple=sm, momentum_threshold=mt,
            min_hold_bars=mh, skip_first_minutes=sf,
            z_spread_threshold=zsp, exit_tighten_bars=tb,
            exit_tighten_rate=tr, volume_spike_multiple=vs,
            per_leg_exit=pl, time_weight_enabled=tw,
        )
        worker_args.append((
            args.dbn_file, instruments, params_dict,
            args.capital, args.max_contracts,
        ))

    csv_path = args.output
    init_csv(csv_path)
    completed_count = 0
    start_time = time.time()

    if sequential:
        registry = ContractRegistry()
        for i, combo in enumerate(combos, 1):
            lb, ze, zx, cb, sm, mt, mh, sf, zsp, tb, tr, vs, pl, tw = combo
            params = V3ParamSet(
                lookback_bars=lb, z_entry=ze, z_exit=zx, confirm_bars=cb,
                stop_multiple=sm, momentum_threshold=mt,
                min_hold_bars=mh, skip_first_minutes=sf,
                z_spread_threshold=zsp, exit_tighten_bars=tb,
                exit_tighten_rate=tr, volume_spike_multiple=vs,
                per_leg_exit=pl, time_weight_enabled=tw,
            )
            elapsed = time.time() - start_time
            avg_per = elapsed / max(1, i - 1)
            remaining = avg_per * (total - i + 1)
            print(
                f"[{i}/{total}] spread={zsp:.1f}, tighten={tb}/{tr:.3f}, "
                f"vol={vs:.1f}, leg={pl}, time={tw}  "
                f"(~{remaining/60:.0f}m remaining)",
                end="", flush=True,
            )
            try:
                sr = run_single(
                    dbn_source=args.dbn_file, instruments=instruments,
                    params=params, capital=args.capital,
                    max_contracts=args.max_contracts, registry=registry,
                )
                append_result_csv(csv_path, sr)
                completed_count += 1
                pf = f"{sr.profit_factor:.2f}" if sr.profit_factor < 100 else "inf"
                print(f"  -> PF={pf}, PnL=${sr.net_pnl:,.0f}, trades={sr.total_trades}")
            except Exception as e:
                print(f"  -> ERROR: {e}")
    else:
        completed = 0
        print(f"Launching {total} backtests across {num_workers} workers...\n")

        with mp.Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(_worker, worker_args):
                completed += 1
                elapsed = time.time() - start_time
                avg_per = elapsed / completed
                remaining = avg_per * (total - completed)

                if isinstance(result, str):
                    print(f"  [{completed}/{total}] {result}  (~{remaining/60:.0f}m remaining)")
                else:
                    append_result_csv(csv_path, result)
                    completed_count += 1
                    p = result.params
                    pf = f"{result.profit_factor:.2f}" if result.profit_factor < 100 else "inf"
                    print(
                        f"  [{completed}/{total}] spread={p.z_spread_threshold:.1f}, "
                        f"tighten={p.exit_tighten_bars}/{p.exit_tighten_rate:.3f}, "
                        f"vol={p.volume_spike_multiple:.1f}, leg={p.per_leg_exit}, "
                        f"time={p.time_weight_enabled}  "
                        f"-> PF={pf}, PnL=${result.net_pnl:,.0f}, "
                        f"trades={result.total_trades}  "
                        f"(~{remaining/60:.0f}m remaining)"
                    )

    total_time = time.time() - start_time
    print(f"\nV3 grid search complete in {total_time/60:.1f} minutes")
    print(f"Results saved to: {csv_path} ({completed_count} rows)")


if __name__ == "__main__":
    main()
