#!/usr/bin/env python3
"""Comprehensive grid search over ALL V3 strategy parameters.

Searches across base V2 params + V3 features + V3.1 win rate improvements.
Designed for high core count machines (80+ cores).

Usage:
  python grid_search_v3.py data/multi.dbn --instruments NQ ES RTY YM --workers 80
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass

from engine.config import EngineConfig, SessionFilter, SlippageConfig, SlippageModel
from engine.contract_registry import ContractRegistry
from engine.multi_engine import MultiInstrumentEngine
from analytics.metrics import MetricsCalculator
from strategies.contrarian_reversion_v3 import ContraMeanReversionV3


@dataclass
class V3ParamSet:
    """One combination of all strategy parameters."""
    # Base params
    lookback_bars: int
    z_entry: float
    z_exit: float
    confirm_bars: int
    stop_multiple: float
    momentum_threshold: float
    min_hold_bars: int
    skip_first_minutes: int
    # V3 params
    z_spread_threshold: float
    volume_spike_multiple: float
    per_leg_exit: bool
    time_weight_enabled: bool
    # V3.1 win rate params
    require_z_widening: bool
    leg_stop_atr_multiple: float
    asymmetric_exit: bool
    vol_regime_filter: bool
    vol_regime_multiple: float


@dataclass
class V3SearchResult:
    """Results from one parameter combination."""
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
    """Run one backtest with a specific parameter set."""
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
        volume_spike_multiple=params.volume_spike_multiple,
        per_leg_exit=params.per_leg_exit,
        time_weight_enabled=params.time_weight_enabled,
        require_z_widening=params.require_z_widening,
        leg_stop_atr_multiple=params.leg_stop_atr_multiple,
        asymmetric_exit=params.asymmetric_exit,
        vol_regime_filter=params.vol_regime_filter,
        vol_regime_multiple=params.vol_regime_multiple,
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
    "z_spread_threshold", "volume_spike_multiple",
    "per_leg_exit", "time_weight_enabled",
    "require_z_widening", "leg_stop_atr_multiple",
    "asymmetric_exit", "vol_regime_filter", "vol_regime_multiple",
    "profit_factor", "win_rate", "sharpe", "sortino", "calmar",
    "net_pnl", "max_dd_pct", "max_dd_dollars", "total_trades",
    "avg_winner", "avg_loser", "expectancy", "avg_hold_bars",
    "best_day", "worst_day",
]


PARAM_COLUMNS = [
    "lookback_bars", "z_entry", "z_exit", "confirm_bars",
    "stop_multiple", "momentum_threshold",
    "min_hold_bars", "skip_first_minutes",
    "z_spread_threshold", "volume_spike_multiple",
    "per_leg_exit", "time_weight_enabled",
    "require_z_widening", "leg_stop_atr_multiple",
    "asymmetric_exit", "vol_regime_filter", "vol_regime_multiple",
]


def _combo_key(row: dict) -> tuple:
    """Create a hashable key from parameter columns for dedup/resume."""
    parts = []
    for col in PARAM_COLUMNS:
        val = row.get(col, "")
        # Normalize: round floats, lowercase bools
        try:
            fval = float(val)
            parts.append(round(fval, 6))
        except (ValueError, TypeError):
            parts.append(str(val).strip().lower())
    return tuple(parts)


def _params_to_key(p: V3ParamSet) -> tuple:
    """Create a hashable key from a V3ParamSet."""
    return (
        round(float(p.lookback_bars), 6),
        round(p.z_entry, 6),
        round(p.z_exit, 6),
        round(float(p.confirm_bars), 6),
        round(p.stop_multiple, 6),
        round(p.momentum_threshold, 6),
        round(float(p.min_hold_bars), 6),
        round(float(p.skip_first_minutes), 6),
        round(p.z_spread_threshold, 6),
        round(p.volume_spike_multiple, 6),
        str(p.per_leg_exit).lower(),
        str(p.time_weight_enabled).lower(),
        str(p.require_z_widening).lower(),
        round(p.leg_stop_atr_multiple, 6),
        str(p.asymmetric_exit).lower(),
        str(p.vol_regime_filter).lower(),
        round(p.vol_regime_multiple, 6),
    )


def load_completed_keys(path: str) -> set[tuple]:
    """Load parameter keys of already-completed runs from existing CSV."""
    completed = set()
    if not os.path.exists(path):
        return completed

    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            # Verify the CSV has the right columns
            if reader.fieldnames and all(c in reader.fieldnames for c in PARAM_COLUMNS):
                for row in reader:
                    completed.add(_combo_key(row))
    except Exception as e:
        print(f"  Warning: Could not read existing CSV for resume: {e}")
        return set()

    return completed


def init_csv(path: str, resume: bool = False) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    if resume and os.path.exists(path):
        # Don't overwrite — we'll append
        return
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(CSV_COLUMNS)


def append_result_csv(path: str, r: V3SearchResult) -> None:
    p = r.params
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            p.lookback_bars, p.z_entry, p.z_exit, p.confirm_bars,
            p.stop_multiple, p.momentum_threshold,
            p.min_hold_bars, p.skip_first_minutes,
            p.z_spread_threshold, p.volume_spike_multiple,
            p.per_leg_exit, p.time_weight_enabled,
            p.require_z_widening, p.leg_stop_atr_multiple,
            p.asymmetric_exit, p.vol_regime_filter, p.vol_regime_multiple,
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


def parse_bool_list(s: str) -> list[bool]:
    return [x.strip().lower() == "true" for x in s.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive V3 grid search — all parameters",
    )

    parser.add_argument("dbn_file", help="Single .dbn file with all instruments")
    parser.add_argument(
        "--instruments", nargs="+", required=True,
        help="Instrument root symbols (e.g. NQ ES RTY YM)",
    )

    # === BASE PARAMS (full V2 ranges) ===
    parser.add_argument("--lookbacks", type=str, default="30,60,90",
                        help="Lookback bars (default: 30,60,90)")
    parser.add_argument("--z-entries", type=str, default="1.5,2.0,2.5",
                        help="Z-entry thresholds (default: 1.5,2.0,2.5)")
    parser.add_argument("--z-exits", type=str, default="0.3,0.5",
                        help="Z-exit thresholds (default: 0.3,0.5)")
    parser.add_argument("--confirm", type=str, default="2,3",
                        help="Confirmation bars (default: 2,3)")
    parser.add_argument("--stop-multiples", type=str, default="2.0,3.0,4.0",
                        help="Stop multiples (default: 2.0,3.0,4.0)")
    parser.add_argument("--momentum-thresholds", type=str, default="0.65,0.75",
                        help="Momentum thresholds (default: 0.65,0.75)")
    parser.add_argument("--min-hold", type=str, default="3,5",
                        help="Min hold bars (default: 3,5)")
    parser.add_argument("--skip-first", type=str, default="0,30",
                        help="Skip first N minutes (default: 0,30)")

    # === V3 PARAMS ===
    parser.add_argument("--z-spreads", type=str, default="2.0,3.0",
                        help="Z-spread thresholds (default: 2.0,3.0)")
    parser.add_argument("--volume-spikes", type=str, default="1.0,1.3",
                        help="Volume spike multiples (default: 1.0,1.3)")
    parser.add_argument("--per-leg", type=str, default="True,False",
                        help="Per-leg exit (default: True,False)")
    parser.add_argument("--time-weight", type=str, default="True,False",
                        help="Time weighting (default: True,False)")

    # === V3.1 WIN RATE PARAMS ===
    parser.add_argument("--z-widening", type=str, default="True,False",
                        help="Require z-score widening (default: True,False)")
    parser.add_argument("--leg-stop-atr", type=str, default="2.0,3.0",
                        help="Per-leg ATR stop multiple (default: 2.0,3.0)")
    parser.add_argument("--asymmetric", type=str, default="True,False",
                        help="Asymmetric exit (default: True,False)")
    parser.add_argument("--vol-regime", type=str, default="True,False",
                        help="Vol regime filter (default: True,False)")
    parser.add_argument("--vol-regime-mult", type=str, default="1.5",
                        help="Vol regime multiple (default: 1.5)")

    # === ENGINE ===
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--max-contracts", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0=auto, default: 0)")
    parser.add_argument("--output", type=str, default="results_grid_v3/grid_v3_full_results.csv")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing CSV — skip already-completed combos")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # Parse all ranges
    lookbacks = parse_int_list(args.lookbacks)
    z_entries = parse_float_list(args.z_entries)
    z_exits = parse_float_list(args.z_exits)
    confirms = parse_int_list(args.confirm)
    stop_mults = parse_float_list(args.stop_multiples)
    mom_thresholds = parse_float_list(args.momentum_thresholds)
    min_holds = parse_int_list(args.min_hold)
    skip_firsts = parse_int_list(args.skip_first)

    z_spreads = parse_float_list(args.z_spreads)
    vol_spikes = parse_float_list(args.volume_spikes)
    per_legs = parse_bool_list(args.per_leg)
    time_weights = parse_bool_list(args.time_weight)

    z_widenings = parse_bool_list(args.z_widening)
    leg_stop_atrs = parse_float_list(args.leg_stop_atr)
    asymmetrics = parse_bool_list(args.asymmetric)
    vol_regimes = parse_bool_list(args.vol_regime)
    vol_regime_mults = parse_float_list(args.vol_regime_mult)

    instruments = [s.upper() for s in args.instruments]

    # Build all combinations
    combos = list(itertools.product(
        lookbacks, z_entries, z_exits, confirms,
        stop_mults, mom_thresholds, min_holds, skip_firsts,
        z_spreads, vol_spikes, per_legs, time_weights,
        z_widenings, leg_stop_atrs, asymmetrics, vol_regimes, vol_regime_mults,
    ))
    total = len(combos)

    num_workers = args.workers if args.workers > 0 else mp.cpu_count()
    est_seconds_per_run = 10
    est_total_seconds = total * est_seconds_per_run / num_workers
    est_hours = est_total_seconds / 3600

    print("=" * 70)
    print("  COMPREHENSIVE GRID SEARCH V3 — All Parameters")
    print("=" * 70)
    print(f"  Data: {args.dbn_file}")
    print(f"  Instruments: {instruments}")
    print(f"  Capital: ${args.capital:,.0f}")
    print()
    print("  BASE PARAMS:")
    print(f"    Lookback:    {lookbacks}")
    print(f"    Z-entry:     {z_entries}")
    print(f"    Z-exit:      {z_exits}")
    print(f"    Confirm:     {confirms}")
    print(f"    Stop mult:   {stop_mults}")
    print(f"    Momentum:    {mom_thresholds}")
    print(f"    Min hold:    {min_holds}")
    print(f"    Skip first:  {skip_firsts}")
    print()
    print("  V3 PARAMS:")
    print(f"    Z-spread:    {z_spreads}")
    print(f"    Vol spike:   {vol_spikes}")
    print(f"    Per-leg:     {per_legs}")
    print(f"    Time weight: {time_weights}")
    print()
    print("  V3.1 WIN RATE PARAMS:")
    print(f"    Z-widening:  {z_widenings}")
    print(f"    Leg ATR stop:{leg_stop_atrs}")
    print(f"    Asymmetric:  {asymmetrics}")
    print(f"    Vol regime:  {vol_regimes}")
    print(f"    Vol reg mult:{vol_regime_mults}")
    print()
    print(f"  Total combinations: {total:,}")
    print(f"  Workers: {num_workers}")
    print(f"  Resume mode: {args.resume}")
    print(f"  Estimated time: ~{est_hours:.1f} hours ({est_total_seconds/60:.0f} min)")
    print("=" * 70)
    print()

    csv_path = args.output

    # --- Resume support: load already-completed combos ---
    already_done = set()
    if args.resume:
        already_done = load_completed_keys(csv_path)
        if already_done:
            print(f"  RESUME: Found {len(already_done):,} completed combos in {csv_path}")
        else:
            print(f"  RESUME: No existing results found, starting fresh")

    # Build worker arguments, skipping already-completed combos
    worker_args = []
    skipped = 0
    for combo in combos:
        (lb, ze, zx, cb, sm, mt, mh, sf,
         zsp, vs, pl, tw,
         zw, lsa, asym, vrf, vrm) = combo

        # Check if this combo was already completed
        if args.resume and already_done:
            params_obj = V3ParamSet(
                lookback_bars=lb, z_entry=ze, z_exit=zx, confirm_bars=cb,
                stop_multiple=sm, momentum_threshold=mt,
                min_hold_bars=mh, skip_first_minutes=sf,
                z_spread_threshold=zsp, volume_spike_multiple=vs,
                per_leg_exit=pl, time_weight_enabled=tw,
                require_z_widening=zw, leg_stop_atr_multiple=lsa,
                asymmetric_exit=asym, vol_regime_filter=vrf,
                vol_regime_multiple=vrm,
            )
            if _params_to_key(params_obj) in already_done:
                skipped += 1
                continue

        params_dict = dict(
            lookback_bars=lb, z_entry=ze, z_exit=zx, confirm_bars=cb,
            stop_multiple=sm, momentum_threshold=mt,
            min_hold_bars=mh, skip_first_minutes=sf,
            z_spread_threshold=zsp, volume_spike_multiple=vs,
            per_leg_exit=pl, time_weight_enabled=tw,
            require_z_widening=zw, leg_stop_atr_multiple=lsa,
            asymmetric_exit=asym, vol_regime_filter=vrf,
            vol_regime_multiple=vrm,
        )
        worker_args.append((
            args.dbn_file, instruments, params_dict,
            args.capital, args.max_contracts,
        ))

    remaining_total = len(worker_args)
    if skipped > 0:
        print(f"  RESUME: Skipping {skipped:,} already-completed, running {remaining_total:,} remaining")
        est_remaining_hours = remaining_total * est_seconds_per_run / num_workers / 3600
        print(f"  Revised estimate: ~{est_remaining_hours:.1f} hours")
        print()

    init_csv(csv_path, resume=args.resume)
    completed_count = len(already_done) if args.resume else 0
    error_count = 0
    start_time = time.time()

    if remaining_total == 0:
        print("  All combinations already completed! Nothing to do.")
        print(f"  Results at: {csv_path}")
    elif num_workers == 1:
        # Sequential mode
        registry = ContractRegistry()
        for i, wa in enumerate(worker_args, 1):
            _, _, params_dict, cap, maxc = wa
            params = V3ParamSet(**params_dict)
            p = params
            elapsed = time.time() - start_time
            avg_per = elapsed / max(1, i - 1)
            eta = avg_per * (remaining_total - i + 1)
            print(
                f"[{i:,}/{remaining_total:,}] lb={p.lookback_bars} ze={p.z_entry} "
                f"st={p.stop_multiple} sp={p.z_spread_threshold} "
                f"vs={p.volume_spike_multiple} leg={p.per_leg_exit} "
                f"widen={p.require_z_widening} asym={p.asymmetric_exit} "
                f"vr={p.vol_regime_filter}  (~{eta/60:.0f}m left)",
                end="", flush=True,
            )
            try:
                sr = run_single(
                    dbn_source=args.dbn_file, instruments=instruments,
                    params=params, capital=cap,
                    max_contracts=maxc, registry=registry,
                )
                append_result_csv(csv_path, sr)
                completed_count += 1
                pf = f"{sr.profit_factor:.2f}" if sr.profit_factor < 100 else "inf"
                print(f"  -> PF={pf} WR={sr.win_rate*100:.1f}% PnL=${sr.net_pnl:,.0f} T={sr.total_trades}")
            except Exception as e:
                error_count += 1
                print(f"  -> ERROR: {e}")
    else:
        # Parallel mode
        completed = 0
        print(f"Launching {remaining_total:,} backtests across {num_workers} workers...\n")

        # maxtasksperchild: recycle workers after N tasks to prevent RAM buildup.
        # Each worker loads the .dbn file + builds data structures that Python
        # won't free back to OS. Without recycling, 80 workers * ~11 tasks each
        # = ~900 tasks before OOM. Recycling adds ~1s overhead per restart.
        with mp.Pool(processes=num_workers, maxtasksperchild=5) as pool:
            for result in pool.imap_unordered(_worker, worker_args):
                completed += 1
                elapsed = time.time() - start_time
                avg_per = elapsed / completed
                eta = avg_per * (remaining_total - completed)

                if isinstance(result, str):
                    error_count += 1
                    if completed % 100 == 0 or completed <= 10:
                        print(f"  [{completed:,}/{remaining_total:,}] {result}  (~{eta/60:.0f}m left)")
                else:
                    append_result_csv(csv_path, result)
                    completed_count += 1
                    p = result.params
                    pf = f"{result.profit_factor:.2f}" if result.profit_factor < 100 else "inf"

                    # Print every 200th, first 10, or notable results
                    if completed <= 10 or completed % 200 == 0 or result.profit_factor > 1.3:
                        print(
                            f"  [{completed:,}/{remaining_total:,}] "
                            f"lb={p.lookback_bars} ze={p.z_entry} st={p.stop_multiple} "
                            f"sp={p.z_spread_threshold} vs={p.volume_spike_multiple} "
                            f"leg={p.per_leg_exit} widen={p.require_z_widening} "
                            f"asym={p.asymmetric_exit} vr={p.vol_regime_filter}  "
                            f"-> PF={pf} WR={result.win_rate*100:.1f}% "
                            f"PnL=${result.net_pnl:,.0f} T={result.total_trades}  "
                            f"(~{eta/60:.0f}m left)"
                        )

    total_time = time.time() - start_time
    print()
    print("=" * 70)
    print(f"  Grid search complete in {total_time/3600:.1f} hours ({total_time/60:.0f} min)")
    print(f"  Results saved: {csv_path}")
    print(f"  Completed: {completed_count:,} / {total:,} ({error_count} errors)")
    print("=" * 70)


if __name__ == "__main__":
    main()
