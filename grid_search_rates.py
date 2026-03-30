#!/usr/bin/env python3
"""Grid search for rates basket (ZN, ZT, ZB, ZF).

Focused parameter ranges tuned for Treasury futures — smaller moves,
tighter spreads, different mean-reversion dynamics than equities.

Usage:
  python grid_search_rates.py data/rates.dbn --instruments ZN ZT ZB --workers 80
  python grid_search_rates.py data/rates.dbn --instruments ZN ZT ZB --workers 80 --resume
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    mp.set_start_method("forkserver")
except (RuntimeError, ValueError):
    pass  # already set, or not available (Windows)

import logging
import time
from dataclasses import dataclass

from engine.config import EngineConfig, SessionFilter, SlippageConfig, SlippageModel
from engine.contract_registry import ContractRegistry
from engine.multi_engine import MultiInstrumentEngine
from analytics.metrics import MetricsCalculator
from strategies.contrarian_reversion_v3 import ContraMeanReversionV3


@dataclass
class ParamSet:
    lookback_bars: int
    z_entry: float
    z_exit: float
    confirm_bars: int
    stop_multiple: float
    momentum_threshold: float
    min_hold_bars: int
    skip_first_minutes: int
    z_spread_threshold: float
    volume_spike_multiple: float
    per_leg_exit: bool
    time_weight_enabled: bool
    require_z_widening: bool
    leg_stop_atr_multiple: float
    asymmetric_exit: bool
    vol_regime_filter: bool
    vol_regime_multiple: float


@dataclass
class SearchResult:
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


PARAM_COLUMNS = [
    "lookback_bars", "z_entry", "z_exit", "confirm_bars",
    "stop_multiple", "momentum_threshold",
    "min_hold_bars", "skip_first_minutes",
    "z_spread_threshold", "volume_spike_multiple",
    "per_leg_exit", "time_weight_enabled",
    "require_z_widening", "leg_stop_atr_multiple",
    "asymmetric_exit", "vol_regime_filter", "vol_regime_multiple",
]

CSV_COLUMNS = PARAM_COLUMNS + [
    "profit_factor", "win_rate", "sharpe", "sortino", "calmar",
    "net_pnl", "max_dd_pct", "max_dd_dollars", "total_trades",
    "avg_winner", "avg_loser", "expectancy", "avg_hold_bars",
    "best_day", "worst_day",
]


def _combo_key(row: dict) -> tuple:
    parts = []
    for col in PARAM_COLUMNS:
        val = row.get(col, "")
        try:
            parts.append(round(float(val), 6))
        except (ValueError, TypeError):
            parts.append(str(val).strip().lower())
    return tuple(parts)


def _params_to_key(p: ParamSet) -> tuple:
    return (
        round(float(p.lookback_bars), 6),
        round(p.z_entry, 6), round(p.z_exit, 6),
        round(float(p.confirm_bars), 6),
        round(p.stop_multiple, 6), round(p.momentum_threshold, 6),
        round(float(p.min_hold_bars), 6), round(float(p.skip_first_minutes), 6),
        round(p.z_spread_threshold, 6), round(p.volume_spike_multiple, 6),
        str(p.per_leg_exit).lower(), str(p.time_weight_enabled).lower(),
        str(p.require_z_widening).lower(), round(p.leg_stop_atr_multiple, 6),
        str(p.asymmetric_exit).lower(), str(p.vol_regime_filter).lower(),
        round(p.vol_regime_multiple, 6),
    )


def load_completed_keys(path: str) -> set[tuple]:
    completed = set()
    if not os.path.exists(path):
        return completed
    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and all(c in reader.fieldnames for c in PARAM_COLUMNS):
                for row in reader:
                    completed.add(_combo_key(row))
    except Exception as e:
        print(f"  Warning: Could not read existing CSV for resume: {e}")
    return completed


def init_csv(path: str, resume: bool = False) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    if resume and os.path.exists(path):
        return
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(CSV_COLUMNS)


def append_result_csv(path: str, r: SearchResult) -> None:
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


def run_single(dbn_source, instruments, params, capital, max_contracts, registry):
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
        results_dir="results_grid_rates",
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
        total_trades=result.num_trades,
        avg_winner=result.avg_winner,
        avg_loser=result.avg_loser,
        expectancy=result.expectancy,
        best_day=result.best_daily_pnl,
        worst_day=result.worst_daily_pnl,
        avg_hold_bars=result.avg_hold_time_bars,
    )


def _worker(args_tuple):
    dbn_source, instruments, params_dict, capital, max_contracts = args_tuple
    logging.disable(logging.CRITICAL)
    params = ParamSet(**params_dict)
    try:
        return run_single(dbn_source, instruments, params, capital, max_contracts, ContractRegistry())
    except Exception as e:
        return f"ERROR: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rates basket grid search")

    parser.add_argument("dbn_file", help=".dbn file")
    parser.add_argument("--instruments", nargs="+", required=True)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--max-contracts", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=str, default="results_grid_rates/grid_rates_results.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    instruments = [s.upper() for s in args.instruments]

    # ================================================================
    # RATES-TUNED PARAMETER RANGES — 2,048 combos
    # ================================================================
    lookbacks = [30, 60]
    z_entries = [1.5, 2.0]
    z_exits = [0.3, 0.5]
    confirms = [2]
    stop_mults = [3.0, 4.0]
    mom_thresholds = [0.65, 0.75]
    min_holds = [3]
    skip_firsts = [0]
    z_spreads = [1.0, 2.0]
    vol_spikes = [1.0, 1.3]
    per_legs = [True, False]
    time_weights = [False]
    z_widenings = [True, False]
    leg_stop_atrs = [3.0]
    asymmetrics = [True, False]
    vol_regimes = [False]
    vol_regime_mults = [1.5]

    combos = list(itertools.product(
        lookbacks, z_entries, z_exits, confirms,
        stop_mults, mom_thresholds, min_holds, skip_firsts,
        z_spreads, vol_spikes, per_legs, time_weights,
        z_widenings, leg_stop_atrs, asymmetrics, vol_regimes, vol_regime_mults,
    ))
    total = len(combos)

    num_workers = args.workers if args.workers > 0 else mp.cpu_count()

    print("=" * 70)
    print("  RATES BASKET GRID SEARCH")
    print("=" * 70)
    print(f"  Data: {args.dbn_file}")
    print(f"  Instruments: {instruments}")
    print(f"  Capital: ${args.capital:,.0f}")
    print(f"  Total combinations: {total:,}")
    print(f"  Workers: {num_workers}")
    print(f"  Resume: {args.resume}")
    print("=" * 70)
    print()

    csv_path = args.output

    already_done = set()
    if args.resume:
        already_done = load_completed_keys(csv_path)
        if already_done:
            print(f"  RESUME: Found {len(already_done):,} completed")

    worker_args = []
    skipped = 0
    for combo in combos:
        (lb, ze, zx, cb, sm, mt, mh, sf,
         zsp, vs, pl, tw, zw, lsa, asym, vrf, vrm) = combo

        if args.resume and already_done:
            p = ParamSet(
                lookback_bars=lb, z_entry=ze, z_exit=zx, confirm_bars=cb,
                stop_multiple=sm, momentum_threshold=mt,
                min_hold_bars=mh, skip_first_minutes=sf,
                z_spread_threshold=zsp, volume_spike_multiple=vs,
                per_leg_exit=pl, time_weight_enabled=tw,
                require_z_widening=zw, leg_stop_atr_multiple=lsa,
                asymmetric_exit=asym, vol_regime_filter=vrf,
                vol_regime_multiple=vrm,
            )
            if _params_to_key(p) in already_done:
                skipped += 1
                continue

        worker_args.append((
            args.dbn_file, instruments,
            dict(
                lookback_bars=lb, z_entry=ze, z_exit=zx, confirm_bars=cb,
                stop_multiple=sm, momentum_threshold=mt,
                min_hold_bars=mh, skip_first_minutes=sf,
                z_spread_threshold=zsp, volume_spike_multiple=vs,
                per_leg_exit=pl, time_weight_enabled=tw,
                require_z_widening=zw, leg_stop_atr_multiple=lsa,
                asymmetric_exit=asym, vol_regime_filter=vrf,
                vol_regime_multiple=vrm,
            ),
            args.capital, args.max_contracts,
        ))

    remaining = len(worker_args)
    if skipped:
        print(f"  Skipping {skipped:,} done, running {remaining:,} remaining\n")

    init_csv(csv_path, resume=args.resume)
    completed_count = len(already_done) if args.resume else 0
    error_count = 0
    start_time = time.time()

    if remaining == 0:
        print("  All done! Nothing to run.")
    elif num_workers == 1:
        registry = ContractRegistry()
        for i, wa in enumerate(worker_args, 1):
            _, _, pd, cap, maxc = wa
            params = ParamSet(**pd)
            elapsed = time.time() - start_time
            eta = (elapsed / max(1, i - 1)) * (remaining - i + 1)
            print(f"[{i}/{remaining}] ze={params.z_entry} sp={params.z_spread_threshold} "
                  f"vs={params.volume_spike_multiple} leg={params.per_leg_exit} "
                  f"widen={params.require_z_widening} asym={params.asymmetric_exit}",
                  end="", flush=True)
            try:
                sr = run_single(args.dbn_file, instruments, params, cap, maxc, registry)
                append_result_csv(csv_path, sr)
                completed_count += 1
                pf = f"{sr.profit_factor:.2f}" if sr.profit_factor < 100 else "inf"
                print(f"  -> PF={pf} WR={sr.win_rate*100:.1f}% PnL=${sr.net_pnl:,.0f} T={sr.total_trades}")
            except Exception as e:
                error_count += 1
                print(f"  -> ERROR: {e}")
    else:
        done = 0
        print(f"Launching {remaining:,} backtests across {num_workers} workers...\n")
        with mp.Pool(processes=num_workers, maxtasksperchild=5) as pool:
            for result in pool.imap_unordered(_worker, worker_args):
                done += 1
                elapsed = time.time() - start_time
                eta = (elapsed / done) * (remaining - done)

                if isinstance(result, str):
                    error_count += 1
                    if done % 100 == 0 or done <= 5:
                        print(f"  [{done}/{remaining}] {result}")
                else:
                    append_result_csv(csv_path, result)
                    completed_count += 1
                    p = result.params
                    pf = f"{result.profit_factor:.2f}" if result.profit_factor < 100 else "inf"

                    if done <= 10 or done % 100 == 0 or result.profit_factor > 1.3:
                        print(
                            f"  [{done}/{remaining}] "
                            f"lb={p.lookback_bars} ze={p.z_entry} sp={p.z_spread_threshold} "
                            f"vs={p.volume_spike_multiple} leg={p.per_leg_exit} "
                            f"widen={p.require_z_widening} asym={p.asymmetric_exit}  "
                            f"-> PF={pf} WR={result.win_rate*100:.1f}% "
                            f"PnL=${result.net_pnl:,.0f} T={result.total_trades}  "
                            f"(~{eta/60:.0f}m left)"
                        )

    total_time = time.time() - start_time
    print()
    print("=" * 70)
    print(f"  Done in {total_time/60:.1f} min")
    print(f"  Results: {csv_path}")
    print(f"  Completed: {completed_count:,} / {total:,} ({error_count} errors)")
    print("=" * 70)


if __name__ == "__main__":
    main()
