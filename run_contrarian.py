#!/usr/bin/env python3
"""CLI entry point for the contrarian mean-reversion basket strategy.

Supports two data modes:
  1. Single .dbn file containing all instruments:
     python run_contrarian.py data/all_equities.dbn --instruments NQ ES RTY YM

  2. Separate .dbn files per instrument:
     python run_contrarian.py --NQ data/NQ.dbn --ES data/ES.dbn --RTY data/RTY.dbn --YM data/YM.dbn
"""

from __future__ import annotations

import argparse
import logging
import sys

from engine.config import EngineConfig, SessionFilter, SlippageConfig, SlippageModel
from engine.contract_registry import ContractRegistry
from engine.multi_engine import MultiInstrumentEngine
from analytics.metrics import MetricsCalculator
from analytics.report import ReportGenerator
from strategies.contrarian_reversion import ContraMeanReversionStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Contrarian Mean-Reversion Basket Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single .dbn file with all instruments:
  python run_contrarian.py data/equities.dbn --instruments NQ ES RTY YM

  # Single file, only 3 instruments:
  python run_contrarian.py data/equities.dbn --instruments NQ ES RTY

  # Separate files per instrument:
  python run_contrarian.py --NQ data/NQ.dbn --ES data/ES.dbn --RTY data/RTY.dbn --YM data/YM.dbn

  # With custom parameters:
  python run_contrarian.py data/equities.dbn --instruments NQ ES RTY YM --theta 0.002 --window 15
        """,
    )

    # Single-file mode: positional arg + --instruments
    parser.add_argument(
        "dbn_file",
        nargs="?",
        default=None,
        help="Single .dbn file containing all instruments",
    )
    parser.add_argument(
        "--instruments",
        nargs="+",
        default=None,
        help="Instrument root symbols in the .dbn file (e.g. NQ ES RTY YM)",
    )

    # Multi-file mode: per-instrument flags
    parser.add_argument("--NQ", help="Path to NQ .dbn file (multi-file mode)")
    parser.add_argument("--ES", help="Path to ES .dbn file (multi-file mode)")
    parser.add_argument("--RTY", help="Path to RTY .dbn file (multi-file mode)")
    parser.add_argument("--YM", help="Path to YM .dbn file (multi-file mode)")

    # Strategy parameters
    parser.add_argument(
        "--window", type=int, default=60,
        help="Signal window length in minutes (default: 60)",
    )
    parser.add_argument(
        "--theta", type=float, default=None,
        help="Deviation threshold. Default: auto-calibrate",
    )
    parser.add_argument(
        "--theta-multiplier", type=float, default=1.5,
        help="Multiply auto-calibrated theta (default: 1.5)",
    )
    parser.add_argument(
        "--stop-multiple", type=float, default=3.0,
        help="Stop loss multiple (default: 3.0)",
    )
    parser.add_argument(
        "--target-fraction", type=float, default=0.4,
        help="Take profit fraction of theta (default: 0.4)",
    )
    parser.add_argument(
        "--max-position-pct", type=float, default=0.10,
        help="Max position per instrument as %% of capital (default: 0.10)",
    )
    parser.add_argument(
        "--cutoff-minutes", type=int, default=15,
        help="Minutes before session close to flatten (default: 15)",
    )
    parser.add_argument(
        "--min-hold-bars", type=int, default=5,
        help="Minimum bars to hold before allowing exit (default: 5)",
    )
    parser.add_argument(
        "--skip-first-minutes", type=int, default=30,
        help="Skip first N minutes of RTH session (default: 30)",
    )
    parser.add_argument(
        "--lookback", type=int, default=60,
        help="Rolling lookback window in bars for z-score (default: 60)",
    )
    parser.add_argument(
        "--z-entry", type=float, default=2.0,
        help="Z-score threshold for entry (default: 2.0)",
    )
    parser.add_argument(
        "--z-exit", type=float, default=0.5,
        help="Z-score threshold for exit (default: 0.5)",
    )
    parser.add_argument(
        "--confirm-bars", type=int, default=3,
        help="Confirmation bars before entry (default: 3)",
    )
    parser.add_argument(
        "--momentum-window", type=int, default=20,
        help="Bars for momentum regime filter (default: 20)",
    )
    parser.add_argument(
        "--momentum-threshold", type=float, default=0.7,
        help="Fraction threshold for trending detection (default: 0.7)",
    )

    # Engine parameters
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Initial capital (default: 100000)",
    )
    parser.add_argument(
        "--max-contracts",
        type=int,
        default=20,
        help="Max contracts per instrument (default: 20)",
    )
    parser.add_argument(
        "--slippage-ticks",
        type=float,
        default=0.0,
        help="Slippage in ticks (default: 0.0)",
    )

    # Output
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Output directory (default: results)",
    )
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # Determine data mode
    single_file = args.dbn_file is not None
    multi_file_paths: dict[str, str] = {}
    for sym in ["NQ", "ES", "RTY", "YM"]:
        path = getattr(args, sym, None)
        if path:
            multi_file_paths[sym] = path

    if single_file and multi_file_paths:
        print(
            "ERROR: Use either a single .dbn file (positional arg + --instruments) "
            "OR per-instrument flags (--NQ, --ES, etc.), not both."
        )
        sys.exit(1)

    if single_file:
        if not args.instruments or len(args.instruments) < 3:
            print(
                "ERROR: When using a single .dbn file, provide at least 3 "
                "instruments via --instruments (e.g. --instruments NQ ES RTY YM)"
            )
            sys.exit(1)
        instrument_list = [s.upper() for s in args.instruments]
        dbn_source = args.dbn_file
    elif multi_file_paths:
        if len(multi_file_paths) < 3:
            print(
                "ERROR: At least 3 instruments required. Provide --NQ, --ES, "
                "--RTY, and/or --YM with .dbn file paths."
            )
            sys.exit(1)
        instrument_list = list(multi_file_paths.keys())
        dbn_source = multi_file_paths
    else:
        print(
            "ERROR: Provide data as either:\n"
            "  Single file:  python run_contrarian.py data/equities.dbn "
            "--instruments NQ ES RTY YM\n"
            "  Multi file:   python run_contrarian.py --NQ nq.dbn --ES es.dbn "
            "--RTY rty.dbn --YM ym.dbn"
        )
        sys.exit(1)

    # Show instrument info
    registry = ContractRegistry()
    print("=" * 60)
    print("  Contrarian Mean-Reversion Basket Strategy")
    print("=" * 60)
    if single_file:
        print(f"  Data file: {args.dbn_file}")
    print(f"  Instruments: {instrument_list}")
    for sym in instrument_list:
        spec = registry.get(sym)
        print(
            f"    {sym}: {spec.description} "
            f"(tick={spec.tick_size}, pv=${spec.point_value}, "
            f"comm=${spec.commission_per_side}/side)"
        )
    print(f"  Capital: ${args.capital:,.0f}")
    print(f"  Signal window: {args.window} minutes")
    print(f"  Theta: {args.theta if args.theta else 'auto-calibrate'}")
    print(f"  Stop multiple: {args.stop_multiple}x sigma")
    print(f"  Max position: {args.max_position_pct:.0%} of capital per instrument")
    print(f"  Session cutoff: {args.cutoff_minutes} min before close")
    print("=" * 60)
    print()

    # Build strategy
    strategy = ContraMeanReversionStrategy(
        signal_window_minutes=args.window,
        theta=args.theta,
        theta_multiplier=args.theta_multiplier,
        stop_multiple=args.stop_multiple,
        target_fraction=args.target_fraction,
        max_position_pct=args.max_position_pct,
        session_cutoff_minutes=args.cutoff_minutes,
        min_hold_bars=args.min_hold_bars,
        skip_first_minutes=args.skip_first_minutes,
        lookback_bars=args.lookback,
        z_entry_threshold=args.z_entry,
        z_exit_threshold=args.z_exit,
        confirm_bars=args.confirm_bars,
        momentum_filter_window=args.momentum_window,
        momentum_threshold=args.momentum_threshold,
    )

    # Build engine config
    config = EngineConfig(
        instrument=instrument_list[0],
        initial_capital=args.capital,
        slippage=SlippageConfig(
            model=SlippageModel.FIXED,
            fixed_ticks=args.slippage_ticks,
        ),
        max_position_size=args.max_contracts,
        session_filter=SessionFilter.FULL_GLOBEX,
        enforce_daily_settlement=True,
        results_dir=args.results_dir,
    )

    # Run — engine handles both single-file and multi-file modes
    engine = MultiInstrumentEngine(
        dbn_paths=dbn_source,
        strategy=strategy,
        config=config,
        instruments=instrument_list if single_file else None,
        registry=registry,
    )
    raw_results = engine.run()

    # Compute metrics from aggregated trades
    result = MetricsCalculator.compute(
        strategy_name=raw_results["strategy_name"],
        instrument="+".join(raw_results["instruments"]),
        initial_capital=raw_results["initial_capital"],
        final_equity=raw_results["final_equity"],
        realized_pnl=raw_results["realized_pnl"],
        unrealized_pnl=raw_results["unrealized_pnl"],
        total_commissions=raw_results["total_commissions"],
        total_slippage_dollars=raw_results["total_slippage_dollars"],
        trades=raw_results["all_trades"],
        equity_curve=raw_results["equity_curve"],
        total_margin_calls=raw_results["total_margin_calls"],
        total_forced_liquidations=raw_results["total_forced_liquidations"],
    )

    # Report
    reporter = ReportGenerator(config.results_dir)
    reporter.save_json(result)
    reporter.save_equity_csv(result)
    reporter.save_trade_log_csv(result)
    if args.html:
        reporter.save_html_report(result)

    # Print summary
    reporter.print_summary({strategy.name: result})

    # Per-instrument breakdown
    print("\nPer-Instrument Breakdown:")
    print("-" * 50)
    for sym, inst_data in raw_results["per_instrument"].items():
        n_trades = len(inst_data["trades"])
        realized = inst_data["realized_pnl"]
        comm = inst_data["total_commissions"]
        slip = inst_data["total_slippage_dollars"]
        print(
            f"  {sym}: {n_trades} trades, "
            f"realized=${realized:,.2f}, "
            f"comm=${comm:,.2f}, "
            f"slip=${slip:,.2f}"
        )
    print()


if __name__ == "__main__":
    main()
