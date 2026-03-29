#!/usr/bin/env python3
"""CLI entry point for V3 contrarian mean-reversion basket strategy.

Uses optimal V2 base parameters with V3 improvements.

Usage:
  python run_contrarian_v3.py data/multi.dbn --instruments NQ ES RTY YM
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
from strategies.contrarian_reversion_v3 import ContraMeanReversionV3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V3 Contrarian Mean-Reversion Basket Strategy",
    )

    parser.add_argument("dbn_file", help="Single .dbn file with all instruments")
    parser.add_argument("--instruments", nargs="+", required=True)

    # Base params (V2 optimal defaults)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--z-entry", type=float, default=2.0)
    parser.add_argument("--z-exit", type=float, default=0.5)
    parser.add_argument("--confirm-bars", type=int, default=2)
    parser.add_argument("--stop-multiple", type=float, default=4.0)
    parser.add_argument("--momentum-threshold", type=float, default=0.75)
    parser.add_argument("--min-hold-bars", type=int, default=3)
    parser.add_argument("--skip-first-minutes", type=int, default=0)

    # V3 params
    parser.add_argument("--z-spread", type=float, default=2.0)
    parser.add_argument("--exit-tighten-bars", type=int, default=20)
    parser.add_argument("--exit-tighten-rate", type=float, default=0.02)
    parser.add_argument("--volume-spike", type=float, default=1.3)
    parser.add_argument("--per-leg-exit", action="store_true", default=True)
    parser.add_argument("--no-per-leg-exit", dest="per_leg_exit", action="store_false")
    parser.add_argument("--time-weight", action="store_true", default=False)
    parser.add_argument("--no-time-weight", dest="time_weight", action="store_false")

    # Engine params
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--max-contracts", type=int, default=20)
    parser.add_argument("--results-dir", default="results_v3")
    parser.add_argument("--html", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    instruments = [s.upper() for s in args.instruments]
    registry = ContractRegistry()

    print("=" * 60)
    print("  Contrarian Mean-Reversion V3 Strategy")
    print("=" * 60)
    print(f"  Data: {args.dbn_file}")
    print(f"  Instruments: {instruments}")
    print(f"  Capital: ${args.capital:,.0f}")
    print()
    print("  Base params:")
    print(f"    lookback={args.lookback}, z_entry={args.z_entry}, z_exit={args.z_exit}")
    print(f"    confirm={args.confirm_bars}, stop={args.stop_multiple}, momentum={args.momentum_threshold}")
    print()
    print("  V3 params:")
    print(f"    z_spread={args.z_spread}, exit_tighten={args.exit_tighten_bars}/{args.exit_tighten_rate}")
    print(f"    volume_spike={args.volume_spike}, per_leg={args.per_leg_exit}, time_weight={args.time_weight}")
    print("=" * 60)
    print()

    strategy = ContraMeanReversionV3(
        lookback_bars=args.lookback,
        z_entry_threshold=args.z_entry,
        z_exit_threshold=args.z_exit,
        confirm_bars=args.confirm_bars,
        stop_multiple=args.stop_multiple,
        momentum_threshold=args.momentum_threshold,
        min_hold_bars=args.min_hold_bars,
        skip_first_minutes=args.skip_first_minutes,
        z_spread_threshold=args.z_spread,
        exit_tighten_bars=args.exit_tighten_bars,
        exit_tighten_rate=args.exit_tighten_rate,
        volume_spike_multiple=args.volume_spike,
        per_leg_exit=args.per_leg_exit,
        time_weight_enabled=args.time_weight,
    )

    config = EngineConfig(
        instrument=instruments[0],
        initial_capital=args.capital,
        slippage=SlippageConfig(model=SlippageModel.FIXED, fixed_ticks=0.0),
        max_position_size=args.max_contracts,
        session_filter=SessionFilter.FULL_GLOBEX,
        enforce_daily_settlement=True,
        results_dir=args.results_dir,
    )

    engine = MultiInstrumentEngine(
        dbn_paths=args.dbn_file,
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

    reporter = ReportGenerator(config.results_dir)
    reporter.save_json(result)
    reporter.save_equity_csv(result)
    reporter.save_trade_log_csv(result)
    if args.html:
        reporter.save_html_report(result)

    reporter.print_summary({strategy.name: result})

    print("\nPer-Instrument Breakdown:")
    print("-" * 50)
    for sym, inst_data in raw["per_instrument"].items():
        n_trades = len(inst_data["trades"])
        realized = inst_data["realized_pnl"]
        print(f"  {sym}: {n_trades} trades, realized=${realized:,.2f}")
    print()


if __name__ == "__main__":
    main()
