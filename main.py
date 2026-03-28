#!/usr/bin/env python3
"""CLI entry point for the futures backtesting engine."""

from __future__ import annotations

import argparse
import logging
import sys

from engine.config import EngineConfig, SessionFilter, SlippageConfig, SlippageModel
from engine.contract_registry import ContractRegistry
from engine.core import BacktestEngine
from analytics.metrics import MetricsCalculator
from analytics.report import ReportGenerator
from analytics.results import BacktestResult
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.vwap_reversion import VWAPMeanReversionStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Futures Backtesting Engine — zero lookahead bias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py data/NQ_1m.dbn --instrument NQ
  python main.py data/ES_1m.dbn --instrument ES --capital 50000 --session rth
  python main.py data/NQ_1m.dbn --instrument NQ --strategies ema vwap --html
        """,
    )
    parser.add_argument("dbn_file", help="Path to .dbn file with OHLCV data")
    parser.add_argument(
        "--instrument",
        default="NQ",
        help="Root symbol (NQ, ES, MNQ, CL, etc.). Default: NQ",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Initial account capital. Default: 100000",
    )
    parser.add_argument(
        "--max-position",
        type=int,
        default=10,
        help="Max position size in contracts. Default: 10",
    )
    parser.add_argument(
        "--session",
        choices=["full", "rth", "eth"],
        default="full",
        help="Session filter. Default: full (Globex)",
    )
    parser.add_argument(
        "--slippage-model",
        choices=["fixed", "normal"],
        default="fixed",
        help="Slippage model. Default: fixed",
    )
    parser.add_argument(
        "--slippage-ticks",
        type=float,
        default=1.0,
        help="Fixed slippage ticks (or normal mean). Default: 1.0",
    )
    parser.add_argument(
        "--slippage-std",
        type=float,
        default=0.5,
        help="Normal slippage std dev in ticks. Default: 0.5",
    )
    parser.add_argument(
        "--queue-penalty",
        action="store_true",
        help="Enable queue position penalty for limit orders",
    )
    parser.add_argument(
        "--no-settlement",
        action="store_true",
        help="Disable daily settlement marking",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["ema", "vwap"],
        default=["ema", "vwap"],
        help="Strategies to run. Default: ema vwap",
    )
    parser.add_argument(
        "--ema-fast", type=int, default=9, help="EMA fast period. Default: 9"
    )
    parser.add_argument(
        "--ema-slow", type=int, default=21, help="EMA slow period. Default: 21"
    )
    parser.add_argument(
        "--vwap-entry-std",
        type=float,
        default=2.0,
        help="VWAP entry std devs. Default: 2.0",
    )
    parser.add_argument(
        "--contracts",
        type=int,
        default=1,
        help="Contracts per trade. Default: 1",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Output directory. Default: results",
    )
    parser.add_argument("--html", action="store_true", help="Generate HTML reports")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # Build config
    session_map = {
        "full": SessionFilter.FULL_GLOBEX,
        "rth": SessionFilter.RTH_ONLY,
        "eth": SessionFilter.ETH_ONLY,
    }

    slippage = SlippageConfig(
        model=SlippageModel.FIXED if args.slippage_model == "fixed" else SlippageModel.NORMAL,
        fixed_ticks=args.slippage_ticks,
        normal_mean_ticks=args.slippage_ticks,
        normal_std_ticks=args.slippage_std,
    )

    config = EngineConfig(
        instrument=args.instrument,
        initial_capital=args.capital,
        slippage=slippage,
        max_position_size=args.max_position,
        session_filter=session_map[args.session],
        enforce_daily_settlement=not args.no_settlement,
        queue_position_penalty=args.queue_penalty,
        results_dir=args.results_dir,
    )

    # Build strategies
    strategies = []
    if "ema" in args.strategies:
        strategies.append(
            EMACrossoverStrategy(
                fast_period=args.ema_fast,
                slow_period=args.ema_slow,
                contracts=args.contracts,
            )
        )
    if "vwap" in args.strategies:
        strategies.append(
            VWAPMeanReversionStrategy(
                contracts=args.contracts,
            )
        )

    if not strategies:
        print("No strategies selected. Use --strategies ema vwap")
        sys.exit(1)

    # Verify contract is known
    registry = ContractRegistry()
    spec = registry.get(config.instrument)
    print(f"Instrument: {spec.description} ({spec.symbol})")
    print(f"  Tick size: {spec.tick_size}, Point value: ${spec.point_value}")
    print(f"  Commission: ${spec.commission_per_side}/side")
    print(f"  Initial margin: ${spec.initial_margin:,.0f}")
    print(f"  Capital: ${config.initial_capital:,.0f}")
    print(f"  Strategies: {[s.name for s in strategies]}")
    print()

    # Run engine
    engine = BacktestEngine(args.dbn_file, strategies, config, registry)
    raw_results = engine.run()

    # Compute metrics
    computed_results: dict[str, BacktestResult] = {}
    for strat_name, raw in raw_results.items():
        result = MetricsCalculator.compute(
            strategy_name=strat_name,
            instrument=config.instrument,
            initial_capital=raw["initial_capital"],
            final_equity=raw["final_equity"],
            realized_pnl=raw["realized_pnl"],
            unrealized_pnl=raw["unrealized_pnl"],
            total_commissions=raw["total_commissions"],
            total_slippage_dollars=raw["total_slippage_dollars"],
            trades=raw["trades"],
            equity_curve=raw["equity_curve"],
            total_margin_calls=raw["total_margin_calls"],
            total_forced_liquidations=raw["total_forced_liquidations"],
        )
        computed_results[strat_name] = result

    # Generate reports
    reporter = ReportGenerator(config.results_dir)
    reporter.save_all(computed_results, generate_html=args.html)


if __name__ == "__main__":
    main()
