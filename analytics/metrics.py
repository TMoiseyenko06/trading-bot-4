"""Metrics calculator for backtest results."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

from engine.account import Trade
from analytics.results import BacktestResult


class MetricsCalculator:
    """Compute all performance metrics from raw backtest data."""

    @staticmethod
    def compute(
        strategy_name: str,
        instrument: str,
        initial_capital: float,
        final_equity: float,
        realized_pnl: float,
        unrealized_pnl: float,
        total_commissions: float,
        total_slippage_dollars: float,
        trades: list[Trade],
        equity_curve: list[tuple[datetime, float, Optional[float]]],
        total_margin_calls: int,
        total_forced_liquidations: int,
        daily_profit_cap: float = 0.0,
    ) -> BacktestResult:
        """Compute all metrics and return a BacktestResult."""

        num_trades = len(trades)
        num_contracts = sum(t.quantity for t in trades)

        # Separate winners and losers (using net_pnl)
        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl <= 0]

        win_rate = len(winners) / num_trades if num_trades > 0 else 0.0
        avg_winner = (
            sum(t.net_pnl for t in winners) / len(winners) if winners else 0.0
        )
        avg_loser = (
            sum(t.net_pnl for t in losers) / len(losers) if losers else 0.0
        )
        largest_winner = max((t.net_pnl for t in trades), default=0.0)
        largest_loser = min((t.net_pnl for t in trades), default=0.0)

        gross_winners = sum(t.net_pnl for t in winners)
        gross_losers = abs(sum(t.net_pnl for t in losers))
        profit_factor = (
            gross_winners / gross_losers if gross_losers > 0 else float("inf")
        )

        expectancy = (
            sum(t.net_pnl for t in trades) / num_trades if num_trades > 0 else 0.0
        )

        # Gross PnL (before commissions and slippage)
        gross_pnl = sum(t.gross_pnl for t in trades)
        net_pnl = final_equity - initial_capital

        # Drawdown from equity curve
        dd_dollars, dd_pct, dd_bars, dd_wall = MetricsCalculator._compute_drawdown(
            equity_curve
        )

        # Daily PnL series
        daily_pnls = MetricsCalculator._compute_daily_pnls(equity_curve)

        # Apply daily profit cap — clamp each day's profit to the cap
        if daily_profit_cap > 0 and daily_pnls:
            daily_pnls = [
                min(p, daily_profit_cap) if p > 0 else p
                for p in daily_pnls
            ]
            # Recalculate capped net PnL and final equity
            net_pnl = sum(daily_pnls)
            final_equity = initial_capital + net_pnl

        avg_daily = (
            sum(daily_pnls) / len(daily_pnls) if daily_pnls else 0.0
        )
        worst_daily = min(daily_pnls) if daily_pnls else 0.0
        best_daily = max(daily_pnls) if daily_pnls else 0.0

        # Rebuild capped equity curve for drawdown/sharpe calculations
        if daily_profit_cap > 0 and daily_pnls:
            capped_equity_curve = []
            eq = initial_capital
            for pnl in daily_pnls:
                eq += pnl
                capped_equity_curve.append((None, eq, eq))
            dd_dollars, dd_pct, dd_bars, dd_wall = MetricsCalculator._compute_drawdown(
                capped_equity_curve
            )

        # Sharpe, Sortino, Calmar
        sharpe = MetricsCalculator._annualized_sharpe(daily_pnls)
        sortino = MetricsCalculator._sortino_ratio(daily_pnls)
        calmar = (
            (net_pnl / abs(dd_dollars)) if dd_dollars != 0 else 0.0
        )

        # Hold time
        avg_hold = (
            sum(t.hold_duration_bars for t in trades) / num_trades
            if num_trades > 0
            else 0.0
        )

        # Consecutive wins/losses
        max_consec_wins, max_consec_losses = MetricsCalculator._consecutive_streaks(
            trades
        )

        return_on_dd = net_pnl / abs(dd_dollars) if dd_dollars != 0 else 0.0

        return BacktestResult(
            strategy_name=strategy_name,
            instrument=instrument,
            initial_capital=initial_capital,
            final_equity=final_equity,
            net_pnl=net_pnl,
            gross_pnl=gross_pnl,
            total_commissions=total_commissions,
            total_slippage_dollars=total_slippage_dollars,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            num_trades=num_trades,
            num_contracts_traded=num_contracts,
            win_rate=win_rate,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            largest_winner=largest_winner,
            largest_loser=largest_loser,
            profit_factor=profit_factor,
            expectancy=expectancy,
            max_drawdown_dollars=dd_dollars,
            max_drawdown_pct=dd_pct,
            max_drawdown_duration_bars=dd_bars,
            max_drawdown_duration_wall=dd_wall,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            return_on_max_drawdown=return_on_dd,
            avg_hold_time_bars=avg_hold,
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            avg_daily_pnl=avg_daily,
            worst_daily_pnl=worst_daily,
            best_daily_pnl=best_daily,
            total_margin_calls=total_margin_calls,
            total_forced_liquidations=total_forced_liquidations,
            trades=trades,
            equity_curve=equity_curve,
        )

    @staticmethod
    def _compute_drawdown(
        equity_curve: list[tuple[datetime, float, Optional[float]]],
    ) -> tuple[float, float, int, Optional[str]]:
        """Compute max drawdown in dollars, percent, bars, and wall time."""
        if not equity_curve:
            return 0.0, 0.0, 0, None

        peak = equity_curve[0][1]
        max_dd = 0.0
        max_dd_pct = 0.0
        dd_start_idx = 0
        max_dd_duration = 0
        current_dd_start = 0
        peak_idx = 0
        max_dd_start_time = equity_curve[0][0]
        max_dd_end_time = equity_curve[0][0]

        for i, (ts, equity, _) in enumerate(equity_curve):
            if equity > peak:
                peak = equity
                peak_idx = i
                current_dd_start = i

            dd = peak - equity
            dd_pct = dd / peak if peak > 0 else 0.0

            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
                dd_start_idx = current_dd_start
                max_dd_duration = i - current_dd_start
                max_dd_start_time = equity_curve[current_dd_start][0]
                max_dd_end_time = ts

        wall_time = None
        if max_dd_start_time and max_dd_end_time:
            delta = max_dd_end_time - max_dd_start_time
            hours = delta.total_seconds() / 3600
            if hours >= 24:
                wall_time = f"{hours / 24:.1f} days"
            else:
                wall_time = f"{hours:.1f} hours"

        return max_dd, max_dd_pct, max_dd_duration, wall_time

    @staticmethod
    def _compute_daily_pnls(
        equity_curve: list[tuple[datetime, float, Optional[float]]],
    ) -> list[float]:
        """Extract daily PnL from settlement marks in the equity curve."""
        if not equity_curve:
            return []

        daily_pnls: list[float] = []
        prev_equity: Optional[float] = None

        for ts, equity, settlement_mark in equity_curve:
            if settlement_mark is not None:
                if prev_equity is not None:
                    daily_pnls.append(equity - prev_equity)
                prev_equity = equity

        # If no settlement marks, approximate from first/last equity per day
        if not daily_pnls and len(equity_curve) > 1:
            by_date: dict[str, list[float]] = {}
            for ts, equity, _ in equity_curve:
                date_key = ts.strftime("%Y-%m-%d")
                if date_key not in by_date:
                    by_date[date_key] = []
                by_date[date_key].append(equity)

            dates = sorted(by_date.keys())
            for i in range(1, len(dates)):
                prev_close = by_date[dates[i - 1]][-1]
                curr_close = by_date[dates[i]][-1]
                daily_pnls.append(curr_close - prev_close)

        return daily_pnls

    @staticmethod
    def _annualized_sharpe(daily_pnls: list[float], trading_days: int = 252) -> float:
        """Annualized Sharpe ratio from daily PnL."""
        if len(daily_pnls) < 2:
            return 0.0
        mean = sum(daily_pnls) / len(daily_pnls)
        variance = sum((p - mean) ** 2 for p in daily_pnls) / (len(daily_pnls) - 1)
        std = math.sqrt(variance)
        if std == 0:
            return 0.0
        return (mean / std) * math.sqrt(trading_days)

    @staticmethod
    def _sortino_ratio(daily_pnls: list[float], trading_days: int = 252) -> float:
        """Sortino ratio using downside deviation."""
        if len(daily_pnls) < 2:
            return 0.0
        mean = sum(daily_pnls) / len(daily_pnls)
        downside = [min(0, p) for p in daily_pnls]
        down_var = sum(d ** 2 for d in downside) / len(daily_pnls)
        down_std = math.sqrt(down_var)
        if down_std == 0:
            return 0.0
        return (mean / down_std) * math.sqrt(trading_days)

    @staticmethod
    def _consecutive_streaks(trades: list[Trade]) -> tuple[int, int]:
        """Max consecutive wins and losses."""
        if not trades:
            return 0, 0
        max_wins = 0
        max_losses = 0
        cur_wins = 0
        cur_losses = 0
        for t in trades:
            if t.net_pnl > 0:
                cur_wins += 1
                cur_losses = 0
                max_wins = max(max_wins, cur_wins)
            else:
                cur_losses += 1
                cur_wins = 0
                max_losses = max(max_losses, cur_losses)
        return max_wins, max_losses
