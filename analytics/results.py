"""Result data objects for backtest output."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from engine.account import Trade


@dataclass
class BacktestResult:
    """Complete result for a single strategy's backtest run."""

    strategy_name: str
    instrument: str

    # Capital
    initial_capital: float
    final_equity: float

    # PnL
    net_pnl: float  # After all costs
    gross_pnl: float  # Before costs
    total_commissions: float
    total_slippage_dollars: float
    realized_pnl: float
    unrealized_pnl: float

    # Trade stats
    num_trades: int
    num_contracts_traded: int
    win_rate: float
    avg_winner: float
    avg_loser: float
    largest_winner: float
    largest_loser: float
    profit_factor: float
    expectancy: float  # Average expected PnL per trade

    # Drawdown
    max_drawdown_dollars: float
    max_drawdown_pct: float
    max_drawdown_duration_bars: int
    max_drawdown_duration_wall: Optional[str]  # Human-readable

    # Risk ratios
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    return_on_max_drawdown: float

    # Time stats
    avg_hold_time_bars: float
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Daily PnL
    avg_daily_pnl: float
    worst_daily_pnl: float
    best_daily_pnl: float

    # Margin events
    total_margin_calls: int
    total_forced_liquidations: int

    # Data
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float, Optional[float]]] = field(
        default_factory=list
    )

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        result = {}
        for key, val in self.__dict__.items():
            if key == "trades":
                result[key] = [
                    {
                        "entry_time": str(t.entry_time),
                        "exit_time": str(t.exit_time),
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "quantity": t.quantity,
                        "side": t.side,
                        "gross_pnl": round(t.gross_pnl, 2),
                        "net_pnl": round(t.net_pnl, 2),
                        "commissions": round(t.commissions, 2),
                        "slippage_ticks": round(t.slippage_ticks, 2),
                        "slippage_dollars": round(t.slippage_dollars, 2),
                        "hold_duration_bars": t.hold_duration_bars,
                        "notional_value": round(t.notional_value, 2),
                    }
                    for t in val
                ]
            elif key == "equity_curve":
                result[key] = [
                    {
                        "timestamp": str(ts),
                        "equity": round(eq, 2),
                        "settlement_mark": round(sm, 2) if sm else None,
                    }
                    for ts, eq, sm in val
                ]
            elif isinstance(val, float):
                result[key] = round(val, 4)
            else:
                result[key] = val
        return result
