"""Report generation: JSON, CSV, stdout summary, and optional HTML."""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from typing import Optional

from analytics.results import BacktestResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate output files and summary reports from backtest results."""

    def __init__(self, results_dir: str = "results") -> None:
        self._results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save_all(
        self, results: dict[str, BacktestResult], generate_html: bool = False
    ) -> None:
        """Save all results and print summary."""
        for name, result in results.items():
            self.save_json(result)
            self.save_equity_csv(result)
            self.save_trade_log_csv(result)
            if generate_html:
                self.save_html_report(result)

        self.print_summary(results)

    def save_json(self, result: BacktestResult) -> str:
        """Save result to JSON file. Returns file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.strategy_name}_{result.instrument}_{timestamp}.json"
        path = os.path.join(self._results_dir, filename)

        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        logger.info("Saved JSON results to %s", path)
        return path

    def save_equity_csv(self, result: BacktestResult) -> str:
        """Save equity curve to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.strategy_name}_{result.instrument}_equity_{timestamp}.csv"
        path = os.path.join(self._results_dir, filename)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "equity", "settlement_mark"])
            for ts, equity, mark in result.equity_curve:
                writer.writerow([ts.isoformat(), f"{equity:.2f}", f"{mark:.2f}" if mark else ""])

        logger.info("Saved equity curve to %s", path)
        return path

    def save_trade_log_csv(self, result: BacktestResult) -> str:
        """Save trade log to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.strategy_name}_{result.instrument}_trades_{timestamp}.csv"
        path = os.path.join(self._results_dir, filename)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "entry_time", "exit_time", "side", "quantity",
                "entry_price", "exit_price", "gross_pnl", "net_pnl",
                "commissions", "slippage_ticks", "slippage_dollars",
                "hold_duration_bars", "notional_value",
            ])
            for t in result.trades:
                writer.writerow([
                    t.entry_time.isoformat(),
                    t.exit_time.isoformat(),
                    t.side,
                    t.quantity,
                    f"{t.entry_price:.4f}",
                    f"{t.exit_price:.4f}",
                    f"{t.gross_pnl:.2f}",
                    f"{t.net_pnl:.2f}",
                    f"{t.commissions:.2f}",
                    f"{t.slippage_ticks:.1f}",
                    f"{t.slippage_dollars:.2f}",
                    t.hold_duration_bars,
                    f"{t.notional_value:.2f}",
                ])

        logger.info("Saved trade log to %s", path)
        return path

    def save_html_report(self, result: BacktestResult) -> str:
        """Generate an HTML report with embedded equity curve chart."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.strategy_name}_{result.instrument}_report_{timestamp}.html"
        path = os.path.join(self._results_dir, filename)

        # Build equity data for inline chart
        eq_labels = []
        eq_values = []
        for ts, equity, _ in result.equity_curve[::max(1, len(result.equity_curve) // 500)]:
            eq_labels.append(ts.strftime("%Y-%m-%d %H:%M"))
            eq_values.append(round(equity, 2))

        html = f"""<!DOCTYPE html>
<html>
<head>
<title>Backtest Report: {result.strategy_name}</title>
<style>
  body {{ font-family: monospace; margin: 40px; background: #1a1a2e; color: #e0e0e0; }}
  h1 {{ color: #00d4ff; }}
  table {{ border-collapse: collapse; margin: 20px 0; }}
  td, th {{ padding: 8px 16px; border: 1px solid #333; text-align: right; }}
  th {{ background: #16213e; color: #00d4ff; text-align: left; }}
  .positive {{ color: #00ff88; }}
  .negative {{ color: #ff4444; }}
  canvas {{ max-width: 100%; margin: 20px 0; }}
</style>
</head>
<body>
<h1>Backtest Report: {result.strategy_name}</h1>
<p>Instrument: {result.instrument} | Capital: ${result.initial_capital:,.0f}</p>

<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Net PnL</td><td class="{'positive' if result.net_pnl >= 0 else 'negative'}">${result.net_pnl:,.2f}</td></tr>
<tr><td>Gross PnL</td><td>${result.gross_pnl:,.2f}</td></tr>
<tr><td>Total Commissions</td><td>${result.total_commissions:,.2f}</td></tr>
<tr><td>Total Slippage</td><td>${result.total_slippage_dollars:,.2f}</td></tr>
<tr><td>Trades</td><td>{result.num_trades}</td></tr>
<tr><td>Win Rate</td><td>{result.win_rate:.1%}</td></tr>
<tr><td>Profit Factor</td><td>{result.profit_factor:.2f}</td></tr>
<tr><td>Sharpe Ratio</td><td>{result.sharpe_ratio:.2f}</td></tr>
<tr><td>Sortino Ratio</td><td>{result.sortino_ratio:.2f}</td></tr>
<tr><td>Max Drawdown</td><td class="negative">${result.max_drawdown_dollars:,.2f} ({result.max_drawdown_pct:.1%})</td></tr>
<tr><td>Avg Hold Time</td><td>{result.avg_hold_time_bars:.1f} bars</td></tr>
<tr><td>Expectancy</td><td>${result.expectancy:,.2f}</td></tr>
<tr><td>Margin Calls</td><td>{result.total_margin_calls}</td></tr>
</table>

<h2>Equity Curve</h2>
<canvas id="chart" width="900" height="400"></canvas>
<script>
const ctx = document.getElementById('chart').getContext('2d');
const labels = {json.dumps(eq_labels)};
const data = {json.dumps(eq_values)};
const w = ctx.canvas.width, h = ctx.canvas.height;
const minV = Math.min(...data), maxV = Math.max(...data);
const range = maxV - minV || 1;
ctx.strokeStyle = '#00d4ff';
ctx.lineWidth = 1.5;
ctx.beginPath();
for (let i = 0; i < data.length; i++) {{
  const x = (i / (data.length - 1)) * w;
  const y = h - ((data[i] - minV) / range) * (h - 40) - 20;
  if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
}}
ctx.stroke();
ctx.fillStyle = '#888';
ctx.font = '12px monospace';
ctx.fillText('$' + maxV.toLocaleString(), 5, 15);
ctx.fillText('$' + minV.toLocaleString(), 5, h - 5);
</script>
</body>
</html>"""

        with open(path, "w") as f:
            f.write(html)

        logger.info("Saved HTML report to %s", path)
        return path

    @staticmethod
    def print_summary(results: dict[str, BacktestResult]) -> None:
        """Print a side-by-side comparison table to stdout."""
        if not results:
            print("No results to display.")
            return

        names = list(results.keys())
        metrics = [
            ("Net PnL", lambda r: f"${r.net_pnl:>12,.2f}"),
            ("Gross PnL", lambda r: f"${r.gross_pnl:>12,.2f}"),
            ("Commissions", lambda r: f"${r.total_commissions:>12,.2f}"),
            ("Slippage $", lambda r: f"${r.total_slippage_dollars:>12,.2f}"),
            ("Trades", lambda r: f"{r.num_trades:>13d}"),
            ("Contracts", lambda r: f"{r.num_contracts_traded:>13d}"),
            ("Win Rate", lambda r: f"{r.win_rate:>12.1%}"),
            ("Avg Winner", lambda r: f"${r.avg_winner:>12,.2f}"),
            ("Avg Loser", lambda r: f"${r.avg_loser:>12,.2f}"),
            ("Largest Win", lambda r: f"${r.largest_winner:>12,.2f}"),
            ("Largest Loss", lambda r: f"${r.largest_loser:>12,.2f}"),
            ("Profit Factor", lambda r: f"{r.profit_factor:>13.2f}"),
            ("Expectancy", lambda r: f"${r.expectancy:>12,.2f}"),
            ("Max DD $", lambda r: f"${r.max_drawdown_dollars:>12,.2f}"),
            ("Max DD %", lambda r: f"{r.max_drawdown_pct:>12.1%}"),
            ("DD Duration", lambda r: f"{r.max_drawdown_duration_bars:>10d} bars"),
            ("Sharpe", lambda r: f"{r.sharpe_ratio:>13.2f}"),
            ("Sortino", lambda r: f"{r.sortino_ratio:>13.2f}"),
            ("Calmar", lambda r: f"{r.calmar_ratio:>13.2f}"),
            ("Avg Hold", lambda r: f"{r.avg_hold_time_bars:>10.1f} bars"),
            ("Max Consec W", lambda r: f"{r.max_consecutive_wins:>13d}"),
            ("Max Consec L", lambda r: f"{r.max_consecutive_losses:>13d}"),
            ("Avg Daily PnL", lambda r: f"${r.avg_daily_pnl:>12,.2f}"),
            ("Worst Day", lambda r: f"${r.worst_daily_pnl:>12,.2f}"),
            ("Best Day", lambda r: f"${r.best_daily_pnl:>12,.2f}"),
            ("Margin Calls", lambda r: f"{r.total_margin_calls:>13d}"),
            ("Forced Liqs", lambda r: f"{r.total_forced_liquidations:>13d}"),
        ]

        col_width = 16
        header = f"{'Metric':<16}" + "".join(f"{n:>{col_width}}" for n in names)
        sep = "-" * len(header)

        print("\n" + sep)
        print("  BACKTEST RESULTS COMPARISON")
        print(sep)
        print(header)
        print(sep)

        for label, fmt_fn in metrics:
            row = f"{label:<16}"
            for name in names:
                row += f"{fmt_fn(results[name]):>{col_width}}"
            print(row)

        print(sep + "\n")
