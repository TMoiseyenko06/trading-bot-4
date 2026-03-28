# Futures Backtesting Engine

A zero-lookahead-bias backtesting engine for CME-style futures contracts. Reads Databento `.dbn` files containing OHLCV bar data, simulates trading with realistic futures market behavior, and supports running multiple strategies simultaneously on a single data pass.

## Architecture

```
engine/              Core engine
  config.py          Engine configuration
  contract_registry  Contract specs (tick size, point value, margins, sessions)
  data_feed.py       Forward-only .dbn iterator
  market_state.py    Read-only state snapshot for strategies
  execution.py       Order types and fill simulation
  account.py         Position, PnL, and margin tracking
  session_handler.py Futures session/calendar awareness
  core.py            Main event loop

strategies/          Trading strategies
  base.py            Abstract strategy interface
  ema_crossover.py   EMA crossover example
  vwap_reversion.py  VWAP mean reversion example

analytics/           Post-run analysis
  metrics.py         Performance metrics calculator
  results.py         Result data objects
  report.py          JSON/CSV/HTML report generation

indicators/          Incremental (online) indicators
  ema.py             Exponential Moving Average
  atr.py             Average True Range
  vwap.py            Volume Weighted Average Price
  rsi.py             Relative Strength Index

main.py              CLI entry point
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and the `databento` SDK.

## Usage

### Basic run with default strategies (EMA crossover + VWAP reversion)

```bash
python main.py data/NQ_1m.dbn --instrument NQ
```

### Custom configuration

```bash
python main.py data/ES_1m.dbn \
  --instrument ES \
  --capital 50000 \
  --session rth \
  --contracts 2 \
  --slippage-model normal \
  --slippage-ticks 1.5 \
  --slippage-std 0.75 \
  --html
```

### Run a single strategy

```bash
python main.py data/NQ_1m.dbn --instrument NQ --strategies ema
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--instrument` | NQ | Root symbol (NQ, ES, MNQ, MES, YM, MYM, CL, GC, ZB) |
| `--capital` | 100000 | Initial account capital |
| `--max-position` | 10 | Max contracts per strategy |
| `--session` | full | Session filter: full, rth, eth |
| `--slippage-model` | fixed | Slippage model: fixed, normal |
| `--slippage-ticks` | 1.0 | Slippage in ticks |
| `--queue-penalty` | off | Require trade-through for limit fills |
| `--no-settlement` | off | Disable daily settlement marking |
| `--strategies` | ema vwap | Strategies to run |
| `--contracts` | 1 | Contracts per trade |
| `--results-dir` | results | Output directory |
| `--html` | off | Generate HTML reports |
| `-v` | off | Verbose/debug logging |

## Supported Contracts

| Symbol | Exchange | Tick Size | Point Value | Commission/Side |
|--------|----------|-----------|-------------|-----------------|
| NQ | CME | 0.25 | $20 | $2.25 |
| MNQ | CME | 0.25 | $2 | $0.62 |
| ES | CME | 0.25 | $50 | $2.25 |
| MES | CME | 0.25 | $5 | $0.62 |
| YM | CME | 1.0 | $5 | $2.25 |
| MYM | CME | 1.0 | $0.50 | $0.62 |
| CL | CME | 0.01 | $1000 | $2.25 |
| GC | COMEX | 0.10 | $100 | $2.25 |
| ZB | CBOT | 1/32 | $1000 | $1.52 |

## Writing a Custom Strategy

```python
from strategies.base import Strategy, SubmitOrderFn
from engine.execution import Order, OrderSide, OrderType, TimeInForce
from engine.market_state import MarketState

class MyStrategy(Strategy):
    def __init__(self):
        super().__init__("MyStrategy")

    def on_init(self):
        self._prev_close = None

    def on_bar(self, state: MarketState, submit_order: SubmitOrderFn):
        bar = state.bar

        # Your incremental logic here
        if self._prev_close is not None and state.position_quantity == 0:
            if bar.close > self._prev_close * 1.001:
                submit_order(Order(
                    side=OrderSide.BUY,
                    quantity=1,
                    order_type=OrderType.MARKET,
                ))

        self._prev_close = bar.close
```

## Design Principles

- **Zero lookahead bias**: Data iterator is private to the event loop. Strategies only see what they are handed. Orders fill in the future, never the present.
- **Futures-native**: Positions in whole contracts. PnL via point value. Margin as performance bond. Daily settlement modeled.
- **Incremental indicators**: No pandas rolling windows. No ta-lib over full arrays. One bar at a time.
- **Single data pass**: Multiple strategies process each bar before advancing. Independent state per strategy.
