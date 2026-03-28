"""Engine configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SlippageModel(Enum):
    FIXED = "fixed"
    NORMAL = "normal"


class SessionFilter(Enum):
    FULL_GLOBEX = "full_globex"
    RTH_ONLY = "rth_only"
    ETH_ONLY = "eth_only"


@dataclass
class SlippageConfig:
    model: SlippageModel = SlippageModel.FIXED
    fixed_ticks: float = 2.80
    # For normal distribution model
    normal_mean_ticks: float = 1.0
    normal_std_ticks: float = 0.5


@dataclass
class EngineConfig:
    """Configuration for the backtesting engine.

    The instrument symbol is used to look up all contract-specific values
    (tick size, point value, commissions, margins, session hours) from the
    ContractRegistry. Nothing instrument-specific is hardcoded in the engine.
    """

    instrument: str = "NQ"  # Root or full symbol -> registry lookup
    initial_capital: float = 100_000.0
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    max_position_size: int = 10  # Max contracts per strategy
    session_filter: SessionFilter = SessionFilter.FULL_GLOBEX
    enforce_daily_settlement: bool = True
    queue_position_penalty: bool = False  # Require trade-through for limits
    results_dir: str = "results"
