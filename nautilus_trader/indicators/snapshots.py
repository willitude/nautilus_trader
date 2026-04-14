"""Market microstructure snapshot dataclasses.

Pure data containers serving as the contract interface between
NautilusTrader Indicators and quant logic (policies, simulators).
"""

from __future__ import annotations

from dataclasses import dataclass

_WINDOW_5S_NS = 5_000_000_000


@dataclass(frozen=True)
class MarketStateConfig:
    window_ns: int = _WINDOW_5S_NS
    mid_history_maxlen: int = 5000


@dataclass(frozen=True)
class MarketStateSnapshot:
    mid: float
    microprice: float
    spread_rel: float
    imbalance_l1: float
    microprice_dev: float
    ofi: float
    volatility_5s: float
    momentum_5s: float
    cross_basis: float
    btc_cross_basis: float


@dataclass(frozen=True)
class ReversalScoreSnapshot:
    score: float
    imbalance: float
    imbalance_strength: float
    autocorr_score: float
    disagreement: float
    ret_short: float
    ret_long: float


@dataclass(frozen=True)
class LiquidityRegime:
    """Liquidity regime state with policy multipliers."""
    state: str  # "high_liquidity", "normal", "low_liquidity", "toxic", "crisis"
    confidence: float
    spread_multiplier: float = 1.0
    size_multiplier: float = 1.0
    inventory_limit_mult: float = 1.0
    description: str = ""


@dataclass(frozen=True)
class BasisMomentumSnapshot:
    basis: float
    btc_basis: float

    momentum_short: float
    momentum_mid: float
    momentum_long: float

    acceleration: float

    basis_btc_divergence: float
    divergence_z: float

    basis_ewma_fast: float
    basis_ewma_slow: float
    ewma_crossover: float

    regime: str
    regime_strength: float
