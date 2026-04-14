# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython-optimized LiquidityRegimeIndicator.

Same logic as liquidity_regime.py but with cdef methods and C-level math
for ~10-30x faster tick-by-tick processing in feature_engineer.py.
"""
from libc.math cimport fabs
from libc.stdint cimport int64_t

from nautilus_trader.indicators.base cimport Indicator
from nautilus_trader.model.data cimport QuoteTick, TradeTick, Bar

from collections import deque
from dataclasses import dataclass


cdef long long _NS_PER_SEC = 1_000_000_000


@dataclass(frozen=True)
class LiquidityRegime:
    """Liquidity regime state with policy multipliers."""
    state: str
    confidence: float
    spread_multiplier: float = 1.0
    size_multiplier: float = 1.0
    inventory_limit_mult: float = 1.0
    description: str = ""


cdef object _DEFAULT_REGIME = LiquidityRegime(
    state="normal", confidence=0.5,
    spread_multiplier=1.0, size_multiplier=1.0,
    inventory_limit_mult=1.0, description="Default normal regime",
)

# Pre-allocated regime singletons to avoid object creation in hot path
cdef object _CRISIS_REGIME = LiquidityRegime(
    state="crisis", confidence=0.9,
    spread_multiplier=3.0, size_multiplier=0.2, inventory_limit_mult=0.5,
    description="Crisis: extreme spread or basis dislocation",
)
cdef object _LOW_LIQ_REGIME = LiquidityRegime(
    state="low_liquidity", confidence=0.75,
    spread_multiplier=2.0, size_multiplier=0.4, inventory_limit_mult=0.7,
    description="Low liquidity: widened spread, thin depth",
)
cdef object _TOXIC_REGIME = LiquidityRegime(
    state="toxic", confidence=0.85,
    spread_multiplier=1.8, size_multiplier=0.6, inventory_limit_mult=1.0,
    description="Toxic flow: high intensity + strong alpha disagreement",
)
cdef object _HIGH_LIQ_REGIME = LiquidityRegime(
    state="high_liquidity", confidence=0.8,
    spread_multiplier=0.6, size_multiplier=1.8, inventory_limit_mult=1.3,
    description="High liquidity: tight spread, deep book, healthy flow",
)
cdef object _NORMAL_REGIME = LiquidityRegime(
    state="normal", confidence=0.6,
    spread_multiplier=1.0, size_multiplier=1.0, inventory_limit_mult=1.0,
    description="Normal regime",
)


cdef class LiquidityRegimeIndicator(Indicator):
    """Cython NautilusTrader Indicator: rule-based + EWMA liquidity regime detector."""

    cdef:
        object spread_history
        object depth_history
        object intensity_history
        object basis_vol_history
        object _trade_timestamps

        double current_spread_ewma
        double current_depth_ewma
        double current_intensity_ewma
        double _alpha
        double _basis_vol

    cdef public object regime

    def __init__(self):
        Indicator.__init__(self, params=[])

        self.spread_history = deque(maxlen=300)
        self.depth_history = deque(maxlen=300)
        self.intensity_history = deque(maxlen=60)
        self.basis_vol_history = deque(maxlen=180)
        self._trade_timestamps = deque(maxlen=500)

        self.current_spread_ewma = 0.0
        self.current_depth_ewma = 1.0
        self.current_intensity_ewma = 1.0
        self._alpha = 0.0
        self._basis_vol = 0.0
        self.regime = _DEFAULT_REGIME

    def set_alpha(self, double alpha):
        self._alpha = alpha

    def set_basis_vol(self, double basis_vol):
        self._basis_vol = basis_vol

    # -- Nautilus Indicator interface -------------------------------------------

    cpdef void handle_quote_tick(self, QuoteTick tick):
        cdef:
            double bid = tick.bid_price.as_double()
            double ask = tick.ask_price.as_double()
            double mid = (bid + ask) * 0.5
            double spread_bps, bid_depth, trade_intensity
            long long ts

        if mid <= 0.0:
            return

        spread_bps = (ask - bid) / mid * 10000.0
        bid_depth = tick.bid_size.as_double()
        ts = <long long>tick.ts_event
        trade_intensity = self._rolling_intensity_c(ts)

        self.regime = self._classify_c(spread_bps, bid_depth, trade_intensity, self._basis_vol, self._alpha)

        if not self.has_inputs:
            self._set_has_inputs(True)
        if not self.initialized:
            self._set_initialized(True)

    cpdef void handle_trade_tick(self, TradeTick tick):
        self._trade_timestamps.append(<long long>tick.ts_event)

    cpdef void handle_bar(self, Bar bar):
        pass

    cpdef void _reset(self):
        self.spread_history.clear()
        self.depth_history.clear()
        self.intensity_history.clear()
        self.basis_vol_history.clear()
        self._trade_timestamps.clear()
        self.current_spread_ewma = 0.0
        self.current_depth_ewma = 1.0
        self.current_intensity_ewma = 1.0
        self._alpha = 0.0
        self._basis_vol = 0.0
        self.regime = _DEFAULT_REGIME

    # -- Legacy API (backward-compatible) ------------------------------------

    def update(self, double spread_bps, double bid_depth_bps, double trade_intensity,
               double basis_vol=0.0, double alpha=0.0):
        self.regime = self._classify_c(spread_bps, bid_depth_bps, trade_intensity, basis_vol, alpha)
        if not self.has_inputs:
            self._set_has_inputs(True)
        if not self.initialized:
            self._set_initialized(True)
        return self.regime

    def get_default(self):
        return _DEFAULT_REGIME

    # -- Internal logic (all cdef) -------------------------------------------

    cdef double _rolling_intensity_c(self, long long current_ts):
        cdef long long cutoff = current_ts - _NS_PER_SEC
        while self._trade_timestamps and (<long long>self._trade_timestamps[0]) < cutoff:
            self._trade_timestamps.popleft()
        return <double>len(self._trade_timestamps)

    cdef object _classify_c(self, double spread_bps, double bid_depth,
                             double trade_intensity, double basis_vol, double alpha):
        cdef:
            double a = 0.1
            double spread_z, depth_ratio, intensity_z

        self.spread_history.append(spread_bps)
        self.depth_history.append(bid_depth)
        self.intensity_history.append(trade_intensity)
        if basis_vol > 0.0:
            self.basis_vol_history.append(basis_vol)

        if self.current_spread_ewma == 0.0:
            self.current_spread_ewma = spread_bps
            self.current_depth_ewma = bid_depth
            self.current_intensity_ewma = trade_intensity
        else:
            self.current_spread_ewma = a * spread_bps + (1.0 - a) * self.current_spread_ewma
            self.current_depth_ewma = a * bid_depth + (1.0 - a) * self.current_depth_ewma
            self.current_intensity_ewma = a * trade_intensity + (1.0 - a) * self.current_intensity_ewma

        if self.current_spread_ewma > 0.0:
            spread_z = (spread_bps - self.current_spread_ewma) / (self.current_spread_ewma + 1e-6)
        else:
            spread_z = 0.0
        depth_ratio = bid_depth / (self.current_depth_ewma + 1e-6)
        intensity_z = trade_intensity / (self.current_intensity_ewma + 1e-6)

        if spread_bps > 25.0 or basis_vol > 0.015:
            return _CRISIS_REGIME
        if spread_z > 2.0 or depth_ratio < 0.3:
            return _LOW_LIQ_REGIME
        if intensity_z > 3.0 and fabs(alpha) > 0.01:
            return _TOXIC_REGIME
        if spread_z < -0.8 and depth_ratio > 1.5 and intensity_z > 0.8:
            return _HIGH_LIQ_REGIME
        return _NORMAL_REGIME


