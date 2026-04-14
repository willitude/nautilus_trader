"""Liquidity Regime Indicator for Regime-Aware Market Making.

NautilusTrader Indicator that classifies the current liquidity state
(High Liquidity, Normal, Low Liquidity, Toxic, Crisis) in real-time.

Combined with ReversalScore for the Liquidity Regime-Aware MM v3 strategy.
"""
from __future__ import annotations

from collections import deque

from nautilus_trader.indicators import Indicator
from nautilus_trader.model.data import QuoteTick, TradeTick

from nautilus_trader.indicators.snapshots import LiquidityRegime

_NS_PER_SEC = 1_000_000_000


_DEFAULT_REGIME = LiquidityRegime(
    state="normal", confidence=0.5,
    spread_multiplier=1.0, size_multiplier=1.0,
    inventory_limit_mult=1.0, description="Default normal regime",
)


class LiquidityRegimeIndicator(Indicator):
    """
    NautilusTrader Indicator: rule-based + EWMA liquidity regime detector.

    ``handle_quote_tick`` extracts spread and depth from the BBO.
    ``handle_trade_tick`` updates a rolling trade intensity counter.

    Alpha and basis_vol are external signals; inject them via
    ``set_alpha()`` and ``set_basis_vol()`` before the tick arrives.
    """

    def __init__(self) -> None:
        super().__init__(params=[])

        self.spread_history: deque[float] = deque(maxlen=300)
        self.depth_history: deque[float] = deque(maxlen=300)
        self.intensity_history: deque[float] = deque(maxlen=60)
        self.basis_vol_history: deque[float] = deque(maxlen=180)

        self.current_spread_ewma: float = 0.0
        self.current_depth_ewma: float = 1.0
        self.current_intensity_ewma: float = 1.0

        # Trade intensity tracking
        self._trade_timestamps: deque[int] = deque(maxlen=500)

        # External signal setters
        self._alpha: float = 0.0
        self._basis_vol: float = 0.0

        self.regime: LiquidityRegime = _DEFAULT_REGIME

    # -- External signal injection -------------------------------------------

    def set_alpha(self, alpha: float) -> None:
        self._alpha = alpha

    def set_basis_vol(self, basis_vol: float) -> None:
        self._basis_vol = basis_vol

    # -- Nautilus Indicator interface -----------------------------------------

    def handle_quote_tick(self, tick: QuoteTick) -> None:
        bid = tick.bid_price.as_double()
        ask = tick.ask_price.as_double()
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return

        spread_bps = (ask - bid) / mid * 10_000
        bid_depth = tick.bid_size.as_double()

        ts = int(tick.ts_event)
        trade_intensity = self._rolling_trade_intensity(ts)

        self.regime = self._classify(spread_bps, bid_depth, trade_intensity, self._basis_vol, self._alpha)

        if not self.has_inputs:
            self._set_has_inputs(True)
        if not self.initialized:
            self._set_initialized(True)

    def handle_trade_tick(self, tick: TradeTick) -> None:
        self._trade_timestamps.append(int(tick.ts_event))

    def handle_bar(self, bar) -> None:
        pass  # not used

    def _reset(self) -> None:
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

    # -- State persistence ---------------------------------------------------

    def get_state(self) -> dict[str, float | str]:
        """Return EWMA running state for Redis persistence."""
        return {
            "spread_ewma": self.current_spread_ewma,
            "depth_ewma": self.current_depth_ewma,
            "intensity_ewma": self.current_intensity_ewma,
            "regime_state": self.regime.state,
        }

    def restore_state(self, state: dict) -> None:
        """Restore EWMA running state after node restart."""
        self.current_spread_ewma = float(state.get("spread_ewma", 0.0))
        self.current_depth_ewma = float(state.get("depth_ewma", 1.0))
        self.current_intensity_ewma = float(state.get("intensity_ewma", 1.0))

    # -- Legacy API (backward-compatible with old LiquidityRegimeDetector) ---

    def update(
        self,
        spread_bps: float,
        bid_depth_bps: float,
        trade_intensity: float,
        basis_vol: float = 0.0,
        alpha: float = 0.0,
    ) -> LiquidityRegime:
        """Legacy update method. Prefer ``handle_quote_tick`` for Nautilus integration."""
        self.regime = self._classify(spread_bps, bid_depth_bps, trade_intensity, basis_vol, alpha)

        if not self.has_inputs:
            self._set_has_inputs(True)
        if not self.initialized:
            self._set_initialized(True)

        return self.regime

    def get_default(self) -> LiquidityRegime:
        return _DEFAULT_REGIME

    # -- Internal logic ------------------------------------------------------

    def _rolling_trade_intensity(self, current_ts: int) -> float:
        """Count trades in the last second."""
        cutoff = current_ts - _NS_PER_SEC
        while self._trade_timestamps and self._trade_timestamps[0] < cutoff:
            self._trade_timestamps.popleft()
        return float(len(self._trade_timestamps))

    def _classify(
        self,
        spread_bps: float,
        bid_depth: float,
        trade_intensity: float,
        basis_vol: float,
        alpha: float,
    ) -> LiquidityRegime:
        self.spread_history.append(spread_bps)
        self.depth_history.append(bid_depth)
        self.intensity_history.append(trade_intensity)
        if basis_vol > 0:
            self.basis_vol_history.append(basis_vol)

        # EWMA updates
        if self.current_spread_ewma == 0:
            self.current_spread_ewma = spread_bps
            self.current_depth_ewma = bid_depth
            self.current_intensity_ewma = trade_intensity
        else:
            a = 0.1
            self.current_spread_ewma = a * spread_bps + (1 - a) * self.current_spread_ewma
            self.current_depth_ewma = a * bid_depth + (1 - a) * self.current_depth_ewma
            self.current_intensity_ewma = a * trade_intensity + (1 - a) * self.current_intensity_ewma

        spread_z = (
            (spread_bps - self.current_spread_ewma) / (self.current_spread_ewma + 1e-6)
            if self.current_spread_ewma > 0
            else 0.0
        )
        depth_ratio = bid_depth / (self.current_depth_ewma + 1e-6)
        intensity_z = trade_intensity / (self.current_intensity_ewma + 1e-6)

        if spread_bps > 25 or basis_vol > 0.015:
            return LiquidityRegime(
                state="crisis", confidence=0.9,
                spread_multiplier=3.0, size_multiplier=0.2, inventory_limit_mult=0.5,
                description="Crisis: extreme spread or basis dislocation",
            )
        if spread_z > 2.0 or depth_ratio < 0.3:
            return LiquidityRegime(
                state="low_liquidity", confidence=0.75,
                spread_multiplier=2.0, size_multiplier=0.4, inventory_limit_mult=0.7,
                description="Low liquidity: widened spread, thin depth",
            )
        if intensity_z > 3.0 and abs(alpha) > 0.01:
            return LiquidityRegime(
                state="toxic", confidence=0.85,
                spread_multiplier=1.8, size_multiplier=0.6, inventory_limit_mult=1.0,
                description="Toxic flow: high intensity + strong alpha disagreement",
            )
        if spread_z < -0.8 and depth_ratio > 1.5 and intensity_z > 0.8:
            return LiquidityRegime(
                state="high_liquidity", confidence=0.8,
                spread_multiplier=0.6, size_multiplier=1.8, inventory_limit_mult=1.3,
                description="High liquidity: tight spread, deep book, healthy flow",
            )
        return LiquidityRegime(
            state="normal", confidence=0.6,
            spread_multiplier=1.0, size_multiplier=1.0, inventory_limit_mult=1.0,
            description="Normal regime",
        )


