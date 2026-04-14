"""Basis Momentum Indicator – cross-exchange basis temporal dynamics.

Discovered via Deep Encoder SHAP analysis:
  - cross_basis_mean (15.9%) and btc_cross_basis_mean (10.3%) were top features
  - Encoder learned start-vs-end comparison (timesteps 0 & 63 most important)
  - Distinct cluster regimes driven by basis level and momentum

This indicator distills those findings into an efficient, tick-level
NautilusTrader Indicator that computes:
  1. Multi-horizon basis momentum (short / mid / long)
  2. Basis acceleration (2nd derivative)
  3. Coin-vs-BTC basis divergence with z-score
  4. Dual-EWMA crossover signal
  5. Regime classification (convergent / divergent / stable / dislocation)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt

from nautilus_trader.indicators import Indicator
from nautilus_trader.model.data import QuoteTick

from nautilus_trader.indicators.snapshots import BasisMomentumSnapshot

_NS_PER_SEC = 1_000_000_000


@dataclass(frozen=True)
class BasisMomentumConfig:
    short_horizon_ns: int = 2 * _NS_PER_SEC
    mid_horizon_ns: int = 10 * _NS_PER_SEC
    long_horizon_ns: int = 30 * _NS_PER_SEC
    history_ns: int = 60 * _NS_PER_SEC
    ewma_alpha_fast: float = 0.15
    ewma_alpha_slow: float = 0.03
    acceleration_window: int = 16
    divergence_ewma_alpha: float = 0.05
    regime_momentum_threshold: float = 0.3
    regime_dislocation_threshold: float = 2.0


_ZERO_SNAPSHOT = BasisMomentumSnapshot(
    basis=0.0, btc_basis=0.0,
    momentum_short=0.0, momentum_mid=0.0, momentum_long=0.0,
    acceleration=0.0,
    basis_btc_divergence=0.0, divergence_z=0.0,
    basis_ewma_fast=0.0, basis_ewma_slow=0.0, ewma_crossover=0.0,
    regime="stable", regime_strength=0.0,
)


class BasisMomentumIndicator(Indicator):
    """NautilusTrader Indicator: cross-exchange basis momentum & regime.

    External signals ``basis`` (coin) and ``btc_basis`` must be injected via
    ``set_basis()`` / ``set_btc_basis()`` before each ``handle_quote_tick``.
    The indicator itself only uses ``QuoteTick.ts_event`` for timing.
    """

    def __init__(self, config: BasisMomentumConfig | None = None) -> None:
        self.config = config or BasisMomentumConfig()
        super().__init__(params=[
            self.config.short_horizon_ns,
            self.config.mid_horizon_ns,
            self.config.long_horizon_ns,
        ])

        self._basis_history: deque[tuple[int, float]] = deque()
        self._btc_basis_history: deque[tuple[int, float]] = deque()
        self._momentum_history: deque[float] = deque(maxlen=self.config.acceleration_window)

        self._ewma_fast: float = 0.0
        self._ewma_slow: float = 0.0
        self._ewma_initialized: bool = False

        self._div_ewma: float = 0.0
        self._div_var_ewma: float = 0.0
        self._div_initialized: bool = False

        self._basis: float = 0.0
        self._btc_basis: float = 0.0

        self.snapshot: BasisMomentumSnapshot = _ZERO_SNAPSHOT

    # -- External signal injection -------------------------------------------

    def set_basis(self, basis_bps: float) -> None:
        """Inject current cross-exchange basis (in bps) before next tick."""
        self._basis = basis_bps

    def set_btc_basis(self, btc_basis_bps: float) -> None:
        """Inject current BTC cross-exchange basis (in bps) before next tick."""
        self._btc_basis = btc_basis_bps

    # -- Nautilus Indicator interface ----------------------------------------

    def handle_quote_tick(self, tick: QuoteTick) -> None:
        ts = int(tick.ts_event)
        self.snapshot = self._compute(ts, self._basis, self._btc_basis)

        if not self.has_inputs:
            self._set_has_inputs(True)
        if not self.initialized:
            self._set_initialized(True)

    def handle_trade_tick(self, tick) -> None:
        pass

    def handle_bar(self, bar) -> None:
        pass

    def _reset(self) -> None:
        self._basis_history.clear()
        self._btc_basis_history.clear()
        self._momentum_history.clear()
        self._ewma_fast = 0.0
        self._ewma_slow = 0.0
        self._ewma_initialized = False
        self._div_ewma = 0.0
        self._div_var_ewma = 0.0
        self._div_initialized = False
        self._basis = 0.0
        self._btc_basis = 0.0
        self.snapshot = _ZERO_SNAPSHOT

    # -- State persistence ---------------------------------------------------

    def get_state(self) -> dict[str, float | bool]:
        """Return EWMA running state for Redis persistence."""
        return {
            "ewma_fast": self._ewma_fast,
            "ewma_slow": self._ewma_slow,
            "ewma_initialized": self._ewma_initialized,
            "div_ewma": self._div_ewma,
            "div_var_ewma": self._div_var_ewma,
            "div_initialized": self._div_initialized,
        }

    def restore_state(self, state: dict) -> None:
        """Restore EWMA running state after node restart."""
        self._ewma_fast = float(state.get("ewma_fast", 0.0))
        self._ewma_slow = float(state.get("ewma_slow", 0.0))
        self._ewma_initialized = bool(state.get("ewma_initialized", False))
        self._div_ewma = float(state.get("div_ewma", 0.0))
        self._div_var_ewma = float(state.get("div_var_ewma", 0.0))
        self._div_initialized = bool(state.get("div_initialized", False))

    # -- Legacy API ----------------------------------------------------------

    def update(
        self,
        ts_event: int,
        basis_bps: float,
        btc_basis_bps: float,
    ) -> BasisMomentumSnapshot:
        """Direct update without QuoteTick. For backtest/research use."""
        self._basis = basis_bps
        self._btc_basis = btc_basis_bps
        self.snapshot = self._compute(ts_event, basis_bps, btc_basis_bps)

        if not self.has_inputs:
            self._set_has_inputs(True)
        if not self.initialized:
            self._set_initialized(True)

        return self.snapshot

    # -- Core logic ----------------------------------------------------------

    def _compute(
        self,
        ts_event: int,
        basis: float,
        btc_basis: float,
    ) -> BasisMomentumSnapshot:
        cfg = self.config

        self._basis_history.append((ts_event, basis))
        self._btc_basis_history.append((ts_event, btc_basis))
        self._trim(ts_event)

        # 1. Multi-horizon momentum (bps/s)
        mom_short = self._momentum(self._basis_history, ts_event, basis, cfg.short_horizon_ns)
        mom_mid = self._momentum(self._basis_history, ts_event, basis, cfg.mid_horizon_ns)
        mom_long = self._momentum(self._basis_history, ts_event, basis, cfg.long_horizon_ns)

        # 2. Acceleration (change in short momentum)
        self._momentum_history.append(mom_short)
        accel = self._acceleration()

        # 3. Dual-EWMA crossover
        if not self._ewma_initialized:
            self._ewma_fast = basis
            self._ewma_slow = basis
            self._ewma_initialized = True
        else:
            self._ewma_fast += cfg.ewma_alpha_fast * (basis - self._ewma_fast)
            self._ewma_slow += cfg.ewma_alpha_slow * (basis - self._ewma_slow)
        crossover = self._ewma_fast - self._ewma_slow

        # 4. Coin-vs-BTC divergence with running z-score
        divergence = basis - btc_basis
        div_z = self._update_divergence_z(divergence)

        # 5. Regime classification
        regime, strength = self._classify_regime(mom_short, mom_long, crossover, div_z)

        return BasisMomentumSnapshot(
            basis=basis,
            btc_basis=btc_basis,
            momentum_short=mom_short,
            momentum_mid=mom_mid,
            momentum_long=mom_long,
            acceleration=accel,
            basis_btc_divergence=divergence,
            divergence_z=div_z,
            basis_ewma_fast=self._ewma_fast,
            basis_ewma_slow=self._ewma_slow,
            ewma_crossover=crossover,
            regime=regime,
            regime_strength=strength,
        )

    # -- Helpers -------------------------------------------------------------

    def _trim(self, ts_event: int) -> None:
        min_ts = ts_event - self.config.history_ns
        while self._basis_history and self._basis_history[0][0] < min_ts:
            self._basis_history.popleft()
        while self._btc_basis_history and self._btc_basis_history[0][0] < min_ts:
            self._btc_basis_history.popleft()

    @staticmethod
    def _momentum(
        history: deque[tuple[int, float]],
        ts_now: int,
        current_val: float,
        horizon_ns: int,
    ) -> float:
        """Basis change rate (bps/s) over the given horizon."""
        target_ts = ts_now - horizon_ns
        past_val = None
        for ts, val in reversed(history):
            if ts <= target_ts:
                past_val = val
                break
        if past_val is None:
            return 0.0
        dt_sec = horizon_ns / _NS_PER_SEC
        if dt_sec <= 0:
            return 0.0
        return (current_val - past_val) / dt_sec

    def _acceleration(self) -> float:
        if len(self._momentum_history) < 3:
            return 0.0
        recent = list(self._momentum_history)
        n = len(recent)
        half = n // 2
        first_half = sum(recent[:half]) / half
        second_half = sum(recent[half:]) / (n - half)
        return second_half - first_half

    def _update_divergence_z(self, divergence: float) -> float:
        """Welford-style running z-score of coin-BTC basis divergence."""
        alpha = self.config.divergence_ewma_alpha
        if not self._div_initialized:
            self._div_ewma = divergence
            self._div_var_ewma = 0.0
            self._div_initialized = True
            return 0.0

        self._div_ewma += alpha * (divergence - self._div_ewma)
        diff = divergence - self._div_ewma
        self._div_var_ewma += alpha * (diff * diff - self._div_var_ewma)

        std = sqrt(self._div_var_ewma) if self._div_var_ewma > 0 else 1e-12
        if std < 1e-12:
            return 0.0
        return diff / std

    def _classify_regime(
        self,
        mom_short: float,
        mom_long: float,
        crossover: float,
        div_z: float,
    ) -> tuple[str, float]:
        cfg = self.config

        # Dislocation: extreme divergence from BTC
        if abs(div_z) > cfg.regime_dislocation_threshold:
            return "dislocation", min(abs(div_z) / (cfg.regime_dislocation_threshold * 2), 1.0)

        # Directional agreement between short and long momentum
        same_direction = (mom_short > 0) == (mom_long > 0) and mom_short != 0

        if same_direction and abs(mom_short) > cfg.regime_momentum_threshold:
            if crossover > 0 and mom_short > 0:
                strength = min(abs(crossover) / 2.0 + abs(mom_short) / 2.0, 1.0)
                return "divergent", strength
            elif crossover < 0 and mom_short < 0:
                strength = min(abs(crossover) / 2.0 + abs(mom_short) / 2.0, 1.0)
                return "convergent", strength

        if abs(mom_short) < cfg.regime_momentum_threshold * 0.5:
            return "stable", max(0.0, 1.0 - abs(mom_short) / cfg.regime_momentum_threshold)

        # Transitional: weak momentum or mixed signals
        if mom_short > 0:
            return "divergent", min(abs(mom_short) / cfg.regime_momentum_threshold, 0.5)
        return "convergent", min(abs(mom_short) / cfg.regime_momentum_threshold, 0.5)
