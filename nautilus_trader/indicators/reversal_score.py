from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from nautilus_trader.indicators import Indicator
from nautilus_trader.model.data import QuoteTick

from nautilus_trader.indicators.snapshots import ReversalScoreSnapshot


@dataclass(frozen=True)
class ReversalScoreConfig:
    short_horizon_ns: int = 1_000_000_000
    long_horizon_ns: int = 5_000_000_000
    history_ns: int = 30_000_000_000
    autocorr_window: int = 32
    alpha_threshold: float = 0.008
    imbalance_weight: float = 0.35
    autocorr_weight: float = 0.25
    disagreement_weight: float = 0.40


class ReversalScore(Indicator):
    """
    NautilusTrader Indicator: toxic-flow / reversal proxy.

    Derived from the paper intuition (2502.18625): strong imbalance can
    improve fills while worsening post-fill returns.

    Uses:
      1. L1 imbalance strength
      2. short/long return autocorrelation proxy
      3. disagreement between alpha direction and imbalance direction

    Alpha must be injected via ``set_alpha()`` before each tick because
    ``handle_quote_tick`` only receives a ``QuoteTick``.
    """

    def __init__(self, config: ReversalScoreConfig | None = None) -> None:
        self.config = config or ReversalScoreConfig()
        super().__init__(params=[
            self.config.short_horizon_ns,
            self.config.long_horizon_ns,
            self.config.alpha_threshold,
        ])
        self._mid_history: deque[tuple[int, float]] = deque()
        self._ret_pairs: deque[tuple[float, float]] = deque(maxlen=self.config.autocorr_window)
        self._alpha: float = 0.0
        self.latest_snapshot: ReversalScoreSnapshot = ReversalScoreSnapshot(
            score=0.0, imbalance=0.0, imbalance_strength=0.0,
            autocorr_score=0.5, disagreement=0.0, ret_short=0.0, ret_long=0.0,
        )

    def set_alpha(self, alpha: float) -> None:
        """Inject the current alpha value before the next tick update."""
        self._alpha = alpha

    # -- Nautilus Indicator interface -----------------------------------------

    def handle_quote_tick(self, tick: QuoteTick) -> None:
        ts_event = int(tick.ts_event)
        mid_price = (tick.bid_price.as_double() + tick.ask_price.as_double()) / 2.0
        bid_size = tick.bid_size.as_double()
        ask_size = tick.ask_size.as_double()

        self.latest_snapshot = self._compute(ts_event, mid_price, bid_size, ask_size, self._alpha)

        if not self.has_inputs:
            self._set_has_inputs(True)
        if not self.initialized:
            self._set_initialized(True)

    def handle_trade_tick(self, tick) -> None:
        pass  # not used

    def handle_bar(self, bar) -> None:
        pass  # not used

    def _reset(self) -> None:
        self._mid_history.clear()
        self._ret_pairs.clear()
        self._alpha = 0.0
        self.latest_snapshot = ReversalScoreSnapshot(
            score=0.0, imbalance=0.0, imbalance_strength=0.0,
            autocorr_score=0.5, disagreement=0.0, ret_short=0.0, ret_long=0.0,
        )

    # -- Legacy API (backward-compatible) ------------------------------------

    def update(
        self,
        ts_event: int,
        mid_price: float,
        bid_size: float,
        ask_size: float,
        alpha: float,
    ) -> ReversalScoreSnapshot:
        """Legacy update method. Prefer ``handle_quote_tick`` for Nautilus integration."""
        self._alpha = alpha
        snapshot = self._compute(ts_event, mid_price, bid_size, ask_size, alpha)
        self.latest_snapshot = snapshot

        if not self.has_inputs:
            self._set_has_inputs(True)
        if not self.initialized:
            self._set_initialized(True)

        return snapshot

    # -- Core logic (shared by both APIs) ------------------------------------

    def _compute(
        self,
        ts_event: int,
        mid_price: float,
        bid_size: float,
        ask_size: float,
        alpha: float,
    ) -> ReversalScoreSnapshot:
        self._mid_history.append((ts_event, mid_price))
        self._trim_history(ts_event)

        imbalance = self._compute_imbalance(bid_size, ask_size)
        imbalance_strength = abs(imbalance)

        ret_short = self._return_over_horizon(ts_event, mid_price, self.config.short_horizon_ns)
        ret_long = self._return_over_horizon(ts_event, mid_price, self.config.long_horizon_ns)

        if ret_short is not None and ret_long is not None:
            self._ret_pairs.append((ret_short, ret_long))

        autocorr_score = self._compute_autocorr_score()
        disagreement = self._compute_disagreement(alpha, imbalance)

        score = (
            self.config.imbalance_weight * imbalance_strength
            + self.config.autocorr_weight * (1.0 - autocorr_score)
            + self.config.disagreement_weight * disagreement
        )
        score = float(np.clip(score, 0.0, 1.0))

        return ReversalScoreSnapshot(
            score=score,
            imbalance=imbalance,
            imbalance_strength=imbalance_strength,
            autocorr_score=autocorr_score,
            disagreement=disagreement,
            ret_short=ret_short or 0.0,
            ret_long=ret_long or 0.0,
        )

    def _trim_history(self, ts_event: int) -> None:
        min_ts = ts_event - self.config.history_ns
        while self._mid_history and self._mid_history[0][0] < min_ts:
            self._mid_history.popleft()

    def _compute_imbalance(self, bid_size: float, ask_size: float) -> float:
        total = bid_size + ask_size
        if total <= 0:
            return 0.0
        return float((bid_size - ask_size) / total)

    def _return_over_horizon(
        self,
        ts_event: int,
        current_mid: float,
        horizon_ns: int,
    ) -> float | None:
        target_ts = ts_event - horizon_ns
        past_mid = None

        for hist_ts, hist_mid in reversed(self._mid_history):
            if hist_ts <= target_ts:
                past_mid = hist_mid
                break

        if past_mid is None or past_mid <= 0:
            return None
        return float(current_mid / past_mid - 1.0)

    def _compute_autocorr_score(self) -> float:
        if len(self._ret_pairs) < 3:
            return 0.5

        short_rets = np.array([pair[0] for pair in self._ret_pairs], dtype=float)
        long_rets = np.array([pair[1] for pair in self._ret_pairs], dtype=float)

        if np.std(short_rets) < 1e-12 or np.std(long_rets) < 1e-12:
            return 0.5

        corr = float(np.corrcoef(short_rets, long_rets)[0, 1])
        if np.isnan(corr):
            return 0.5

        return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

    def _compute_disagreement(self, alpha: float, imbalance: float) -> float:
        if abs(alpha) <= self.config.alpha_threshold:
            return 0.0
        if abs(imbalance) <= 1e-12:
            return 0.0
        return 1.0 if np.sign(alpha) != np.sign(imbalance) else 0.0
