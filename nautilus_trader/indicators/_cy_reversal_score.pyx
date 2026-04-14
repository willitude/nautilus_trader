# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython-optimized ReversalScore indicator.

Same logic as reversal_score.py but with cdef methods and C-level math
for ~10-30x faster tick-by-tick processing in feature_engineer.py.
"""
from libc.math cimport fabs, sqrt, NAN, isnan

from nautilus_trader.indicators.base cimport Indicator
from nautilus_trader.model.data cimport QuoteTick, TradeTick, Bar

from collections import deque
from dataclasses import dataclass


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


@dataclass(frozen=True)
class ReversalScoreSnapshot:
    score: float
    imbalance: float
    imbalance_strength: float
    autocorr_score: float
    disagreement: float
    ret_short: float
    ret_long: float


cdef class ReversalScore(Indicator):
    """Cython NautilusTrader Indicator: toxic-flow / reversal proxy."""

    cdef:
        object config
        object _mid_history      # deque of (int64, double) tuples
        object _ret_pairs        # deque of (double, double) tuples
        double _alpha

        # Config cached as C types for hot-path access
        long long _short_horizon_ns
        long long _long_horizon_ns
        long long _history_ns
        int _autocorr_window
        double _alpha_threshold
        double _imbalance_weight
        double _autocorr_weight
        double _disagreement_weight

    cdef public object latest_snapshot

    def __init__(self, config=None):
        if config is None:
            config = ReversalScoreConfig()
        self.config = config
        Indicator.__init__(self, params=[
            config.short_horizon_ns,
            config.long_horizon_ns,
            config.alpha_threshold,
        ])

        self._short_horizon_ns = config.short_horizon_ns
        self._long_horizon_ns = config.long_horizon_ns
        self._history_ns = config.history_ns
        self._autocorr_window = config.autocorr_window
        self._alpha_threshold = config.alpha_threshold
        self._imbalance_weight = config.imbalance_weight
        self._autocorr_weight = config.autocorr_weight
        self._disagreement_weight = config.disagreement_weight

        self._mid_history = deque()
        self._ret_pairs = deque(maxlen=self._autocorr_window)
        self._alpha = 0.0
        self.latest_snapshot = ReversalScoreSnapshot(
            score=0.0, imbalance=0.0, imbalance_strength=0.0,
            autocorr_score=0.5, disagreement=0.0, ret_short=0.0, ret_long=0.0,
        )

    def set_alpha(self, double alpha):
        self._alpha = alpha

    # -- Nautilus Indicator interface -------------------------------------------

    cpdef void handle_quote_tick(self, QuoteTick tick):
        cdef:
            long long ts_event = <long long>tick.ts_event
            double bid = tick.bid_price.as_double()
            double ask = tick.ask_price.as_double()
            double mid_price = (bid + ask) * 0.5
            double bid_size = tick.bid_size.as_double()
            double ask_size = tick.ask_size.as_double()

        self.latest_snapshot = self._compute_c(ts_event, mid_price, bid_size, ask_size, self._alpha)

        if not self.has_inputs:
            self._set_has_inputs(True)
        if not self.initialized:
            self._set_initialized(True)

    cpdef void handle_trade_tick(self, TradeTick tick):
        pass

    cpdef void handle_bar(self, Bar bar):
        pass

    cpdef void _reset(self):
        self._mid_history.clear()
        self._ret_pairs.clear()
        self._alpha = 0.0
        self.latest_snapshot = ReversalScoreSnapshot(
            score=0.0, imbalance=0.0, imbalance_strength=0.0,
            autocorr_score=0.5, disagreement=0.0, ret_short=0.0, ret_long=0.0,
        )

    # -- Legacy API (backward-compatible) ------------------------------------

    def update(self, long long ts_event, double mid_price, double bid_size, double ask_size, double alpha):
        self._alpha = alpha
        cdef object snapshot = self._compute_c(ts_event, mid_price, bid_size, ask_size, alpha)
        self.latest_snapshot = snapshot
        if not self.has_inputs:
            self._set_has_inputs(True)
        if not self.initialized:
            self._set_initialized(True)
        return snapshot

    # -- Core logic (all cdef for C-speed) -----------------------------------

    cdef object _compute_c(self, long long ts_event, double mid_price,
                            double bid_size, double ask_size, double alpha):
        cdef:
            double imbalance, imbalance_strength, total
            double ret_short_val, ret_long_val
            int has_short, has_long
            double autocorr_score, disagreement, score

        self._mid_history.append((ts_event, mid_price))
        self._trim_c(ts_event)

        total = bid_size + ask_size
        if total <= 0.0:
            imbalance = 0.0
        else:
            imbalance = (bid_size - ask_size) / total
        imbalance_strength = fabs(imbalance)

        has_short = self._return_over_horizon_c(ts_event, mid_price, self._short_horizon_ns, &ret_short_val)
        has_long = self._return_over_horizon_c(ts_event, mid_price, self._long_horizon_ns, &ret_long_val)

        if has_short and has_long:
            self._ret_pairs.append((ret_short_val, ret_long_val))

        autocorr_score = self._autocorr_c()
        disagreement = self._disagreement_c(alpha, imbalance)

        score = (self._imbalance_weight * imbalance_strength
                 + self._autocorr_weight * (1.0 - autocorr_score)
                 + self._disagreement_weight * disagreement)
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0

        return ReversalScoreSnapshot(
            score=score,
            imbalance=imbalance,
            imbalance_strength=imbalance_strength,
            autocorr_score=autocorr_score,
            disagreement=disagreement,
            ret_short=ret_short_val if has_short else 0.0,
            ret_long=ret_long_val if has_long else 0.0,
        )

    cdef void _trim_c(self, long long ts_event):
        cdef long long min_ts = ts_event - self._history_ns
        while self._mid_history and (<long long>self._mid_history[0][0]) < min_ts:
            self._mid_history.popleft()

    cdef int _return_over_horizon_c(self, long long ts_event, double current_mid,
                                     long long horizon_ns, double *out) noexcept:
        cdef:
            long long target_ts = ts_event - horizon_ns
            long long hist_ts
            double hist_mid
        out[0] = 0.0
        for item in reversed(self._mid_history):
            hist_ts = <long long>item[0]
            if hist_ts <= target_ts:
                hist_mid = <double>item[1]
                if hist_mid > 0.0:
                    out[0] = current_mid / hist_mid - 1.0
                    return 1
                return 0
        return 0

    cdef double _autocorr_c(self):
        cdef:
            int n = len(self._ret_pairs)
            double sum_s = 0.0, sum_l = 0.0
            double sum_ss = 0.0, sum_ll = 0.0, sum_sl = 0.0
            double mean_s, mean_l, var_s, var_l, std_s, std_l, corr
            double s, l
            int i

        if n < 3:
            return 0.5

        for i in range(n):
            s = <double>self._ret_pairs[i][0]
            l = <double>self._ret_pairs[i][1]
            sum_s += s
            sum_l += l
            sum_ss += s * s
            sum_ll += l * l
            sum_sl += s * l

        mean_s = sum_s / n
        mean_l = sum_l / n
        var_s = sum_ss / n - mean_s * mean_s
        var_l = sum_ll / n - mean_l * mean_l

        if var_s < 1e-24 or var_l < 1e-24:
            return 0.5

        std_s = sqrt(var_s)
        std_l = sqrt(var_l)
        corr = (sum_sl / n - mean_s * mean_l) / (std_s * std_l)

        if isnan(corr):
            return 0.5

        corr = (corr + 1.0) * 0.5
        if corr < 0.0:
            corr = 0.0
        elif corr > 1.0:
            corr = 1.0
        return corr

    cdef double _disagreement_c(self, double alpha, double imbalance):
        if fabs(alpha) <= self._alpha_threshold:
            return 0.0
        if fabs(imbalance) <= 1e-12:
            return 0.0
        if (alpha > 0.0 and imbalance < 0.0) or (alpha < 0.0 and imbalance > 0.0):
            return 1.0
        return 0.0
