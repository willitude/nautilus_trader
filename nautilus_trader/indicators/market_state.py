"""Market State Indicator — SSOT for base market features.

Computes spread, imbalance, microprice deviation, OFI, short-term
volatility, and momentum from raw QuoteTicks.  Runs identically in
live and backtest modes (Nautilus Indicator contract).

Cross-exchange basis is injected externally via ``set_cross_basis()``
and ``set_btc_cross_basis()`` because it requires data from two
different exchanges — the same pattern used by BasisMomentumIndicator.

All formulas use microprice (size-weighted) as the reference price
to stay consistent with the offline ``engineer.py`` pipeline.
"""
from __future__ import annotations

import math
from collections import deque

from nautilus_trader.indicators import Indicator
from nautilus_trader.model.data import QuoteTick

from nautilus_trader.indicators.snapshots import MarketStateConfig
from nautilus_trader.indicators.snapshots import MarketStateSnapshot


def _microprice_dev_scalar(microprice: float, mid: float) -> float:
    if mid <= 0:
        return 0.0
    return (microprice - mid) / mid

_ZERO_SNAPSHOT = MarketStateSnapshot(
    mid=0.0, microprice=0.0, spread_rel=0.0, imbalance_l1=0.0,
    microprice_dev=0.0, ofi=0.0, volatility_5s=0.0, momentum_5s=0.0,
    cross_basis=0.0, btc_cross_basis=0.0,
)


class MarketStateIndicator(Indicator):
    """NautilusTrader Indicator: base market-state features from L1 quotes.

    Usage (Actor or Strategy)::

        ind = MarketStateIndicator()
        # each tick:
        ind.set_cross_basis(computed_basis)
        ind.set_btc_cross_basis(computed_btc_basis)
        ind.handle_quote_tick(tick)
        snapshot = ind.snapshot
    """

    def __init__(self, config: MarketStateConfig | None = None) -> None:
        cfg = config or MarketStateConfig()
        super().__init__(params=[cfg.window_ns])
        self._window_ns = cfg.window_ns

        self._prev_bid: float = 0.0
        self._prev_ask: float = 0.0
        self._ofi_initialized: bool = False
        self._mid_history: deque[tuple[int, float]] = deque(maxlen=cfg.mid_history_maxlen)

        self._cross_basis: float = 0.0
        self._btc_cross_basis: float = 0.0

        self.snapshot: MarketStateSnapshot = _ZERO_SNAPSHOT

    # -- External signal injection (cross-exchange, computed by Actor) ------

    def set_cross_basis(self, value: float) -> None:
        self._cross_basis = value

    def set_btc_cross_basis(self, value: float) -> None:
        self._btc_cross_basis = value

    # -- Nautilus Indicator interface ----------------------------------------

    def handle_quote_tick(self, tick: QuoteTick) -> None:
        bid = tick.bid_price.as_double()
        ask = tick.ask_price.as_double()
        bid_size = tick.bid_size.as_double()
        ask_size = tick.ask_size.as_double()
        ts = int(tick.ts_event)

        mid = (bid + ask) / 2.0
        spread = ask - bid
        spread_rel = spread / mid if mid > 0 else 0.0

        total = bid_size + ask_size
        imbalance_l1 = (bid_size - ask_size) / total if total > 0 else 0.0

        if total > 0:
            microprice = (bid * ask_size + ask * bid_size) / total
        else:
            microprice = mid
        microprice_dev = _microprice_dev_scalar(microprice, mid)

        if self._ofi_initialized:
            ofi = (bid - self._prev_bid) - (ask - self._prev_ask)
        else:
            ofi = 0.0
            self._ofi_initialized = True
        self._prev_bid = bid
        self._prev_ask = ask

        self._mid_history.append((ts, mid))
        cutoff = ts - self._window_ns
        while self._mid_history and self._mid_history[0][0] < cutoff:
            self._mid_history.popleft()

        volatility_5s = 0.0
        momentum_5s = 0.0
        if len(self._mid_history) >= 2:
            old_mid = self._mid_history[0][1]
            if old_mid > 0:
                momentum_5s = math.log(mid / old_mid)
                rets: list[float] = []
                prev_m = self._mid_history[0][1]
                for _, m in list(self._mid_history)[1:]:
                    if prev_m > 0:
                        rets.append(math.log(m / prev_m))
                    prev_m = m
                if rets:
                    mean_r = sum(rets) / len(rets)
                    var = sum((r - mean_r) ** 2 for r in rets) / len(rets)
                    volatility_5s = math.sqrt(var) if var > 0 else 0.0

        self.snapshot = MarketStateSnapshot(
            mid=mid,
            microprice=microprice,
            spread_rel=spread_rel,
            imbalance_l1=imbalance_l1,
            microprice_dev=microprice_dev,
            ofi=ofi,
            volatility_5s=volatility_5s,
            momentum_5s=momentum_5s,
            cross_basis=self._cross_basis,
            btc_cross_basis=self._btc_cross_basis,
        )

        if not self.has_inputs:
            self._set_has_inputs(True)
        if not self.initialized:
            self._set_initialized(True)

    def handle_trade_tick(self, tick) -> None:
        pass

    def handle_bar(self, bar) -> None:
        pass

    def _reset(self) -> None:
        self._prev_bid = 0.0
        self._prev_ask = 0.0
        self._ofi_initialized = False
        self._mid_history.clear()
        self._cross_basis = 0.0
        self._btc_cross_basis = 0.0
        self.snapshot = _ZERO_SNAPSHOT
