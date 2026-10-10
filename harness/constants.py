"""Local schema and scoring constants. Not copied into the miner image.

Inference only needs ``NUM_PERCENTILES`` and ``HORIZON_SECONDS``, which stay in
``synth_ultra.constants``.
"""

from __future__ import annotations

import numpy as np

from synth_ultra.constants import HORIZON_SECONDS, NUM_PERCENTILES

LATENCY_BUDGET_S = 0.005
QUANTILE_GRID = (2.0 * np.arange(1, NUM_PERCENTILES + 1) - 1.0) / 200.0
DEPTH_LEVELS = 20
CANDLE_WINDOW_S = 3600
TRADE_WINDOW_S = 60
BOOK_TICKER_WINDOW_S = 60
LIQUIDATION_WINDOW_S = 3600
DEPTH_BANDS = 100
DEPTH_BAND_USD = 10.0
# FAQ.md: time prompts on wall-clock :00, :20, :40. The event window opens
# 10s later and stays open for 5s. Half of the accepted moves are delayed
# by uniform(0, 500 ms); current_time_ms stays the move's receive time.
CLOCK_INTERVAL_MS = 20_000
EVENT_WINDOW_OPEN_MS = 10_000
EVENT_WINDOW_LEN_MS = 5_000
EVENT_DELAY_MAX_MS = 500
SCHEMA_VERSION = 5
# input.md: prompt.trigger.kind. `venue` is None on a clock trigger.
TRIGGER_KINDS = ("time", "event", "event_delayed")
TRIGGER_VENUES = ("spot", "futures")
# SPECIFICATION.md drops a prompt when the spot book is not live around the
# target instant. The feed normally updates many times per second (FAQ.md),
# so a gap longer than this means that instant has no live book.
SPOT_FEED_LIVE_MS = 1000


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """Acklam's rational approximation to the standard normal quantile function."""
    p = np.asarray(p, dtype=np.float64)
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577509590705e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    phigh = 1.0 - plow
    x = np.empty_like(p)

    low = p < plow
    high = p > phigh
    mid = ~(low | high)

    if np.any(low):
        q = np.sqrt(-2.0 * np.log(p[low]))
        x[low] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if np.any(high):
        q = np.sqrt(-2.0 * np.log(1.0 - p[high]))
        x[high] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if np.any(mid):
        q = p[mid] - 0.5
        r = q * q
        x[mid] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return x


INV_NORM = _norm_ppf(QUANTILE_GRID)
_U = QUANTILE_GRID - 0.5
LAPLACE_Z = -np.sign(_U) * np.log(np.maximum(1.0 - 2.0 * np.abs(_U), 1e-12))

__all__ = [
    "BOOK_TICKER_WINDOW_S",
    "CANDLE_WINDOW_S",
    "CLOCK_INTERVAL_MS",
    "DEPTH_BAND_USD",
    "DEPTH_BANDS",
    "DEPTH_LEVELS",
    "EVENT_DELAY_MAX_MS",
    "EVENT_WINDOW_LEN_MS",
    "EVENT_WINDOW_OPEN_MS",
    "HORIZON_SECONDS",
    "INV_NORM",
    "LAPLACE_Z",
    "LATENCY_BUDGET_S",
    "LIQUIDATION_WINDOW_S",
    "NUM_PERCENTILES",
    "QUANTILE_GRID",
    "SCHEMA_VERSION",
    "SPOT_FEED_LIVE_MS",
    "TRADE_WINDOW_S",
    "TRIGGER_KINDS",
    "TRIGGER_VENUES",
]
