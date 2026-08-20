"""Competition constants from SPECIFICATION.md and input.md."""

from __future__ import annotations

import numpy as np

NUM_PERCENTILES = 100
HORIZON_SECONDS = 10
LATENCY_BUDGET_S = 0.005
QUANTILE_GRID = (2.0 * np.arange(1, NUM_PERCENTILES + 1) - 1.0) / 200.0
DEPTH_LEVELS = 20
CANDLE_WINDOW_S = 3600
TRADE_WINDOW_S = 60
BOOK_TICKER_WINDOW_S = 60
SCHEMA_VERSION = 3


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
