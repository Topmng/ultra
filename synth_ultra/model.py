"""BTC 10s percentile forecast. Target: <5 ms, CPU-only, deterministic."""

from __future__ import annotations

import numpy as np

from synth_ultra.constants import HORIZON_SECONDS, INV_NORM, NUM_PERCENTILES
from synth_ultra.features import extract

_EPS = 1e-12
_MIN_PRICE = 1e-8
_MIN_SIGMA = 3e-5
_MAX_SIGMA = 2e-2
_WIDTH_SCALE = 10 ** 0.5


def _sanitize(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).reshape(NUM_PERCENTILES)
    out = np.nan_to_num(out, nan=_MIN_PRICE, posinf=1e16, neginf=_MIN_PRICE)
    out = np.maximum(out, _MIN_PRICE)
    np.maximum.accumulate(out, out=out)
    diffs = np.diff(out)
    if np.any(diffs <= 0.0):
        out[1:] += np.cumsum(np.where(diffs <= 0.0, _EPS, 0.0))
    return out


def predict_percentiles(payload: dict) -> np.ndarray:
    """Return 100 centered-quantile prices for BTC 10s ahead.

    Output constraints (SPECIFICATION.md): shape (100,), float64, finite,
    strictly positive, non-decreasing. Quantile grid q_i = (2i-1)/200.
    """
    f = extract(payload)
    px = max(f["price"], _MIN_PRICE)

    vol_10s = f["vol_1s"] * np.sqrt(float(HORIZON_SECONDS))
    sigma = 0.85 * vol_10s + 0.15 * max(f["spread_rel"], 0.0)
    if f["stale_ms"] > 250.0:
        sigma *= 1.0 + min(f["stale_ms"] / 1000.0, 1.0)
    sigma = float(np.clip(sigma, _MIN_SIGMA, _MAX_SIGMA)) * _WIDTH_SCALE

    mu = (
        0.18 * f["top_obi"] * sigma
        + 0.12 * f["obi"] * sigma
        + 0.16 * f["flow_fast"] * sigma
        + 0.08 * f["flow"] * sigma
        + 0.22 * f["momentum"]
        + 0.10 * np.clip(f["basis"], -0.001, 0.001)
    )
    mu = float(np.clip(mu, -4.0 * sigma, 4.0 * sigma))

    log_px = np.log(px)
    out = np.exp(log_px + mu + sigma * INV_NORM, dtype=np.float64)
    return _sanitize(out)
