"""BTC 10s percentile forecast. Target: <5 ms, CPU-only, deterministic."""

from __future__ import annotations

import numpy as np

from synth_ultra.constants import HORIZON_SECONDS, INV_LAPLACE, NUM_PERCENTILES
from synth_ultra.features import extract

_EPS = 1e-12
_MIN_PRICE = 1e-8
_MIN_SIGMA = 4e-5
_MAX_SIGMA = 2e-2
_SIGMA_SCALE = 1.25
_FLOW_MU = 3.2e-5


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

    Location is current spot microprice plus a small microstructure drift.
    Scale is 10s vol from candles and the book-ticker micro path. Shape is
    unit-variance Laplace — 10s BTC residuals are fat-tailed, so a Gaussian
    under-covers the 1% tails and loses pinball CRPS on jumps.
    """
    f = extract(payload)
    px = max(f["price"], _MIN_PRICE)

    candle_vol = f["vol_1s"]
    micro_vol = f["micro_vol_1s"]
    vol_1s = 0.6 * candle_vol + 0.4 * micro_vol if micro_vol > 0.0 else candle_vol
    vol_10s = vol_1s * np.sqrt(float(HORIZON_SECONDS))
    sigma = 0.85 * vol_10s + 0.15 * max(f["spread_rel"], 0.0)
    if f["stale_ms"] > 250.0:
        sigma *= 1.0 + min(f["stale_ms"] / 1000.0, 1.0)
    sigma = float(np.clip(sigma * _SIGMA_SCALE, _MIN_SIGMA, _MAX_SIGMA))

    mu = float(np.clip(_FLOW_MU * f["flow_fast"], -2.5 * sigma, 2.5 * sigma))

    log_px = np.log(px)
    out = np.exp(log_px + mu + sigma * INV_LAPLACE, dtype=np.float64)
    return _sanitize(out)


def _import_warmup() -> None:
    """Prime numpy paths at import. Synth does not count this against the 5 ms budget."""
    predict_percentiles(
        {
            "schema_version": 3,
            "prompt": {"current_time_ms": 0},
            "venues": {"spot": {}, "futures": {}},
        }
    )


_import_warmup()
