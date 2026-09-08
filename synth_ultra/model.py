"""BTC 10s percentile forecast. Target: <5 ms, CPU-only, deterministic."""

from __future__ import annotations

import numpy as np

from synth_ultra.constants import HORIZON_SECONDS, LAPLACE_Z, NUM_PERCENTILES
from synth_ultra.features import extract

_EPS = 1e-12
_MIN_PRICE = 1e-8
_MIN_B = 8e-6
_MAX_B = 2e-2
_VOL_K = 0.82
_JUMP_Z = 2.0
_JUMP_MR = 0.18


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
    b = _VOL_K * vol_10s + 0.04 * max(f["spread_rel"], 0.0)
    if f["stale_ms"] > 250.0:
        b *= 1.0 + min(f["stale_ms"] / 2000.0, 0.5)
    b = float(np.clip(b, _MIN_B, _MAX_B))

    mu = (
        0.20 * f["top1_obi"] * b
        + 0.08 * f["top_obi"] * b
        + 0.10 * f["book_obi"] * b
        + 0.10 * f["book_obi_avg"] * b
        + 0.40 * f["flow1s"] * b
        + 0.08 * f["flow20"] * b
        + 0.08 * f["ret_1s"]
        + 0.55 * f["lead_1s"]
        + 0.12 * float(np.clip(f["basis"], -4e-4, 4e-4))
    )
    # Fade only still-extending jumps; skip if the last second already reversed.
    if abs(f["ret_10s"]) > _JUMP_Z * b and f["ret_1s"] * f["ret_10s"] >= 0.0:
        mu -= _JUMP_MR * f["ret_10s"]
    mu = float(np.clip(mu, -3.0 * b, 3.0 * b))

    log_px = np.log(px)
    out = np.exp(log_px + mu + b * LAPLACE_Z, dtype=np.float64)
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
