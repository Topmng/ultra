"""BTC 10s percentile forecast. Target: <5 ms, CPU-only, deterministic.

Laplace location-scale ladder with Cornish-Fisher skew, coefficients fit to
minimize pinball CRPS on saved database pairs. Never-raise fallback so a
non-answer is not charged under FAQ.md.
"""

from __future__ import annotations

import numpy as np

from synth_ultra.constants import HORIZON_SECONDS, LAPLACE_Z, NUM_PERCENTILES
from synth_ultra.features import extract

_EPS = 1e-12
_MIN_PRICE = 1e-8
_MIN_SIGMA = 2e-5
_MAX_SIGMA = 2e-2

# CRPS-tuned coeffs (saved-pair fit). See feature order in _mu_vec / _abs_vec.
_B_MU = np.array(
    [
        0.1621919418405019,
        2.3087496491810143e-05,
        1.229369687682148e-05,
        0.17545516716617593,
        -0.057379035335046284,
        -0.04259338416892133,
        0.1443709676053327,
        -3.2008585305186402e-06,
        -1.9592192261792666e-05,
        5.0426006039934125e-06,
        9.575090331414568e-06,
        1.229295582578837e-05,
        -8.295260715008355e-06,
        0.012820538129957629,
        -2.572950899550816,
        8.016922705093469e-05,
        0.22254131953857567,
        -0.09424357096906136,
        0.06611336142142803,
        -0.06323465974250335,
    ],
    dtype=np.float64,
)
_B_ABS = np.array(
    [
        0.3112795455044023,
        -0.046124149923485416,
        -0.27015317939963984,
        0.05338272789119308,
        -0.10630187950653455,
        0.0419512533220603,
        0.4830032497381303,
        0.14988666656143482,
        0.05522215787942206,
        0.287906734773655,
        -0.21714497223880505,
        1.9651266320709486e-05,
        -1.046717418160122e-05,
        2.5430565877878673e-05,
        -3.602539570706466,
        -0.00012844804630290628,
        0.00012784684025356208,
        0.00016267855991521185,
        0.018053123447344804,
        -0.5633128002994777,
        -0.09328297099808118,
        0.11197699497939505,
        -0.19790132877227598,
        0.34658171714722014,
    ],
    dtype=np.float64,
)
_KA = 0.74
_KB = 0.17
_FL = 3e-6
_SH = 1.42
_G0 = 0.97


def _sanitize(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).reshape(NUM_PERCENTILES)
    out = np.nan_to_num(out, nan=_MIN_PRICE, posinf=1e16, neginf=_MIN_PRICE)
    out = np.maximum(out, _MIN_PRICE)
    np.maximum.accumulate(out, out=out)
    diffs = np.diff(out)
    if np.any(diffs <= 0.0):
        out[1:] += np.cumsum(np.where(diffs <= 0.0, _EPS, 0.0))
    return out


def _ladder(px: float, mu: float, sigma: float, skew: float = 0.0) -> np.ndarray:
    """Log-Laplace percentiles with optional Cornish-Fisher skew."""
    z = LAPLACE_Z
    if abs(skew) > 1e-12:
        z = z + (skew / 6.0) * (z * z - 1.0)
    log_px = np.log(max(float(px), _MIN_PRICE))
    out = np.exp(log_px + float(mu) + float(sigma) * z, dtype=np.float64)
    return _sanitize(out)


def _mu_vec(f: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            f["basis"],
            f["s_imb"],
            f["f_imb"],
            f["s_m5"],
            f["f_m5"],
            f["s_m1"],
            f["f_m1"],
            f["flow_fast"],
            f["flow"],
            f["s_flow2"],
            f["f_flow2"],
            f["obi"],
            f["top_obi"],
            f["momentum"],
            f["spread_rel"],
            1.0,
            f["c1"],
            f["s_m3"],
            f["f_m3"],
            f["c3"],
        ],
        dtype=np.float64,
    )


def _abs_vec(f: dict[str, float]) -> np.ndarray:
    vol10 = f["vol_1s"] * np.sqrt(float(HORIZON_SECONDS))
    vol_fast10 = f["vol_fast"] * np.sqrt(float(HORIZON_SECONDS))
    return np.array(
        [
            vol10,
            abs(f["momentum"]),
            abs(f["basis"]),
            abs(f["s_m5"]),
            abs(f["f_m5"]),
            abs(f["s_m1"]),
            abs(f["f_m1"]),
            f["s_rv"],
            f["f_rv"],
            f["s_rng"],
            f["f_rng"],
            abs(f["s_imb"]),
            abs(f["obi"]),
            abs(f["flow_fast"]),
            f["spread_rel"],
            np.log1p(max(f["s_qty2"], 0.0)) / 10.0,
            np.log1p(max(f["f_qty2"], 0.0)) / 10.0,
            1.0,
            f["c1"],
            abs(f["c1"]),
            vol_fast10,
            abs(f["s_m3"]),
            abs(f["f_m3"]),
            abs(f["c3"]),
        ],
        dtype=np.float64,
    )


def predict_percentiles(payload: dict) -> np.ndarray:
    """Return 100 centered-quantile prices for BTC 10s ahead.

    Output constraints (SPECIFICATION.md): shape (100,), float64, finite,
    strictly positive, non-decreasing. Quantile grid q_i = (2i-1)/200.

    Never raise. FAQ.md charges a non-answer (exception, timeout, or an output
    that fails validation) the 95th percentile of models that did answer.
    """
    try:
        f = extract(payload)
        px = max(f["price"], _MIN_PRICE)

        mu = float(np.dot(_B_MU, _mu_vec(f)) * _SH)
        pred_abs = max(float(np.dot(_B_ABS, _abs_vec(f))), 0.0)
        sigma = float(np.clip(_KA * pred_abs + _KB * abs(mu) + _FL, _MIN_SIGMA, _MAX_SIGMA))
        if f["stale_ms"] > 250.0:
            sigma *= 1.0 + min(f["stale_ms"] / 1000.0, 1.0)
            sigma = float(np.clip(sigma, _MIN_SIGMA, _MAX_SIGMA))
        mu = float(np.clip(mu, -5.0 * sigma, 5.0 * sigma))
        skew = float(_G0 * np.tanh(mu / max(sigma, 1e-12)))
        return _ladder(px, mu, sigma, skew)
    except Exception:
        px = _MIN_PRICE
        try:
            venues = (payload or {}).get("venues") or {}
            spot = venues.get("spot") or {}
            book = spot.get("book_ticker") or {}
            bid = np.asarray(book.get("bid_price", []), dtype=np.float64)
            ask = np.asarray(book.get("ask_price", []), dtype=np.float64)
            if bid.size and ask.size and np.isfinite(bid[-1]) and np.isfinite(ask[-1]):
                px = max(0.5 * (float(bid[-1]) + float(ask[-1])), _MIN_PRICE)
            else:
                candles = spot.get("candles_1s") or {}
                ohlcv = np.asarray(candles.get("ohlcv", []), dtype=np.float64)
                if ohlcv.ndim == 2 and ohlcv.shape[0] and ohlcv.shape[1] >= 4:
                    px = max(float(ohlcv[-1, 3]), _MIN_PRICE)
        except Exception:
            px = _MIN_PRICE
        return _ladder(px, 0.0, _MIN_SIGMA, 0.0)


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
