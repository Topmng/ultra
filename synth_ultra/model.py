"""BTC 10s percentile forecast. Target: <5 ms, CPU-only, deterministic.

Price-space Laplace jump-mixture (core + heavy tails). Location is linear in
book and trade features. Scale starts from that fit, then widens toward the
mean absolute 10-second move in the payload's own candle history whenever the
fit is tighter than the market just was. Never-raise fallback so a non-answer
is not charged under FAQ.md.
"""

from __future__ import annotations

import gc

import numpy as np

from synth_ultra.constants import HORIZON_SECONDS, NUM_PERCENTILES
from synth_ultra.features import extract

_EPS = 1e-12
_MIN_PRICE = 1e-8
_MIN_SIGMA_PX = 0.5
_MAX_SIGMA_PX = 800.0

# CRPS-tuned coeffs (saved-pair fit). See feature order in _mu_vec / _abs_vec.
_B_MU = np.array(
    [
        0.16219725476591365,
        3.0392548397260206e-05,
        1.2921967376416503e-05,
        0.17545357593694502,
        -0.057379556931711054,
        -0.042592960514635785,
        0.14437278080879343,
        -5.040059876530939e-06,
        -2.9598223746399192e-05,
        8.80503231208016e-07,
        6.543536873659752e-06,
        3.819089008691624e-05,
        -3.851394849050198e-05,
        0.012819614731266383,
        -2.5729505064060203,
        7.80781830077767e-05,
        0.2225417214564695,
        -0.09424387369323395,
        0.06611178032601009,
        -0.06323517171109437,
        1.3194603589745686e-05,
        9.25776900504624e-06,
        6.419523696500204e-06,
        9.704432289751613e-06,
        -7.61375359503243e-05,
        8.219239603035196e-09,
    ],
    dtype=np.float64,
)
_B_ABS = np.array(
    [
        0.31128035091375017,
        -0.04612335008069495,
        -0.27015375057439384,
        0.05338090769963195,
        -0.10630344326911181,
        0.04195075214608767,
        0.48300301336701507,
        0.14988622199220886,
        0.05522149798895418,
        0.28790522175963235,
        -0.2171458492361658,
        7.440005212080741e-06,
        1.5006480516698034e-05,
        5.258077361233235e-06,
        -3.602539619772853,
        -2.7531539341820717e-05,
        0.00029563500111100656,
        7.860095041094238e-05,
        0.018052856090945528,
        -0.5633117688000335,
        -0.09328221925825791,
        0.11197666410197983,
        -0.19790213408961582,
        0.3465817220321161,
        -5.886429731585097e-08,
        -1.194041355427959e-06,
        1.1416268283201004e-06,
        1.3031647192685995e-06,
        -5.381035088350931e-07,
        -0.0004201840209259798,
        -1.4343343808409522e-05,
    ],
    dtype=np.float64,
)
_KA = 0.75
_KB = 0.20
_FL = 5e-5
_SH = 1.20
_FR = 0.15
_FV = 0.05
_SM = 0.90

# Standardized quantiles of (1-w)*Laplace(0,1) + w*Laplace(0,r), w=0.15, r=3.
# Precomputed so inference stays numpy-only (no scipy in the miner image).
_MIX_Z = np.array(
    [
        -8.195296766019496,
        -5.290455382749503,
        -4.189473492373635,
        -3.558928761226856,
        -3.130018036741951,
        -2.809478451891556,
        -2.5554904666449834,
        -2.3461068991409015,
        -2.1685150645304785,
        -2.014637822514757,
        -1.879079631466685,
        -1.7580695625741907,
        -1.6488748569423617,
        -1.5494554430067913,
        -1.4582502010738922,
        -1.3740392432906956,
        -1.2958520713528672,
        -1.2229045008325568,
        -1.1545542250063079,
        -1.0902688057552605,
        -1.0296021595823457,
        -0.9721769807684978,
        -0.9176713964804792,
        -0.865808692117394,
        -0.8163492998117764,
        -0.7690844793903884,
        -0.7238312817274705,
        -0.6804284955014599,
        -0.6387333564233322,
        -0.5986188536579183,
        -0.559971508386832,
        -0.5226895288992309,
        -0.48668126838854353,
        -0.4518639279418573,
        -0.4181624595326374,
        -0.3855086332303556,
        -0.3538402400763217,
        -0.3231004076881543,
        -0.29323701004778674,
        -0.26420215637718525,
        -0.2359517467597128,
        -0.2084450843380587,
        -0.1816445356867318,
        -0.15551523237167816,
        -0.1300248078599066,
        -0.10514316488530429,
        -0.0808422691479815,
        -0.05709596585577331,
        -0.03387981614877673,
        -0.01117095088164434,
        0.01117095088164428,
        0.03387981614877675,
        0.05709596585577319,
        0.08084226914798152,
        0.10514316488530434,
        0.13002480785990664,
        0.1555152323716778,
        0.18164453568673153,
        0.20844508433805856,
        0.23595174675971262,
        0.26420215637718536,
        0.29323701004778674,
        0.323100407688154,
        0.3538402400763216,
        0.38550863323035545,
        0.41816245953263753,
        0.45186392794185726,
        0.48668126838854336,
        0.5226895288992306,
        0.5599715083868319,
        0.5986188536579181,
        0.6387333564233322,
        0.6804284955014599,
        0.7238312817274708,
        0.7690844793903882,
        0.8163492998117761,
        0.8658086921173938,
        0.9176713964804795,
        0.9721769807684977,
        1.029602159582346,
        1.0902688057552605,
        1.1545542250063072,
        1.2229045008325565,
        1.2958520713528667,
        1.374039243290695,
        1.458250201073892,
        1.5494554430067913,
        1.6488748569423621,
        1.758069562574191,
        1.8790796314666847,
        2.0146378225147576,
        2.1685150645304794,
        2.346106899140902,
        2.5554904666449842,
        2.8094784518915548,
        3.130018036741951,
        3.558928761226851,
        4.189473492373635,
        5.290455382749499,
        8.195296766019453,
    ],
    dtype=np.float64,
)
# Mean absolute deviation of the mixture quantile grid. A price scale of `s`
# then has the same mean absolute deviation as `s * _MIX_Z`.
_Z_ABS = float(np.mean(np.abs(_MIX_Z)))


def _sanitize(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).reshape(NUM_PERCENTILES)
    out = np.nan_to_num(out, nan=_MIN_PRICE, posinf=1e16, neginf=_MIN_PRICE)
    out = np.maximum(out, _MIN_PRICE)
    np.maximum.accumulate(out, out=out)
    diffs = np.diff(out)
    if np.any(diffs <= 0.0):
        out[1:] += np.cumsum(np.where(diffs <= 0.0, _EPS, 0.0))
    return out


def _realized_mix_scale(close: np.ndarray, px: float) -> float:
    """Mixture scale whose mean absolute deviation matches recent 10s moves.

    Uses whatever candle history the payload actually carries, so the floor
    tracks the current regime instead of a fixed volatility constant.
    """
    if close.size < 40:
        return 0.0
    c = np.maximum(np.asarray(close, dtype=np.float64), 1e-12)
    step = int(HORIZON_SECONDS)
    moved = np.log(c[step:] / c[:-step])
    if moved.size < 30:
        return 0.0
    mean_abs = float(np.mean(np.abs(moved - np.median(moved))))
    if mean_abs <= 0.0 or not np.isfinite(mean_abs):
        return 0.0
    return float(px) * mean_abs / _Z_ABS


def _spot_closes(payload: dict) -> np.ndarray:
    ohlcv = (((payload.get("venues") or {}).get("spot") or {}).get("candles_1s") or {}).get("ohlcv")
    if ohlcv is None:
        return np.empty(0, dtype=np.float64)
    arr = ohlcv if isinstance(ohlcv, np.ndarray) else np.asarray(ohlcv, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 4:
        return np.empty(0, dtype=np.float64)
    return arr[:, 3]


def _blend_scale(fitted: float, realized: float) -> float:
    """Raise a too-tight fitted scale toward the realized-move scale.

    When the fitted scale is already wider, keep it: live bursts should be
    able to fatten the distribution immediately. The 0.8 power closes most of
    the gap and is the holdout minimum between 'ignore the hour' and 'replace
    the fitted scale outright'.
    """
    if not np.isfinite(realized) or realized <= fitted or fitted <= 0.0:
        return fitted
    return float(fitted * (realized / fitted) ** 0.8)


def _ladder(px: float, mu_px: float, sigma_px: float) -> np.ndarray:
    """Price-space mixture percentiles: px + mu + sigma * mix_z(tau)."""
    out = float(px) + float(mu_px) + float(sigma_px) * _MIX_Z
    return _sanitize(out)


def _mu_vec(f: dict[str, float], vol10: float) -> np.ndarray:
    imb = 0.5 * (f["s_imb"] + f["f_imb"])
    flow = 0.5 * (f["flow_fast"] + f["f_flow2"])
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
            imb,
            flow,
            imb * vol10,
            flow * vol10,
            np.tanh(f["c1"] * 3000.0),
            f["basis"] * vol10,
        ],
        dtype=np.float64,
    )


def _abs_vec(f: dict[str, float], vol10: float, vol_fast10: float) -> np.ndarray:
    rng = max(f["s_rng"], f["f_rng"])
    rv = max(f["s_rv"], f["f_rv"])
    log_fqty = np.log1p(max(f["f_qty2"], 0.0)) / 10.0
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
            log_fqty,
            1.0,
            f["c1"],
            abs(f["c1"]),
            vol_fast10,
            abs(f["s_m3"]),
            abs(f["f_m3"]),
            abs(f["c3"]),
            rng,
            rv,
            max(vol10, vol_fast10),
            max(rng - vol10, 0.0),
            log_fqty * rng,
            f["stale_ms"] / 1000.0,
            1.0 if f["stale_ms"] > 250.0 else 0.0,
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
        vol10 = f["vol_1s"] * np.sqrt(float(HORIZON_SECONDS))
        vol_fast10 = f["vol_fast"] * np.sqrt(float(HORIZON_SECONDS))
        rng = max(f["s_rng"], f["f_rng"])
        rv = max(f["s_rv"], f["f_rv"])

        mu_ret = float(np.dot(_B_MU, _mu_vec(f, vol10)) * _SH)
        pred_abs = max(float(np.dot(_B_ABS, _abs_vec(f, vol10, vol_fast10))), 0.0)
        # Location/scale in price units (CRPS is dollar-denominated).
        mu_px = mu_ret * px
        sigma_px = (_KA * pred_abs + _KB * abs(mu_ret) + _FL) * px
        sigma_px = max(sigma_px, (_FR * rng + _FV * rv) * px)
        sigma_px = float(np.clip(sigma_px * _SM, _MIN_SIGMA_PX, _MAX_SIGMA_PX))
        sigma_px = _blend_scale(sigma_px, _realized_mix_scale(_spot_closes(payload), px))
        sigma_px = float(np.clip(sigma_px, _MIN_SIGMA_PX, _MAX_SIGMA_PX))
        mu_px = float(np.clip(mu_px, -5.0 * sigma_px, 5.0 * sigma_px))
        return _ladder(px, mu_px, sigma_px)
    except Exception:
        px = _MIN_PRICE
        try:
            venues = (payload or {}).get("venues") or {}
            spot = venues.get("spot") or {}
            book = spot.get("book_ticker") or {}
            bid = book.get("bid_price")
            ask = book.get("ask_price")
            if bid is not None and ask is not None and len(bid) and len(ask):
                bp = float(bid[-1] if not isinstance(bid, np.ndarray) else bid.reshape(-1)[-1])
                ap = float(ask[-1] if not isinstance(ask, np.ndarray) else ask.reshape(-1)[-1])
                if np.isfinite(bp) and np.isfinite(ap):
                    px = max(0.5 * (bp + ap), _MIN_PRICE)
            if px <= _MIN_PRICE:
                candles = spot.get("candles_1s") or {}
                ohlcv = candles.get("ohlcv")
                if ohlcv is not None:
                    arr = ohlcv if isinstance(ohlcv, np.ndarray) else np.asarray(ohlcv, dtype=np.float64)
                    if arr.ndim == 2 and arr.shape[0] and arr.shape[1] >= 4:
                        px = max(float(arr[-1, 3]), _MIN_PRICE)
        except Exception:
            px = _MIN_PRICE
        # Wide mixture fallback so a degraded path is not overconfident.
        return _ladder(px, 0.0, max(8.0, 1e-4 * px))


def _import_warmup() -> None:
    """Prime numpy paths at import. Synth does not count this against the 5 ms budget."""
    predict_percentiles(
        {
            "schema_version": 3,
            "prompt": {"current_time_ms": 0},
            "venues": {"spot": {}, "futures": {}},
        }
    )
    # Automatic GC pauses are a common source of multi-ms latency spikes.
    gc.collect()
    gc.disable()


_import_warmup()
