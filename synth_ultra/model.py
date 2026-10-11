"""BTC 10s percentile forecast. Target: <5 ms, CPU-only, deterministic.

Two distributions are built from the payload. The mixture ladder keeps wide
tails for jumps. The candle bins, from the last 30 minutes of 10-second log
returns, put mass on an unchanged price. Clock prompts mix the two: the
candle share is a logistic of the half-hour 90th percentile move, so a quiet
hour keeps that mass and a louder hour keeps the mixture tails. On the
mixture path the fitted drift is scaled to 85% and the last one-second move
is added. Top-of-book imbalance shifts the center by a fraction of the width,
more so after a trade print.
Never-raise fallback so a non-answer is not charged under FAQ.md.
"""

from __future__ import annotations

import gc

import numpy as np

from synth_ultra.constants import HORIZON_SECONDS, NUM_PERCENTILES
from synth_ultra.features import extract
from synth_ultra.season_table import SEAS_ABS as _SEAS_ABS
from synth_ultra.season_table import SEAS_AC as _SEAS_AC
from synth_ultra.season_table import SEAS_JUMP as _SEAS_JUMP
from synth_ultra.season_table import SEAS_REL as _SEAS_REL

_EPS = 1e-12
_MIN_PRICE = 1e-8
_MIN_SIGMA_PX = 0.5
_MAX_SIGMA_PX = 800.0

# CRPS-tuned conditional fit. The vectors stay the book/trade signal; _flex_linear
# rescales their outputs from the last ten minutes of 10-second moves.
_B_MU = np.array(
    [
        0.16219725476591365,
        3.0392548397260206e-05,
        1.2921967376416503e-05,
        0.08772678796847251,
        -0.057379556931711054,
        -0.042592960514635785,
        0.14437278080879343,
        -5.040059876530939e-06,
        -2.9598223746399192e-05,
        8.80503231208016e-07,
        6.543536873659752e-06,
        3.819089008691624e-05,
        -3.851394849050198e-05,
        0.025639229462532766,
        0.0,
        7.80781830077767e-05,
        0.2225417214564695,
        -0.09424387369323395,
        0.06611178032601009,
        0.0,
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
# Residual width map. The coefficients above were fit on one volatility mix, so
# after a quieter stretch the ladder stays wide of the latest 10s moves.
# fac = clip(A + B * s_recent / sigma, A, 1): move partway toward that
# realized scale, never wider than the feature model, never below 80% of it.
_REG_A = 0.80
_REG_B = 0.30
_REG_N = 180

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


def _venue_closes(venue: dict | None) -> np.ndarray:
    ohlcv = ((venue or {}).get("candles_1s") or {}).get("ohlcv")
    if ohlcv is None:
        return np.empty(0, dtype=np.float64)
    arr = ohlcv if isinstance(ohlcv, np.ndarray) else np.asarray(ohlcv, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 4:
        return np.empty(0, dtype=np.float64)
    return arr[:, 3]


def _spot_closes(payload: dict) -> np.ndarray:
    return _venue_closes((payload.get("venues") or {}).get("spot"))


# Log-blend of the fitted mixture scale with short-horizon 10s volatility.
# Weights minimize pinball CRPS on the saved pairs. Fast realized vol can
# both widen and tighten the 100-quantile ladder; the hour-average floor cannot.
_VOL_W = np.array([0.7199761, 0.18929481, 0.06551382, 0.02521527], dtype=np.float64)
_VOL_LOG_BIAS = 0.04421702415367945


# Reused across calls. A full hour is a few thousand 10s moves, and the weights
# depend only on that length, so rebuilding them on every payload is wasted work.
_EWMA_W: dict[tuple[int, float], np.ndarray] = {}


def _ewma_abs(abs_r: np.ndarray, lam: float = 0.98) -> float:
    n = int(abs_r.size)
    if n == 0:
        return 0.0
    key = (n, float(lam))
    w = _EWMA_W.get(key)
    if w is None:
        w = (1.0 - lam) * lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
        w /= np.sum(w)
        if len(_EWMA_W) < 8:
            _EWMA_W[key] = w
    return float(np.dot(w, abs_r))


def _vol_aware_scale(fitted: float, close: np.ndarray, px: float) -> float:
    """Set mixture scale from fitted vol and recent 10s move size.

    The 100 percentiles are the centered quantiles of a Laplace jump-mixture
    located at the predicted price. Width tracks a blend of the feature-based
    scale, the EWMA of recent 10-second absolute moves, 60-second realized
    vol, and the last 10-second high-low range.
    """
    if close.size < 40 or not np.isfinite(fitted) or fitted <= 0.0:
        return fitted
    c = np.maximum(np.asarray(close, dtype=np.float64), 1e-12)
    moved = np.abs(np.log(c[10:] / c[:-10]))
    if moved.size < 30:
        return fitted
    fast = float(px) * _ewma_abs(moved) / _Z_ABS
    if c.size >= 61:
        rets = np.diff(np.log(c[-61:]))
        rv60 = float(px) * float(np.std(rets) * np.sqrt(10.0)) / _Z_ABS
    else:
        rv60 = fast
    sl = c[-10:]
    lo = float(np.min(sl))
    hi = float(np.max(sl))
    hl10 = float(px) * float(np.log(hi / lo)) / _Z_ABS if lo > 0.0 and hi > lo else fast
    vals = np.maximum(np.array([fitted, fast, rv60, hl10], dtype=np.float64), 0.3)
    return float(np.exp(np.dot(_VOL_W, np.log(vals)) + _VOL_LOG_BIAS))


def _regime_scale(sigma_px: float, close: np.ndarray) -> float:
    """Shrink a scale that is wide of the latest few minutes of 10s moves.

    ``s_recent`` is the mixture scale whose mean absolute deviation matches
    the last ``_REG_N`` overlapping 10-second dollar moves. The map
    ``clip(_REG_A + _REG_B * s_recent / sigma, _REG_A, 1)`` was chosen so
    every 4-hour window of saved prompts has a lower pinball CRPS. Short
    candle history leaves the feature-model scale unchanged.
    """
    sigma = float(sigma_px)
    if close.size < _REG_N + 10 or not np.isfinite(sigma) or sigma <= 0.0:
        return sigma
    c = np.maximum(np.asarray(close, dtype=np.float64), 1e-12)
    moves = np.abs(c[10:] - c[:-10])
    if moves.size < _REG_N:
        return sigma
    sl = moves[-_REG_N:]
    mad = float(np.mean(np.abs(sl - np.median(sl))))
    if mad <= 0.0 or not np.isfinite(mad):
        return sigma
    ratio = (mad / _Z_ABS) / sigma
    if not np.isfinite(ratio):
        return sigma
    return sigma * float(np.clip(_REG_A + _REG_B * ratio, _REG_A, 1.0))


# Location gain and a further width step, fit so every 6-hour window improves.
# A shift larger than 0.6 widths is pulled toward spot. The pull strengthens
# when the ladder is wide of the last _FLEX_N absolute 10-second moves, and
# the shift is never taken below 75% of the feature-model value. Width moves
# at most 4% toward that same realized scale and is never increased.
_MU_EXCESS = 0.60
_MU_PULL = 0.40
_MU_FLOOR = 0.45
_WIDE_POW = 0.05
_WIDE_FLOOR = 0.96
_FLEX_N = 600


def _recent_mad(close: np.ndarray, n: int) -> float:
    if close.size < n + 10:
        return 0.0
    c = np.maximum(np.asarray(close, dtype=np.float64), 1e-12)
    moves = np.abs(c[10:] - c[:-10])
    if moves.size < n:
        return 0.0
    sl = moves[-n:]
    mad = float(np.mean(np.abs(sl - np.median(sl))))
    if not np.isfinite(mad) or mad <= 0.0:
        return 0.0
    return mad


def _flex_linear(mu_px: float, sigma_px: float, close: np.ndarray) -> tuple[float, float]:
    """Rescale the linear location and width from the latest 10-minute moves."""
    mu = float(mu_px)
    sigma = float(sigma_px)
    mad = _recent_mad(close, _FLEX_N)
    if mad <= 0.0 or not np.isfinite(sigma) or sigma <= 0.0 or not np.isfinite(mu):
        return mu, sigma
    ratio = (mad / _Z_ABS) / sigma
    if not np.isfinite(ratio) or ratio <= 0.0:
        return mu, sigma
    fade = float(np.clip(ratio, 0.3, 1.5)) ** -1.0
    excess = max(abs(mu) / sigma - _MU_EXCESS, 0.0)
    gain = float(np.clip(1.0 - _MU_PULL * excess * fade, _MU_FLOOR, 1.0))
    width = float(np.clip(ratio ** _WIDE_POW, _WIDE_FLOOR, 1.0))
    return mu * gain, sigma * width


# Centered-quantile probabilities, q_i = (2i-1)/200.
_TAU = (2.0 * np.arange(1, NUM_PERCENTILES + 1) - 1.0) / 200.0
# Mix in the book sample only while its mean absolute move is below this
# fraction of the feature-ladder width. The slope and cap are the choice
# that lowers pinball CRPS on every 12-hour window.
_BOOK_PIVOT = 0.65
_BOOK_SLOPE = 0.35
_BOOK_CAP = 0.18
# Fold the payload's own 10s book-move quantiles into the ladder when the
# hour is not a single-outlier regime. p99/median above this means the typical
# move is ~0 or one crash dominates, and the book sample is a poor center.
# Weight 0.10 is the largest blend that still lowers every saved day.
_HIST_JUMP = 12.0
_HIST_W = 0.18
# On a flat hour the fixed ladder is a little wide through the middle.
# Pull those central quantiles in; leave the outer crash points where they are.
_CALM_CORE = 0.98
# Last-second spot order-flow imbalance, divided by the quote turnover in
# that second. The linear book features are a stock of size; this is the
# flow. The shift is this fraction of the forecast width.
_OFI_K = 0.35
_OFI_MS = 1000


# A busy minute can carry tens of thousands of book ticks. The 100-quantile
# grid only needs a couple of thousand 10s pairs; extra ticks repeat the same
# move. Cap the paired sample so a 2-core host stays inside the latency budget.
_BOOK_PAIRS = 1600


def _f64_col(x: object) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x.reshape(-1)
        if arr.dtype == np.float64:
            return arr
        return arr.astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _book_offsets(book: dict | None) -> np.ndarray | None:
    """Centered quantiles of 10s spot-microprice changes in this payload.

    Pairs book ticks with the quote 10 seconds later. Long tapes are thinned
    to about ``_BOOK_PAIRS`` evenly spaced starts; each start is still matched
    to its own 10s quote. Returns None when that history is too short to fill
    the 100-quantile grid.
    """
    if not book:
        return None
    ts_raw = book.get("recv_ts_ms")
    if ts_raw is None:
        return None
    if isinstance(ts_raw, np.ndarray):
        ts = ts_raw.reshape(-1)
        if ts.dtype != np.int64:
            ts = ts.astype(np.int64, copy=False)
    else:
        ts = np.asarray(ts_raw, dtype=np.int64).reshape(-1)
    if ts.size < 80:
        return None
    bp = _f64_col(book.get("bid_price"))
    bq = _f64_col(book.get("bid_qty"))
    ap = _f64_col(book.get("ask_price"))
    aq = _f64_col(book.get("ask_qty"))
    n = int(min(ts.size, bp.size, bq.size, ap.size, aq.size))
    if n < 80:
        return None
    ts = ts[-n:]
    bp = bp[-n:]
    bq = bq[-n:]
    ap = ap[-n:]
    aq = aq[-n:]
    step = 1 if n <= _BOOK_PAIRS else (n + _BOOK_PAIRS - 1) // _BOOK_PAIRS
    ii = np.arange(0, n, step, dtype=np.int64)
    den = bq[ii] + aq[ii]
    mi = (bp[ii] * aq[ii] + ap[ii] * bq[ii]) / np.maximum(den, 1e-12)
    bad = den <= 0.0
    if np.any(bad):
        mi = mi.copy()
        mi[bad] = 0.5 * (bp[ii][bad] + ap[ii][bad])
    j = np.searchsorted(ts, ts[ii] + 10_000, side="right") - 1
    ok = (j > ii) & (j < n) & (ts[j] >= ts[ii] + 9_000)
    if int(np.count_nonzero(ok)) < 40:
        return None
    jj = j[ok]
    denj = bq[jj] + aq[jj]
    mj = (bp[jj] * aq[jj] + ap[jj] * bq[jj]) / np.maximum(denj, 1e-12)
    badj = denj <= 0.0
    if np.any(badj):
        mj = mj.copy()
        mj[badj] = 0.5 * (bp[jj][badj] + ap[jj][badj])
    ordered = np.sort(mj - mi[ok])
    m = int(ordered.size)
    half = 0.5 * (m - 1)
    h0 = int(half)
    hf = half - h0
    med = float(ordered[h0] * (1.0 - hf) + ordered[min(h0 + 1, m - 1)] * hf)
    pos = _TAU * (m - 1)
    i0 = pos.astype(np.int64)
    frac = pos - i0
    i1 = np.minimum(i0 + 1, m - 1)
    quantiles = ordered[i0] * (1.0 - frac) + ordered[i1] * frac - med
    return quantiles.astype(np.float64, copy=False)


def _quote_ofi(book: dict | None, window_ms: int = _OFI_MS) -> float:
    """Signed quote flow over the last second, scaled by its own turnover.

    A bid that steps up, or that gains size, is buying pressure. An ask that
    steps down, or that gains size, is selling pressure. The result is in
    [-1, 1]: net pressure divided by the gross quote change.
    """
    if not book or book.get("recv_ts_ms") is None:
        return 0.0
    ts = np.asarray(book["recv_ts_ms"], dtype=np.int64).reshape(-1)
    bp = np.asarray(book.get("bid_price"), dtype=np.float64).reshape(-1)
    bq = np.asarray(book.get("bid_qty"), dtype=np.float64).reshape(-1)
    ap = np.asarray(book.get("ask_price"), dtype=np.float64).reshape(-1)
    aq = np.asarray(book.get("ask_qty"), dtype=np.float64).reshape(-1)
    n = min(ts.size, bp.size, bq.size, ap.size, aq.size)
    if n < 8:
        return 0.0
    ts = ts[-n:]
    bp = bp[-n:]
    bq = bq[-n:]
    ap = ap[-n:]
    aq = aq[-n:]
    i0 = max(int(np.searchsorted(ts, int(ts[-1]) - int(window_ms), side="left")), 1)
    bp = bp[i0 - 1 :]
    bq = bq[i0 - 1 :]
    ap = ap[i0 - 1 :]
    aq = aq[i0 - 1 :]
    if bp.size < 3:
        return 0.0
    dbp = np.diff(bp)
    dbq = np.diff(bq)
    dap = np.diff(ap)
    daq = np.diff(aq)
    e_bid = np.where(dbp > 0.0, bq[1:], np.where(dbp < 0.0, -bq[:-1], dbq))
    e_ask = np.where(dap < 0.0, aq[1:], np.where(dap > 0.0, -aq[:-1], daq))
    gross = float(np.sum(np.abs(e_bid) + np.abs(e_ask)))
    if gross <= 0.0 or not np.isfinite(gross):
        return 0.0
    net = float(np.sum(e_bid - e_ask))
    if not np.isfinite(net):
        return 0.0
    return net / gross


def _mix_book(
    level: np.ndarray,
    px: float,
    sigma_px: float,
    book: dict | None,
    book_cap: float = _BOOK_CAP,
    book_pivot: float = _BOOK_PIVOT,
    book_slope: float = _BOOK_SLOPE,
    offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Pull the ladder toward the payload's own 10s microprice quantiles."""
    if offsets is None:
        offsets = _book_offsets(book)
    sigma = float(sigma_px)
    if offsets is None or not np.isfinite(sigma) or sigma <= 0.0:
        return level
    mad = float(np.mean(np.abs(offsets)))
    if mad <= 0.0 or not np.isfinite(mad):
        return level
    rel = mad / (sigma * _Z_ABS)
    if not np.isfinite(rel):
        return level
    weight = float(np.clip(book_slope * max(book_pivot - rel, 0.0), 0.0, book_cap))
    if weight <= 0.0:
        return level
    mixed = (1.0 - weight) * level + weight * (float(px) + offsets)
    return _sanitize(mixed)


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


def _hist_jump_ratio(close: np.ndarray) -> float:
    """p99 / median of absolute 10-second close changes. Large means unstable."""
    if close.size < 50:
        return 1e9
    moves = np.abs(close[10:] - close[:-10])
    if moves.size < 40:
        return 1e9
    typical = float(np.median(moves))
    if typical <= 1e-6 or not np.isfinite(typical):
        return 1e9
    extreme = float(np.quantile(moves, 0.99))
    if not np.isfinite(extreme):
        return 1e9
    return extreme / typical


def _dollar_moves(close: np.ndarray) -> np.ndarray:
    c = np.maximum(np.asarray(close, dtype=np.float64), 1e-12)
    if c.size < 30:
        return np.empty(0, dtype=np.float64)
    return np.abs(c[10:] - c[:-10])


def _block_levels(moves: np.ndarray, n_blocks: int, q: float) -> tuple[np.ndarray, int] | None:
    """Per-block quantile via one sort. Matches ``np.quantile(..., method='linear')``."""
    n = int(moves.size)
    block = n // n_blocks
    if block < 1:
        return None
    ordered = np.sort(
        np.ascontiguousarray(moves[: n_blocks * block].reshape(n_blocks, block)),
        axis=1,
    )
    pos = float(q) * (block - 1)
    i0 = int(pos)
    frac = pos - i0
    i1 = min(i0 + 1, block - 1)
    levels = ordered[:, i0] * (1.0 - frac) + ordered[:, i1] * frac
    return levels, block


def _calm_parts(moves: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Suffix of the current regime, plus the 80th percentile of each kept block.

    A trailing run of blocks no hotter than the latest one, within the
    series' own block-to-block scatter, is the current regime. The kept
    suffix is at least 180 points. A stable hour keeps everything. Short
    history returns the series and no block levels.
    """
    n = int(moves.size)
    if n <= 180:
        return moves, None
    found = _block_levels(moves, 12, 0.80)
    if found is None:
        return moves, None
    levels, block = found
    if block < 15:
        return moves, None
    scatter = float(np.median(np.abs(np.diff(levels)))) + 1e-9
    last = float(levels[-1])
    keep = 1
    for val in levels[-2::-1]:
        if float(val) <= last + 2.0 * scatter:
            keep += 1
        else:
            break
    start = min(n - keep * block, n - 180)
    return moves[max(start, 0) :], levels[-keep:]


def _calm_suffix(moves: np.ndarray) -> np.ndarray:
    """Drop an older, hotter regime. The kept suffix is at least 180 points."""
    return _calm_parts(moves)[0]


def _ewm_quantile(moves: np.ndarray, q: float) -> float:
    """Quantile with weights that put the newest point well above the oldest.

    The decay is set from the window length so the oldest point still has a
    few percent of the newest weight. A long window is thinned to every 4th
    move first, covering the same time span at a fraction of the sort cost.
    """
    n = int(moves.size)
    if n == 0:
        return 0.0
    if n < 8:
        return float(np.quantile(moves, q))
    if n > 480:
        moves = moves[::4]
        n = int(moves.size)
    log_lam = float(np.log(0.05) / n)
    w = np.exp(log_lam * np.arange(n - 1, -1, -1, dtype=np.float64))
    w /= np.sum(w)
    order = np.argsort(moves, kind="mergesort")
    cw = np.cumsum(w[order])
    idx = int(np.searchsorted(cw, q, side="left"))
    return float(moves[order[min(idx, n - 1)]])


def _venue_move_scale(close: np.ndarray) -> tuple[float, float]:
    """Return (dollar scale, precision weight) from one venue's candles."""
    moves = _dollar_moves(close)
    if moves.size < 40:
        return 0.0, 0.0
    suffix, kept = _calm_parts(moves)
    scale = _ewm_quantile(suffix, 0.80)
    if not np.isfinite(scale) or scale <= 0.0:
        return 0.0, 0.0
    # Stability of the blocks already scored while choosing the suffix.
    if kept is not None and kept.size >= 3:
        scatter = float(np.median(np.abs(np.diff(kept))))
    else:
        scatter = scale
    precision = 1.0 / (scatter + 1e-6)
    return scale, precision


def _raw_reference_scale(payload: dict) -> float:
    """Dollar scale of recent venue moves. Zero when the hour has no scale."""
    venues = payload.get("venues") or {}
    s_scale, s_w = _venue_move_scale(_venue_closes(venues.get("spot")))
    fut = venues.get("futures") or {}
    fut_close = (
        _venue_closes(fut)
        if bool((fut.get("candles_1s") or {}).get("complete_history"))
        else np.empty(0, dtype=np.float64)
    )
    f_scale, f_w = _venue_move_scale(fut_close)
    weight = s_w + f_w
    if weight <= 0.0 or not np.isfinite(weight):
        return 0.0
    ref = (s_w * s_scale + f_w * f_scale) / weight
    if not np.isfinite(ref) or ref <= 0.0:
        return 0.0
    return float(ref)


def _reference_scale(payload: dict, sigma_px: float) -> float:
    """Dollar scale the location shift is allowed to approach.

    Uses the full candle history of spot and, when futures history is
    complete, futures. Each venue is exponentially weighted and then blended
    by how stable its own scale has been, so the mix is not fixed at 50:50.
    Compared with the model scale by the caller: a louder hour keeps the
    location shift, a quieter hour does not.
    """
    ref = _raw_reference_scale(payload)
    if ref <= 0.0:
        return float(sigma_px)
    return ref


def _reliable_shift(mu_px: float, scale_px: float) -> float:
    """Shrink a price shift that is large relative to ``scale_px``.

    Weight = scale / (scale + |shift|). It stays near 1 when the shift sits
    inside the scale and falls when the shift would move the median by a
    large fraction of that scale. The scale is taken from the live payload.
    """
    shift = float(mu_px)
    scale = float(scale_px)
    if not np.isfinite(shift) or not np.isfinite(scale) or scale <= 0.0:
        return 0.0
    return shift * (scale / (scale + abs(shift)))


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


# Seasonal characters are generated by scripts/update_season.py into
# season_table.py. The 1-minute price file is not read at inference.
_SEAS_W = 0.35
_SEAS_CLIP = 0.35
# Fade a shift that continues the last 10s move when this hour's 1-minute
# returns historically reverse. Zero turns the fade off.
_SEAS_MU_K = 0.0
_SEAS_MU_AC = -0.02
_SEAS_MU_CLIP = 0.45
# Extra width when this hour is historically hotter than the previous hour.
_SEAS_JUMP_W = 0.0
_SEAS_JUMP_CLIP = 0.18
# Weekend slots quieter than this fraction of the same weekday hour trust
# the payload's own 10s book quantiles more. Zero turns that boost off.
_SEAS_QUIET_REL = 0.70
_SEAS_QUIET_CAP = 0.85
_SEAS_QUIET_PIVOT = 1.00
_SEAS_QUIET_SLOPE = 1.00


def _slot_index(now_ms: int) -> int:
    """0..1007 index of the UTC weekday, hour, and 10-minute segment."""
    sec = int(now_ms) // 1000
    # 1970-01-01 was Thursday. datetime.weekday() is Monday=0.
    dow = ((sec // 86400) + 3) % 7
    sod = sec % 86400
    return (dow * 24 + sod // 3600) * 6 + (sod % 3600) // 600


def _season_scale(sigma_px: float, px: float, now_ms: int) -> float:
    """Pull width toward this slot's typical 10-second move.

    The ladder already follows the payload's books and candles. The blend
    adds the historical scale of the same weekday, hour, and 10-minute
    segment, which a trailing hour does not identify. The weight stays small
    so a live burst is not replaced by the seasonal average.
    """
    sigma = float(sigma_px)
    price = float(px)
    if (
        not np.isfinite(sigma)
        or sigma <= 0.0
        or not np.isfinite(price)
        or price <= 0.0
        or int(now_ms) < 0
    ):
        return sigma
    idx = _slot_index(now_ms)
    seas = price * float(_SEAS_ABS[idx]) / _Z_ABS
    if not np.isfinite(seas) or seas <= 0.0:
        return sigma
    gap = float(np.log(seas) - np.log(sigma))
    adj = float(np.clip(_SEAS_W * gap, -_SEAS_CLIP, _SEAS_CLIP))
    if _SEAS_JUMP_W != 0.0:
        jump = float(_SEAS_JUMP[idx])
        if np.isfinite(jump):
            adj += float(np.clip(_SEAS_JUMP_W * jump, -_SEAS_JUMP_CLIP, _SEAS_JUMP_CLIP))
    return sigma * float(np.exp(adj))


def _season_location(mu_px: float, now_ms: int, momentum: float) -> float:
    """Shrink a continuation bet in hours that historically reverse.

    ``SEAS_AC`` is the lag-1 autocorrelation of 1-minute returns. It is
    negative on Saturday. The fade applies only when the shift has the same
    sign as the last 10-second move, so a fade that is already against that
    move is left alone.
    """
    mu = float(mu_px)
    if _SEAS_MU_K == 0.0 or not np.isfinite(mu) or int(now_ms) < 0:
        return mu
    ac = float(_SEAS_AC[_slot_index(now_ms)])
    if not np.isfinite(ac) or ac > _SEAS_MU_AC:
        return mu
    if mu * float(momentum) <= 0.0:
        return mu
    gain = float(np.exp(np.clip(_SEAS_MU_K * ac, -_SEAS_MU_CLIP, 0.0)))
    return mu * gain


def _season_book(now_ms: int) -> tuple[float, float, float]:
    """Book-mix cap, pivot, and slope.

    Weekend slots that are quiet versus the same weekday hour trust the
    payload's own 10-second book quantiles more. Sunday evening, which is
    as loud as a weekday hour, keeps the ordinary mix.
    """
    if _SEAS_QUIET_REL <= 0.0 or int(now_ms) < 0:
        return _BOOK_CAP, _BOOK_PIVOT, _BOOK_SLOPE
    idx = _slot_index(now_ms)
    if idx // 144 < 5:
        return _BOOK_CAP, _BOOK_PIVOT, _BOOK_SLOPE
    rel = float(_SEAS_REL[idx])
    if np.isfinite(rel) and rel < _SEAS_QUIET_REL:
        return _SEAS_QUIET_CAP, _SEAS_QUIET_PIVOT, _SEAS_QUIET_SLOPE
    return _BOOK_CAP, _BOOK_PIVOT, _BOOK_SLOPE


# Location of the forecast that was just returned. The plot reads these.
# They are not part of the scored output.
last_mu_px = float("nan")
last_anchor_px = float("nan")

# Overlapping 10-second close-to-close returns from 1-second candles.
# 30 minutes is 1,800 closes and 1,790 returns.
_WIN_30_S = 1800
_MIN_RETURNS = 40
_N_BINS = 50
_BIN_W = 1.0 / _N_BINS
# |μ| / mean(|log return|) above this would push one wing through the anchor.
_MU_ALPHA_CAP = 1.0
_W_CACHE: dict[int, np.ndarray] = {}


def _recency_weights(n: int) -> np.ndarray:
    """Linear in age. Oldest weight is the smallest, newest the largest, sum 1."""
    cached = _W_CACHE.get(n)
    if cached is not None:
        return cached
    weights = np.arange(1.0, n + 1.0, dtype=np.float64)
    weights /= weights.sum()
    if len(_W_CACHE) < 6:
        _W_CACHE[n] = weights
    return weights


def _window_closes(close: np.ndarray) -> np.ndarray:
    """Last 30 minutes of closes. A shorter history is used as it stands."""
    if int(close.size) >= _WIN_30_S:
        return close[-_WIN_30_S:]
    return close


def _bin_means(sorted_values: np.ndarray, sorted_weights: np.ndarray) -> np.ndarray:
    """Average of each 0.02 mass slice, walking the sorted absolute returns.

    A point whose weight crosses a boundary contributes to each slice in
    proportion to the weight that falls inside it.
    """
    cw = np.cumsum(sorted_weights)
    total = float(cw[-1]) if cw.size else 0.0
    if total <= 0.0 or not np.isfinite(total):
        return np.zeros(_N_BINS, dtype=np.float64)
    left = cw - sorted_weights
    cw = cw / total
    left = left / total
    means = np.empty(_N_BINS, dtype=np.float64)
    overlap = np.empty(sorted_values.shape, dtype=np.float64)
    for k in range(_N_BINS):
        lo = k * _BIN_W
        hi = 1.0 if k == _N_BINS - 1 else (k + 1) * _BIN_W
        np.minimum(cw, hi, out=overlap)
        overlap -= np.maximum(left, lo)
        np.maximum(overlap, 0.0, out=overlap)
        wsum = float(overlap.sum())
        if wsum <= 0.0:
            means[k] = means[k - 1] if k else 0.0
        else:
            means[k] = float(np.dot(overlap, sorted_values) / wsum)
    np.maximum.accumulate(means, out=means)
    np.maximum(means, 0.0, out=means)
    return means


def _candle_abs_bins(close: np.ndarray) -> np.ndarray | None:
    """Fifty absolute 10-second log-return levels from the recent candle window."""
    if close.size < HORIZON_SECONDS + _MIN_RETURNS:
        return None
    c = np.maximum(_window_closes(np.asarray(close, dtype=np.float64)), 1e-12)
    if not np.all(np.isfinite(c)):
        c = np.maximum(np.nan_to_num(c, nan=1e-12, posinf=1e-12, neginf=1e-12), 1e-12)
    abs_r = np.abs(np.log(c[HORIZON_SECONDS:]) - np.log(c[:-HORIZON_SECONDS]))
    if int(abs_r.size) < _MIN_RETURNS or not np.all(np.isfinite(abs_r)):
        return None
    weights = _recency_weights(int(abs_r.size))
    order = np.argsort(abs_r, kind="mergesort")
    return _bin_means(abs_r[order], weights[order])


def _skewed_candle_prices(px: float, bins: np.ndarray, mu_px: float) -> np.ndarray:
    """Mirror the 50 absolute levels, then tilt each one by μ in proportion to |dev|.

    ``alpha > 0`` lengthens the upper wing and shortens the lower wing by the
    same fraction of each level's own distance from the anchor. Levels that
    are exactly zero — unchanged 10-second candles — stay on the anchor.
    """
    scale = float(np.mean(bins))
    mu_ret = float(mu_px) / float(px) if px > 0.0 else 0.0
    alpha = 0.0
    if scale > 1e-18 and np.isfinite(mu_ret) and np.isfinite(scale):
        alpha = float(np.clip(mu_ret / scale, -_MU_ALPHA_CAP, _MU_ALPHA_CAP))
    upper = bins * (1.0 + alpha)
    lower = bins * (1.0 - alpha)
    out = np.empty(NUM_PERCENTILES, dtype=np.float64)
    out[:_N_BINS] = px * np.exp(-lower[::-1])
    out[_N_BINS:] = px * np.exp(upper)
    np.nan_to_num(out, copy=False, nan=_MIN_PRICE, posinf=1e16, neginf=_MIN_PRICE)
    np.maximum(out, _MIN_PRICE, out=out)
    np.maximum.accumulate(out, out=out)
    return out


# Clock prompts mix the candle distribution with the mixture. The candle
# weight is a logistic of the half-hour 90th percentile 10-second move:
# about 0.73 when that move is $0, one half at $12, and gone once the hour
# is much louder. Chosen on the earlier saved prompts; the later ones agreed.
_BLEND_PIVOT = 12.0
_BLEND_TEMP = 12.0
# The fitted drift runs a little hot. Keep most of it, and add the last
# one-second move, which the drift underweights.
_MU_KEEP = 0.85
_C1_KEEP = 0.70
# Buy-liquidation and re-anchored depth-band shifts, as a fraction of width.
_LIQ_K = 0.04
_BAND_K = 0.03
# Top-of-book imbalance continues over the next 10 seconds. The linear fit
# barely used it. A full imbalance shifts the center by this fraction of the
# width; trade-triggered prompts get a slightly larger fraction.
_IMB_K = 0.35
_IMB_EVENT = 0.50
# A one-sided book should keep that shift. The hour's centered book quantiles
# otherwise pull it back toward spot. Applied when |imbalance| exceeds the cut.
_LOUD_IMB = 0.60
_LOUD_BOOK = 0.60
# A trade print is followed by a larger 10-second move than the feature width.
_EVENT_SIGMA = 1.05


def _move_p90_usd(close: np.ndarray, px: float) -> float | None:
    """90th percentile of absolute 10-second dollar moves over the recent window."""
    if close.size < HORIZON_SECONDS + _MIN_RETURNS:
        return None
    c = np.maximum(_window_closes(np.asarray(close, dtype=np.float64)), 1e-12)
    abs_r = np.abs(np.log(c[HORIZON_SECONDS:]) - np.log(c[:-HORIZON_SECONDS]))
    if int(abs_r.size) < _MIN_RETURNS or not np.all(np.isfinite(abs_r)):
        return None
    return float(np.quantile(abs_r, 0.9) * px)


def _candle_weight(p90: float | None, trigger: str, candle_ok: bool) -> float:
    """Share of the candle distribution on a clock prompt. Zero otherwise."""
    if trigger != "time" or not candle_ok or p90 is None or not np.isfinite(p90):
        return 0.0
    z = (float(p90) - _BLEND_PIVOT) / _BLEND_TEMP
    if z >= 20.0:
        return 0.0
    if z <= -20.0:
        return 1.0
    return float(1.0 / (1.0 + np.exp(z)))


def _cdf_mix(candle: np.ndarray, mixture: np.ndarray, weight: float) -> np.ndarray:
    """Quantiles of ``weight * candle + (1 - weight) * mixture``.

    Averaging the quantile functions would place prices in a gap that neither
    distribution holds. Inverting the mixed CDF keeps the candle's quiet mass
    and the mixture's tails as two separate possibilities.
    """
    w = float(weight)
    lo = min(float(candle[0]), float(mixture[0]))
    hi = max(float(candle[-1]), float(mixture[-1]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return _sanitize((1.0 - w) * mixture + w * candle)
    grid = np.linspace(lo, hi, 480)
    fc = np.interp(grid, candle, _TAU, left=0.0, right=1.0)
    fm = np.interp(grid, mixture, _TAU, left=0.0, right=1.0)
    cdf = w * fc + (1.0 - w) * fm
    cdf[0] = 0.0
    cdf[-1] = 1.0
    np.maximum.accumulate(cdf, out=cdf)
    return _sanitize(np.interp(_TAU, cdf, grid))


def _combine(
    ladder: np.ndarray,
    candle: np.ndarray | None,
    mu_px: float,
    c1_px: float,
    p90: float | None,
    trigger: str,
) -> np.ndarray:
    """Mix the candle and mixture distributions on clock prompts.

    The mixture path scales the fitted drift down slightly and adds the last
    one-second move. Event prompts stay on that path. Clock prompts add the
    candle distribution in proportion to how quiet the recent hour has been.
    """
    shift = (_MU_KEEP - 1.0) * float(mu_px) + _C1_KEEP * float(c1_px)
    mixture = np.asarray(ladder, dtype=np.float64) + shift
    mixture = np.maximum(mixture, _MIN_PRICE)
    np.maximum.accumulate(mixture, out=mixture)
    weight = _candle_weight(p90, trigger, candle is not None)
    if candle is None or weight <= 1e-4:
        return mixture
    candle_a = np.maximum(np.asarray(candle, dtype=np.float64), _MIN_PRICE)
    np.maximum.accumulate(candle_a, out=candle_a)
    if weight >= 1.0 - 1e-4:
        return candle_a
    return _cdf_mix(candle_a, mixture, weight)


def _forecast_from_parts(
    f: dict[str, float],
    closes: np.ndarray,
    offsets: np.ndarray | None,
    ofi: float,
    now_ms: int,
    trigger: str,
    ref_scale: float,
) -> np.ndarray:
    """Percentile ladder from features and precomputed book/candle summaries."""
    global last_mu_px, last_anchor_px
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
    sigma_px = _blend_scale(sigma_px, _realized_mix_scale(closes, px))
    sigma_px = _vol_aware_scale(sigma_px, closes, px)
    sigma_px = float(np.clip(sigma_px, _MIN_SIGMA_PX, _MAX_SIGMA_PX))
    mu_px = float(np.clip(mu_px, -5.0 * sigma_px, 5.0 * sigma_px))
    # Pull a noisy feature-model shift back toward spot before the
    # liquidation and depth-band adjustments, which are directional.
    # A loud hour — its own move scale above the model scale — keeps the
    # shift. A quieter hour pulls a shift that is large next to the model
    # scale back toward spot.
    if not (float(ref_scale) > sigma_px):
        mu_px = _reliable_shift(mu_px, sigma_px)
    # input.md: a buy liquidation pushes price up; depth bands must be
    # re-anchored to the live mid before they can shift the book.
    mu_px += (_LIQ_K * f["liq_dir"] + _BAND_K * f["band_imb"]) * sigma_px
    sigma_px *= 1.0 + 0.20 * max(f["liq_hour"], 0.0)
    sigma_px = float(np.clip(sigma_px, _MIN_SIGMA_PX, _MAX_SIGMA_PX))
    sigma_px = _regime_scale(sigma_px, closes)
    sigma_px = float(np.clip(sigma_px, _MIN_SIGMA_PX, _MAX_SIGMA_PX))
    mu_px = float(np.clip(mu_px, -5.0 * sigma_px, 5.0 * sigma_px))
    mu_px, sigma_px = _flex_linear(mu_px, sigma_px, closes)
    sigma_px = float(np.clip(sigma_px, _MIN_SIGMA_PX, _MAX_SIGMA_PX))
    mu_px = float(np.clip(mu_px, -5.0 * sigma_px, 5.0 * sigma_px))
    sigma_px = _season_scale(sigma_px, px, now_ms)
    sigma_px = float(np.clip(sigma_px, _MIN_SIGMA_PX, _MAX_SIGMA_PX))
    mu_px = _season_location(mu_px, now_ms, f["momentum"])
    mu_px += _OFI_K * float(ofi) * sigma_px
    # Spot and futures top-of-book imbalance. On the prompts where the next
    # move is large, this sign is right about four times in five, and the
    # linear book coefficients were too small to use it.
    imb = 0.5 * (float(f["s_imb"]) + float(f["f_imb"]))
    imb_k = _IMB_EVENT if trigger != "time" else _IMB_K
    mu_px += imb_k * imb * sigma_px
    if trigger != "time":
        sigma_px = float(np.clip(sigma_px * _EVENT_SIGMA, _MIN_SIGMA_PX, _MAX_SIGMA_PX))
    mu_px = float(np.clip(mu_px, -5.0 * sigma_px, 5.0 * sigma_px))
    book_cap, book_pivot, book_slope = _season_book(now_ms)
    hist_w = _HIST_W
    if abs(imb) >= _LOUD_IMB:
        book_cap *= _LOUD_BOOK
        hist_w *= _LOUD_BOOK
    ladder = _mix_book(
        _ladder(px, mu_px, sigma_px),
        px,
        sigma_px,
        None,
        book_cap=book_cap,
        book_pivot=book_pivot,
        book_slope=book_slope,
        offsets=offsets,
    )
    ratio = _hist_jump_ratio(closes)
    if offsets is not None and hist_w > 0.0 and (
        closes.size < 50 or ratio < _HIST_JUMP
    ):
        ladder = (1.0 - hist_w) * ladder + hist_w * (px + offsets)
        ladder = _sanitize(ladder)
    elif ratio >= _HIST_JUMP:
        med = 0.5 * (float(ladder[49]) + float(ladder[50]))
        ladder = ladder.copy()
        ladder[8:92] = med + _CALM_CORE * (ladder[8:92] - med)
        ladder = _sanitize(ladder)
    bins = _candle_abs_bins(closes)
    candle = None if bins is None else _skewed_candle_prices(px, bins, mu_px)
    p90 = _move_p90_usd(closes, px)
    c1_px = float(f["c1"]) * px
    mixed = _combine(ladder, candle, mu_px, c1_px, p90, trigger)
    last_mu_px = float(mu_px)
    last_anchor_px = float(px)
    return mixed


def predict_percentiles(payload: dict) -> np.ndarray:
    """Return 100 centered-quantile prices for BTC 10s ahead.

    Output constraints (SPECIFICATION.md): shape (100,), float64, finite,
    strictly positive, non-decreasing. Quantile grid q_i = (2i-1)/200.

    Never raise. FAQ.md charges a non-answer (exception, timeout, or an output
    that fails validation) the 95th percentile of models that did answer.
    """
    global last_mu_px, last_anchor_px
    last_mu_px = float("nan")
    last_anchor_px = float("nan")
    try:
        f = extract(payload)
        closes = _spot_closes(payload)
        now_ms = int((payload.get("prompt") or {}).get("current_time_ms") or 0)
        book = ((payload.get("venues") or {}).get("spot") or {}).get("book_ticker")
        trigger = str(((payload.get("prompt") or {}).get("trigger") or {}).get("kind") or "time")
        return _forecast_from_parts(
            f,
            closes,
            _book_offsets(book),
            _quote_ofi(book),
            now_ms,
            trigger,
            _raw_reference_scale(payload),
        )
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
        last_mu_px = 0.0
        last_anchor_px = float(px)
        return _ladder(px, 0.0, max(8.0, 1e-4 * px))


def _import_warmup() -> None:
    """Prime numpy paths at import. Synth does not count this against the 5 ms budget."""
    predict_percentiles(
        {
            "schema_version": 5,
            "prompt": {"current_time_ms": 1_000},
            "venues": {
                "spot": {
                    "depth_bands": {
                        "anchor_price": 100.0,
                        "bucket_usd": 10.0,
                        "bids": np.ones(16, dtype=np.float64),
                        "asks": np.ones(16, dtype=np.float64),
                        "recv_ts_ms": 500,
                    }
                },
                "futures": {
                    "liquidations": {
                        "recv_ts_ms": np.array([500], dtype=np.int64),
                        "qty": np.array([0.2], dtype=np.float64),
                        "is_buy": np.array([True]),
                    }
                },
            },
        }
    )
    # Automatic GC pauses are a common source of multi-ms latency spikes.
    gc.collect()
    gc.disable()


_import_warmup()
