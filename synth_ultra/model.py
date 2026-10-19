"""10s percentile forecast for BTC, ETH, SOL, and XRP. Target: <5 ms, CPU-only, deterministic.

Two distributions are built from the payload. The mixture ladder keeps wide
tails for jumps. The candle bins, from the last 30 minutes of 10-second log
returns, put mass on an unchanged price. Clock prompts mix the two: the
candle share is a logistic of the half-hour 90th percentile move, reduced
when the last minute is hotter than that half hour. A quiet hour keeps the
candle mass and a louder hour keeps the mixture tails. On the
mixture path the fitted drift is scaled to 85% and 60% of the last
one-second move is added. Top-of-book imbalance shifts the center by a
fraction of the width, more so after a trade print, and a trade print also
moves the forecast along the last second of futures taker flow, then
pulls back the spent 1s futures micro-move. Finally ``stillness`` puts a
point mass on the current microprice, sized by how still the live spot
and futures quotes have been.
Never-raise fallback so a non-answer is not charged under FAQ.md.

Every hour-of-candles statistic is read from one pass over the 10-second
moves (``_Hour``), and scalar steps use ``math`` rather than NumPy wrappers,
whose per-call overhead dominates at these array sizes.
"""

from __future__ import annotations

import gc
import math

import numpy as np

from synth_ultra.constants import HORIZON_SECONDS, NUM_PERCENTILES, asset_spec, price_scale
from synth_ultra.features import extract, reference_price
from synth_ultra.season_table import SEAS_ABS as _SEAS_ABS
from synth_ultra.season_table import SEAS_REL as _SEAS_REL
from synth_ultra.stillness import apply as apply_stillness
from synth_ultra.stillness import still_features, taker_flow

_EPS = 1e-12
_MIN_PRICE = 1e-8
_MIN_SIGMA_PX = 0.5
_MAX_SIGMA_PX = 800.0
_SQRT_H = math.sqrt(float(HORIZON_SECONDS))

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
        -0.002,
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
_SM = 1.04
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


def _clip(x: float, lo: float, hi: float) -> float:
    """Scalar clip. NaN passes through, as with ``np.clip``."""
    return lo if x < lo else (hi if x > hi else x)


def _sanitize(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).reshape(NUM_PERCENTILES)
    # A non-finite value makes the sum non-finite, so the slow scrub only runs then.
    if not math.isfinite(float(out.sum())):
        out = np.nan_to_num(out, nan=_MIN_PRICE, posinf=1e16, neginf=_MIN_PRICE)
    out = np.maximum(out, _MIN_PRICE)
    np.maximum.accumulate(out, out=out)
    flat = out[1:] <= out[:-1]
    if flat.any():
        out[1:] += np.cumsum(np.where(flat, _EPS, 0.0))
    return out


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
_VOL_W = (0.7199761, 0.18929481, 0.06551382, 0.02521527)
_VOL_LOG_BIAS = 0.04421702415367945

# Weight vectors depend only on the series length, which is the same few
# thousand points on nearly every call.
_EWMA_W: dict[tuple[int, float], np.ndarray] = {}
_W_CACHE: dict[int, np.ndarray] = {}


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


def _median(x: np.ndarray) -> float:
    """``np.median`` through one partition, without its Python-level overhead."""
    n = int(x.size)
    k = n // 2
    if n % 2:
        return float(np.partition(x, k)[k])
    part = np.partition(x, (k - 1, k))
    return 0.5 * (float(part[k - 1]) + float(part[k]))


def _mad(x: np.ndarray) -> float:
    """Mean absolute deviation from the median."""
    d = x - _median(x)
    np.abs(d, out=d)
    return float(d.sum()) / d.size


# Overlapping 10-second close-to-close returns from 1-second candles.
# 30 minutes is 1,800 closes and 1,790 returns.
_WIN_30_S = 1800
_WIN_RET = _WIN_30_S - HORIZON_SECONDS
_MIN_RETURNS = 40
_N_BINS = 50
_BIN_W = 1.0 / _N_BINS
_BIN_EDGES = np.array([k * _BIN_W for k in range(_N_BINS)] + [1.0], dtype=np.float64)
_BIN_SPAN = np.diff(_BIN_EDGES)
# Floor under a mean absolute 10-second log return, and the fast window.
_ACT_EPS = 1e-6
_ACT_FAST = 60
_FLEX_N = 600


class _Hour:
    """Spot 1-second closes and their overlapping 10-second moves.

    Every width step reads the same hour of candles. Taking the log, the
    10-second differences and their absolute values once keeps a call to a
    handful of passes over 3,600 points.
    """

    __slots__ = ("n", "c", "lc", "r10", "a10", "d10")

    def __init__(self, close: np.ndarray) -> None:
        c = close if isinstance(close, np.ndarray) else np.asarray(close, dtype=np.float64)
        c = c.reshape(-1)
        if c.dtype != np.float64:
            c = c.astype(np.float64)
        self.n = int(c.size)
        self.c = c
        self.lc = self.r10 = self.a10 = self.d10 = None
        if self.n < 40:
            return
        lc = np.log(c)
        if not math.isfinite(float(lc.sum())):
            keep = np.isfinite(lc)
            c = c[keep]
            lc = lc[keep]
            self.n = int(c.size)
            self.c = c
            if self.n < 40:
                return
        h = HORIZON_SECONDS
        self.lc = lc
        self.r10 = lc[h:] - lc[:-h]
        self.a10 = np.abs(self.r10)
        self.d10 = np.abs(c[h:] - c[:-h])

    def realized_mix(self, px: float) -> float:
        """Mixture scale whose mean absolute deviation matches the hour's 10s moves."""
        if self.r10 is None:
            return 0.0
        mean_abs = _mad(self.r10)
        if mean_abs <= 0.0 or not math.isfinite(mean_abs):
            return 0.0
        return px * mean_abs / _Z_ABS

    def vol_aware(self, fitted: float, px: float, log_floor: float = 0.3) -> float:
        """Log-blend the fitted scale with EWMA 10s moves, 60s vol and the 10s range."""
        if self.a10 is None or not math.isfinite(fitted) or fitted <= 0.0:
            return fitted
        fast = px * _ewma_abs(self.a10) / _Z_ABS
        if self.n >= 61:
            tail = self.lc[-61:]
            rets = tail[1:] - tail[:-1]
            dev = rets - float(rets.sum()) / rets.size
            std = math.sqrt(float((dev * dev).sum()) / rets.size)
            rv60 = px * (std * math.sqrt(10.0)) / _Z_ABS
        else:
            rv60 = fast
        sl = self.c[-10:]
        lo = float(sl.min())
        hi = float(sl.max())
        hl10 = px * math.log(hi / lo) / _Z_ABS if lo > 0.0 and hi > lo else fast
        acc = _VOL_LOG_BIAS
        for w, v in zip(_VOL_W, (fitted, fast, rv60, hl10)):
            acc += w * math.log(max(v, log_floor))
        return math.exp(acc)

    def recent_mad(self, n: int) -> float:
        """Mean absolute deviation of the last ``n`` 10-second dollar moves."""
        if self.d10 is None or self.d10.size < n:
            return 0.0
        mad = _mad(self.d10[-n:])
        if not math.isfinite(mad) or mad <= 0.0:
            return 0.0
        return mad

    def jump_ratio(self) -> float:
        """p99 / median of absolute 10-second close changes. Large means unstable."""
        if self.d10 is None or self.n < 50:
            return 1e9
        d = self.d10
        m = int(d.size)
        k = m // 2
        pos = 0.99 * (m - 1)
        j = int(pos)
        frac = pos - j
        j1 = min(j + 1, m - 1)
        part = np.partition(d, (k - 1, k, j, j1))
        typical = float(part[k]) if m % 2 else 0.5 * (float(part[k - 1]) + float(part[k]))
        if typical <= 1e-6 or not math.isfinite(typical):
            return 1e9
        a = float(part[j])
        b = float(part[j1])
        extreme = a + (b - a) * frac if frac < 0.5 else b - (b - a) * (1.0 - frac)
        if not math.isfinite(extreme):
            return 1e9
        return extreme / typical

    def candle_parts(self, px: float) -> tuple[np.ndarray | None, float | None, float]:
        """Fifty absolute 10s log-return levels, their dollar p90, and activity.

        All three read the last 30 minutes of returns. Activity is the log
        ratio of the last minute's mean |10s return| to the half-hour mean.
        """
        if self.a10 is None or self.n < HORIZON_SECONDS + _MIN_RETURNS:
            return None, None, 0.0
        aw = self.a10[-_WIN_RET:]
        m = int(aw.size)
        # Tied returns add the same slope to the running integral in any order,
        # so an unstable sort gives the same bin means.
        order = np.argsort(aw)
        sv = aw[order]
        bins = _bin_means(sv, _recency_weights(m)[order])
        pos = 0.9 * (m - 1)
        i0 = int(pos)
        frac = pos - i0
        a = float(sv[i0])
        b = float(sv[min(i0 + 1, m - 1)])
        q90 = a + (b - a) * frac if frac < 0.5 else b - (b - a) * (1.0 - frac)
        activity = 0.0
        if m >= _ACT_FAST:
            slow = float(aw.sum()) / m
            fast = float(aw[-_ACT_FAST:].sum()) / _ACT_FAST
            activity = math.log((_ACT_EPS + fast) / (_ACT_EPS + slow))
        return bins, q90 * px, activity


def _realized_mix_scale(close: np.ndarray, px: float) -> float:
    return _Hour(close).realized_mix(float(px))


def _vol_aware_scale(fitted: float, close: np.ndarray, px: float) -> float:
    return _Hour(close).vol_aware(float(fitted), float(px))


def _regime_scale(sigma_px: float, close: np.ndarray, hour: _Hour | None = None) -> float:
    """Shrink a scale that is wide of the latest few minutes of 10s moves.

    ``s_recent`` is the mixture scale whose mean absolute deviation matches
    the last ``_REG_N`` overlapping 10-second dollar moves. The map
    ``clip(_REG_A + _REG_B * s_recent / sigma, _REG_A, 1)`` was chosen so
    every 4-hour window of saved prompts has a lower pinball CRPS. Short
    candle history leaves the feature-model scale unchanged.
    """
    sigma = float(sigma_px)
    if not math.isfinite(sigma) or sigma <= 0.0:
        return sigma
    h = hour if hour is not None else _Hour(close)
    mad = h.recent_mad(_REG_N)
    if mad <= 0.0:
        return sigma
    ratio = (mad / _Z_ABS) / sigma
    if not math.isfinite(ratio):
        return sigma
    return sigma * _clip(_REG_A + _REG_B * ratio, _REG_A, 1.0)


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


def _flex_linear(
    mu_px: float, sigma_px: float, close: np.ndarray, hour: _Hour | None = None
) -> tuple[float, float]:
    """Rescale the linear location and width from the latest 10-minute moves."""
    mu = float(mu_px)
    sigma = float(sigma_px)
    h = hour if hour is not None else _Hour(close)
    mad = h.recent_mad(_FLEX_N)
    if mad <= 0.0 or not math.isfinite(sigma) or sigma <= 0.0 or not math.isfinite(mu):
        return mu, sigma
    ratio = (mad / _Z_ABS) / sigma
    if not math.isfinite(ratio) or ratio <= 0.0:
        return mu, sigma
    fade = 1.0 / _clip(ratio, 0.3, 1.5)
    excess = max(abs(mu) / sigma - _MU_EXCESS, 0.0)
    gain = _clip(1.0 - _MU_PULL * excess * fade, _MU_FLOOR, 1.0)
    width = _clip(ratio**_WIDE_POW, _WIDE_FLOOR, 1.0)
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
# 0.27 is the blend that lowered both the early and late saved prompts.
_HIST_JUMP = 12.0
_HIST_W = 0.27
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


def _i64_col(x: object) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x.reshape(-1)
        if arr.dtype == np.int64:
            return arr
        return arr.astype(np.int64, copy=False)
    return np.asarray(x, dtype=np.int64).reshape(-1)


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
    ts = _i64_col(ts_raw)
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
    ts_i = ts[::step]
    j = np.searchsorted(ts, ts_i + 10_000, side="right") - 1
    ii = np.arange(0, n, step, dtype=np.int64)
    ok = (j > ii) & (ts[j] >= ts_i + 9_000)
    k = int(np.count_nonzero(ok))
    if k < 40:
        return None
    both = _micro_at(bp, bq, ap, aq, np.concatenate((ii[ok], j[ok])))
    ordered = np.sort(both[k:] - both[:k])
    m = int(ordered.size)
    half = 0.5 * (m - 1)
    h0 = int(half)
    hf = half - h0
    med = float(ordered[h0] * (1.0 - hf) + ordered[min(h0 + 1, m - 1)] * hf)
    pos = _TAU * (m - 1)
    i0 = pos.astype(np.int64)
    frac = pos - i0
    i1 = np.minimum(i0 + 1, m - 1)
    return ordered[i0] * (1.0 - frac) + ordered[i1] * frac - med


def _micro_at(
    bp: np.ndarray, bq: np.ndarray, ap: np.ndarray, aq: np.ndarray, idx: np.ndarray
) -> np.ndarray:
    """Microprice at the given ticks. An empty quote falls back to the mid."""
    b = bp[idx]
    a = ap[idx]
    qb = bq[idx]
    qa = aq[idx]
    den = qb + qa
    out = (b * qa + a * qb) / np.maximum(den, 1e-12)
    bad = den <= 0.0
    if bad.any():
        out[bad] = 0.5 * (b[bad] + a[bad])
    return out


def _quote_ofi(book: dict | None, window_ms: int = _OFI_MS) -> float:
    """Signed quote flow over the last second, scaled by its own turnover.

    A bid that steps up, or that gains size, is buying pressure. An ask that
    steps down, or that gains size, is selling pressure. The result is in
    [-1, 1]: net pressure divided by the gross quote change.
    """
    if not book or book.get("recv_ts_ms") is None:
        return 0.0
    ts = _i64_col(book["recv_ts_ms"])
    bp = _f64_col(book.get("bid_price"))
    bq = _f64_col(book.get("bid_qty"))
    ap = _f64_col(book.get("ask_price"))
    aq = _f64_col(book.get("ask_qty"))
    n = min(ts.size, bp.size, bq.size, ap.size, aq.size)
    if n < 8:
        return 0.0
    ts = ts[-n:]
    i0 = max(int(np.searchsorted(ts, int(ts[-1]) - int(window_ms), side="left")), 1)
    k = n - (i0 - 1)
    if k < 3:
        return 0.0
    bp = bp[-k:]
    bq = bq[-k:]
    ap = ap[-k:]
    aq = aq[-k:]
    dbp = bp[1:] - bp[:-1]
    dap = ap[1:] - ap[:-1]
    e_bid = np.where(dbp > 0.0, bq[1:], np.where(dbp < 0.0, -bq[:-1], bq[1:] - bq[:-1]))
    e_ask = np.where(dap < 0.0, aq[1:], np.where(dap > 0.0, -aq[:-1], aq[1:] - aq[:-1]))
    gross = float((np.abs(e_bid) + np.abs(e_ask)).sum())
    if gross <= 0.0 or not math.isfinite(gross):
        return 0.0
    net = float((e_bid - e_ask).sum())
    if not math.isfinite(net):
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
    if offsets is None or not math.isfinite(sigma) or sigma <= 0.0:
        return level
    mad = float(np.abs(offsets).sum()) / offsets.size
    if mad <= 0.0 or not math.isfinite(mad):
        return level
    rel = mad / (sigma * _Z_ABS)
    if not math.isfinite(rel):
        return level
    weight = _clip(book_slope * max(book_pivot - rel, 0.0), 0.0, book_cap)
    if weight <= 0.0:
        return level
    return (1.0 - weight) * level + weight * (float(px) + offsets)


def _blend_scale(fitted: float, realized: float) -> float:
    """Raise a too-tight fitted scale toward the realized-move scale.

    When the fitted scale is already wider, keep it: live bursts should be
    able to fatten the distribution immediately. The 0.8 power closes most of
    the gap and is the holdout minimum between 'ignore the hour' and 'replace
    the fitted scale outright'.
    """
    if not math.isfinite(realized) or realized <= fitted or fitted <= 0.0:
        return fitted
    return fitted * (realized / fitted) ** 0.8


def _hist_jump_ratio(close: np.ndarray) -> float:
    """p99 / median of absolute 10-second close changes. Large means unstable."""
    return _Hour(close).jump_ratio()


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
            math.tanh(f["c1"] * 3000.0),
            f["basis"] * vol10,
        ],
        dtype=np.float64,
    )


def _abs_vec(f: dict[str, float], vol10: float, vol_fast10: float) -> np.ndarray:
    rng = max(f["s_rng"], f["f_rng"])
    rv = max(f["s_rv"], f["f_rv"])
    log_fqty = math.log1p(max(f["f_qty2"], 0.0)) / 10.0
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
            math.log1p(max(f["s_qty2"], 0.0)) / 10.0,
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
# Weekend slots quieter than this fraction of the same weekday hour trust
# the payload's own 10s book quantiles more.
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
        not math.isfinite(sigma)
        or sigma <= 0.0
        or not math.isfinite(price)
        or price <= 0.0
        or int(now_ms) < 0
    ):
        return sigma
    seas = price * float(_SEAS_ABS[_slot_index(now_ms)]) / _Z_ABS
    if not math.isfinite(seas) or seas <= 0.0:
        return sigma
    adj = _clip(_SEAS_W * (math.log(seas) - math.log(sigma)), -_SEAS_CLIP, _SEAS_CLIP)
    return sigma * math.exp(adj)


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
    if math.isfinite(rel) and rel < _SEAS_QUIET_REL:
        return _SEAS_QUIET_CAP, _SEAS_QUIET_PIVOT, _SEAS_QUIET_SLOPE
    return _BOOK_CAP, _BOOK_PIVOT, _BOOK_SLOPE


# Location of the forecast that was just returned. The plot reads these.
# They are not part of the scored output.
last_mu_px = float("nan")
last_anchor_px = float("nan")

# |μ| / mean(|log return|) above this would push one wing through the anchor.
_MU_ALPHA_CAP = 1.0


def _bin_means(sorted_values: np.ndarray, sorted_weights: np.ndarray) -> np.ndarray:
    """Average of each 0.02 mass slice, walking the sorted absolute returns.

    A point whose weight crosses a boundary contributes to each slice in
    proportion to the weight that falls inside it. The running integral of
    value over mass is linear between points, so each slice is a difference
    of that integral at its two edges.
    """
    n = int(sorted_values.size)
    if n == 0:
        return np.zeros(_N_BINS, dtype=np.float64)
    cw = np.cumsum(sorted_weights)
    total = float(cw[-1])
    if total <= 0.0 or not math.isfinite(total):
        return np.zeros(_N_BINS, dtype=np.float64)
    mass = np.empty(n + 1, dtype=np.float64)
    mass[0] = 0.0
    np.divide(cw, total, out=mass[1:])
    area = np.empty(n + 1, dtype=np.float64)
    area[0] = 0.0
    np.cumsum(sorted_weights * sorted_values, out=area[1:])
    area[1:] /= total
    g = np.interp(_BIN_EDGES, mass, area)
    means = g[1:] - g[:-1]
    means /= _BIN_SPAN
    np.maximum.accumulate(means, out=means)
    np.maximum(means, 0.0, out=means)
    return means


def _candle_abs_bins(close: np.ndarray) -> np.ndarray | None:
    """Fifty absolute 10-second log-return levels from the recent candle window."""
    return _Hour(close).candle_parts(1.0)[0]


def _skewed_candle_prices(px: float, bins: np.ndarray, mu_px: float) -> np.ndarray:
    """Mirror the 50 absolute levels, then tilt each one by μ in proportion to |dev|.

    ``alpha > 0`` lengthens the upper wing and shortens the lower wing by the
    same fraction of each level's own distance from the anchor. Levels that
    are exactly zero — unchanged 10-second candles — stay on the anchor.
    """
    scale = float(bins.sum()) / bins.size
    mu_ret = float(mu_px) / float(px) if px > 0.0 else 0.0
    alpha = 0.0
    if scale > 1e-18 and math.isfinite(mu_ret) and math.isfinite(scale):
        alpha = _clip(mu_ret / scale, -_MU_ALPHA_CAP, _MU_ALPHA_CAP)
    expo = np.empty(NUM_PERCENTILES, dtype=np.float64)
    np.multiply(bins[::-1], -(1.0 - alpha), out=expo[:_N_BINS])
    np.multiply(bins, 1.0 + alpha, out=expo[_N_BINS:])
    out = np.exp(expo)
    out *= px
    if not math.isfinite(float(out.sum())):
        np.nan_to_num(out, copy=False, nan=_MIN_PRICE, posinf=1e16, neginf=_MIN_PRICE)
    np.maximum(out, _MIN_PRICE, out=out)
    np.maximum.accumulate(out, out=out)
    return out


# Clock prompts mix the candle distribution with the mixture. The candle
# weight is a logistic of the half-hour 90th percentile 10-second move:
# about 0.70 when that move is $0, one half at $7.5, and gone once the hour
# is much louder. A hotter last minute, relative to that half hour, lowers
# the candle share. Pivot and temperature were re-chosen with the stillness
# point mass in place; both halves of the saved prompts agreed.
_BLEND_PIVOT = 7.5
_BLEND_TEMP = 9.0
_BLEND_BETA = 0.12
# The fitted drift runs a little hot. Keep most of it, and add part of the
# last one-second move, which the drift underweights.
_MU_KEEP = 0.85
_C1_KEEP = 0.60
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
# Non-clock forecasts are tight between the 25% and 75% prices: that central
# interval covered 46.8% on both the early and later saved prompts. This
# fraction of the inner quartile gap is added outside the median. It fades to
# zero at the median and at the 10% and 90% prices. 0.08 is where both slices
# bottomed.
_EVENT_CENTER = 0.08
_EVENT_TENT = np.zeros(_TAU.shape, dtype=np.float64)
_left_band = (_TAU > 0.10) & (_TAU < 0.50)
_EVENT_TENT[_left_band] = np.where(
    _TAU[_left_band] <= 0.25,
    (_TAU[_left_band] - 0.10) / 0.15,
    (0.50 - _TAU[_left_band]) / 0.25,
)
_right_band = (_TAU > 0.50) & (_TAU < 0.90)
_EVENT_TENT[_right_band] = np.where(
    _TAU[_right_band] <= 0.75,
    (_TAU[_right_band] - 0.50) / 0.25,
    (0.90 - _TAU[_right_band]) / 0.15,
)
_EVENT_TENT[:11] = 0.0
_EVENT_TENT[49:51] = 0.0
_EVENT_TENT[89:] = 0.0
_EVENT_LEFT = _TAU < 0.50


def _widen_event_center(pct: np.ndarray) -> np.ndarray:
    """Move the 25% price down and the 75% price up.

    Clock forecasts do not come through here. The median, the 10% and 90%
    prices, and every quantile outside that band are copied from the input.
    """
    q = np.asarray(pct, dtype=np.float64)
    q25 = 0.5 * (float(q[24]) + float(q[25]))
    q50 = 0.5 * (float(q[49]) + float(q[50]))
    q75 = 0.5 * (float(q[74]) + float(q[75]))
    gap = np.where(_EVENT_LEFT, -_EVENT_CENTER * (q50 - q25), _EVENT_CENTER * (q75 - q50))
    return _sanitize(q + gap * _EVENT_TENT)


def _candle_weight(
    p90: float | None,
    trigger: str,
    candle_ok: bool,
    activity: float,
    scale: float = 1.0,
) -> float:
    """Share of the candle distribution on a clock prompt. Zero otherwise.

    ``scale`` is 1 for BTC. Other assets use the same log-return threshold:
    the fitted $7.5 pivot is a BTC price, and a raw dollar comparison keeps
    ETH, SOL, and XRP on the quiet-hour candle path.
    """
    if trigger != "time" or not candle_ok or p90 is None or not math.isfinite(p90):
        return 0.0
    span = scale if math.isfinite(scale) and scale > 0.0 else 1.0
    z = (float(p90) - _BLEND_PIVOT * span) / (_BLEND_TEMP * span) + _BLEND_BETA * float(activity)
    if z >= 20.0:
        return 0.0
    if z <= -20.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def _component_parts(q: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right-continuous CDF and point mass of a piecewise-linear quantile function.

    A run of equal prices is an atom: the CDF at that price is the rightmost
    tau sitting on it, and its mass spans that run. Past the last price the
    CDF is 1, and the last price also carries the open right tail.
    """
    right_n = np.searchsorted(q, x, side="right")
    left_i = np.searchsorted(q, x, side="left")
    idx = right_n - 1
    out = np.zeros(x.shape, dtype=np.float64)
    last = q.size - 1
    hi = idx >= last
    mid = (idx >= 0) & ~hi
    out[hi] = 1.0
    if mid.any():
        i = idx[mid]
        q_i = q[i]
        span = q[i + 1] - q_i
        frac = np.zeros(i.shape, dtype=np.float64)
        ok = span > 1e-18
        frac[ok] = (x[mid][ok] - q_i[ok]) / span[ok]
        np.clip(frac, 0.0, 1.0, out=frac)
        t_i = _TAU[i]
        out[mid] = t_i + (_TAU[i + 1] - t_i) * frac
    mass = np.zeros(x.shape, dtype=np.float64)
    multi = (idx > left_i) & (left_i < q.size)
    if multi.any():
        mass[multi] = _TAU[idx[multi]] - _TAU[left_i[multi]]
    mass[x == q[-1]] += 1.0 - float(_TAU[-1])
    return out, mass


def _cdf_mix(candle: np.ndarray, mixture: np.ndarray, weight: float) -> np.ndarray:
    """Quantiles of ``weight * candle + (1 - weight) * mixture``.

    The two quantile functions are inverted on their sorted breakpoints.
    Repeated prices stay a point mass instead of being smeared across a
    uniform price grid. Probability below the first represented tau sits on
    the left endpoint, and probability above the last sits on the right
    endpoint of the component that owns that tail. Between breakpoints the
    quantile is linear in probability.
    """
    w = float(weight)
    c = candle
    m = mixture
    knots = np.unique(np.concatenate((c, m)))
    if knots.size < 2 or not math.isfinite(w):
        return _sanitize((1.0 - w) * m + w * c)
    if bool((m[1:] > m[:-1]).all()):
        # A strictly increasing ladder has no atoms: its CDF is a plain
        # interpolation, and its only point mass is the open right tail.
        top = knots >= m[-1]
        cdf_m = np.interp(knots, m, _TAU, left=0.0, right=1.0)
        cdf_m[top] = 1.0
        atom_m = np.where(knots == m[-1], 1.0 - float(_TAU[-1]), 0.0)
    else:
        cdf_m, atom_m = _component_parts(m, knots)
    cdf_c, atom_c = _component_parts(c, knots)
    right = w * cdf_c + (1.0 - w) * cdf_m
    jump = w * atom_c + (1.0 - w) * atom_m
    np.maximum.accumulate(right, out=right)
    right[-1] = 1.0
    left = np.maximum(right - jump, 0.0)
    left[0] = 0.0
    np.minimum(left, right, out=left)
    j = np.clip(np.searchsorted(right, _TAU, side="left"), 0, knots.size - 1)
    out = knots[j]
    prev = np.maximum(j - 1, 0)
    r_prev = right[prev]
    l_j = left[j]
    span = l_j - r_prev
    cont = (_TAU <= l_j + 1e-15) & (j > 0) & (span > 1e-15)
    if cont.any():
        frac = np.clip((_TAU[cont] - r_prev[cont]) / span[cont], 0.0, 1.0)
        k_prev = knots[prev[cont]]
        out[cont] = k_prev + frac * (knots[j[cont]] - k_prev)
    return _sanitize(out)


def _combine(
    ladder: np.ndarray,
    candle: np.ndarray | None,
    mu_px: float,
    c1_px: float,
    p90: float | None,
    trigger: str,
    activity: float,
    scale: float = 1.0,
) -> np.ndarray:
    """Mix the candle and mixture distributions on clock prompts.

    The mixture path scales the fitted drift down slightly and adds the last
    one-second move. Event prompts stay on that path. Clock prompts add the
    candle distribution in proportion to how quiet the recent hour has been,
    and how quiet the last minute is relative to that hour.
    """
    shift = (_MU_KEEP - 1.0) * float(mu_px) + _C1_KEEP * float(c1_px)
    mixture = ladder + shift
    np.maximum(mixture, _MIN_PRICE, out=mixture)
    np.maximum.accumulate(mixture, out=mixture)
    weight = _candle_weight(p90, trigger, candle is not None, activity, scale)
    if candle is None or weight <= 1e-4:
        return mixture
    if weight >= 1.0 - 1e-4:
        return candle
    return _cdf_mix(candle, mixture, weight)


def _forecast_from_parts(
    f: dict[str, float],
    closes: np.ndarray,
    offsets: np.ndarray | None,
    ofi: float,
    now_ms: int,
    trigger: str,
    hour: _Hour | None = None,
    asset: str = "BTC",
) -> np.ndarray:
    """Percentile ladder from features and precomputed book/candle summaries."""
    global last_mu_px, last_anchor_px
    h = hour if hour is not None else _Hour(closes)
    px = max(f["price"], _MIN_PRICE)
    scale = price_scale(asset, px)
    lo = _MIN_SIGMA_PX * scale
    hi = _MAX_SIGMA_PX * scale
    vol10 = f["vol_1s"] * _SQRT_H
    vol_fast10 = f["vol_fast"] * _SQRT_H
    rng = max(f["s_rng"], f["f_rng"])
    rv = max(f["s_rv"], f["f_rv"])

    mu_ret = float(np.dot(_B_MU, _mu_vec(f, vol10)) * _SH)
    pred_abs = max(float(np.dot(_B_ABS, _abs_vec(f, vol10, vol_fast10))), 0.0)
    # Location/scale in price units (CRPS is dollar-denominated).
    mu_px = mu_ret * px
    sigma_px = (_KA * pred_abs + _KB * abs(mu_ret) + _FL) * px
    sigma_px = max(sigma_px, (_FR * rng + _FV * rv) * px)
    sigma_px = _clip(sigma_px * _SM, lo, hi)
    sigma_px = _blend_scale(sigma_px, h.realized_mix(px))
    sigma_px = h.vol_aware(sigma_px, px, log_floor=0.3 * scale)
    sigma_px = _clip(sigma_px, lo, hi)
    mu_px = _clip(mu_px, -5.0 * sigma_px, 5.0 * sigma_px)
    # input.md: a buy liquidation pushes price up; depth bands must be
    # re-anchored to the live mid before they can shift the book.
    mu_px += (_LIQ_K * f["liq_dir"] + _BAND_K * f["band_imb"]) * sigma_px
    sigma_px *= 1.0 + 0.20 * max(f["liq_hour"], 0.0)
    sigma_px = _clip(sigma_px, lo, hi)
    sigma_px = _regime_scale(sigma_px, closes, h)
    sigma_px = _clip(sigma_px, lo, hi)
    mu_px = _clip(mu_px, -5.0 * sigma_px, 5.0 * sigma_px)
    mu_px, sigma_px = _flex_linear(mu_px, sigma_px, closes, h)
    sigma_px = _clip(sigma_px, lo, hi)
    mu_px = _clip(mu_px, -5.0 * sigma_px, 5.0 * sigma_px)
    sigma_px = _season_scale(sigma_px, px, now_ms)
    sigma_px = _clip(sigma_px, lo, hi)
    mu_px += _OFI_K * float(ofi) * sigma_px
    # Spot and futures top-of-book imbalance. On the prompts where the next
    # move is large, this sign is right about four times in five, and the
    # linear book coefficients were too small to use it.
    imb = 0.5 * (float(f["s_imb"]) + float(f["f_imb"]))
    imb_k = _IMB_EVENT if trigger != "time" else _IMB_K
    mu_px += imb_k * imb * sigma_px
    mu_px = _clip(mu_px, -5.0 * sigma_px, 5.0 * sigma_px)
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
    ratio = h.jump_ratio()
    if offsets is not None and hist_w > 0.0 and (h.n < 50 or ratio < _HIST_JUMP):
        ladder = (1.0 - hist_w) * ladder + hist_w * (px + offsets)
    elif ratio >= _HIST_JUMP:
        med = 0.5 * (float(ladder[49]) + float(ladder[50]))
        ladder = ladder.copy()
        ladder[8:92] = med + _CALM_CORE * (ladder[8:92] - med)
    bins, p90, activity = h.candle_parts(px)
    candle = None if bins is None else _skewed_candle_prices(px, bins, mu_px)
    c1_px = float(f["c1"]) * px
    mixed = _combine(ladder, candle, mu_px, c1_px, p90, trigger, activity, scale)
    if trigger != "time":
        mixed = _widen_event_center(mixed)
    last_mu_px = float(mu_px)
    last_anchor_px = float(px)
    return mixed


# Futures taker flow in the last second leads spot after a trade print. One
# standard deviation of that flow (5.6 BTC on the saved trade-triggered
# prompts) moves the forecast by this fraction of its width; the flow is
# clipped at four deviations. Both halves of the saved prompts improved.
_FLOW_K = 0.10
_FLOW_SD = 5.644958570072203
_FLOW_CLIP = 4.0
_FLOW_MS = 1000
# The linear path and the last-second candle already spend the 1s futures
# micro return. On trade prints that overshoots: one standardized 1s futures
# move pulls the ladder back by this fraction of its width. Fit on the first
# 60% of saved event prompts; the later 40% improved by more.
_FM1_K = -0.08
_FM1_SD = 7.34486046119216e-05
_FM1_CLIP = 4.0


def _event_shift(
    q: np.ndarray,
    f: dict[str, float],
    fut: dict | None,
    now_ms: int,
    flow_sd: float = _FLOW_SD,
    fm1_sd: float = _FM1_SD,
) -> np.ndarray:
    """Trade-print location: futures taker flow minus the spent 1s futures move."""
    flow = taker_flow((fut or {}).get("trades"), now_ms, _FLOW_MS)
    zf = _clip(flow / flow_sd, -_FLOW_CLIP, _FLOW_CLIP)
    denom = fm1_sd if math.isfinite(fm1_sd) and fm1_sd > 0.0 else _FM1_SD
    zm = _clip(float(f["f_m1"]) / denom, -_FM1_CLIP, _FM1_CLIP)
    delta = _FLOW_K * zf + _FM1_K * zm
    if not (delta != 0.0) or not math.isfinite(delta):
        return q
    med = 0.5 * (float(q[49]) + float(q[50]))
    width = float(np.abs(q - med).sum()) / NUM_PERCENTILES
    return q + delta * width


def _flow_shift(q: np.ndarray, fut: dict | None, now_ms: int) -> np.ndarray:
    """Futures-flow half of ``_event_shift``, for research timings."""
    return _event_shift(q, {"f_m1": 0.0}, fut, now_ms)


def _fallback(payload: dict) -> np.ndarray:
    """Wide mixture around the last quote so a degraded path is not overconfident."""
    global last_mu_px, last_anchor_px
    px = _MIN_PRICE
    try:
        venues = (payload or {}).get("venues") or {}
        px = max(reference_price(venues.get("spot"), venues.get("futures")), _MIN_PRICE)
    except Exception:
        px = _MIN_PRICE
    last_mu_px = 0.0
    last_anchor_px = float(px)
    asset = str(((payload or {}).get("prompt") or {}).get("asset") or "BTC")
    return _ladder(px, 0.0, max(8.0 * price_scale(asset, px), 1e-4 * px))


def predict_percentiles(payload: dict) -> np.ndarray:
    """Return 100 centered-quantile prices for the prompted asset, 10s ahead.

    Output constraints (SPECIFICATION.md): shape (100,), float64, finite,
    strictly positive, non-decreasing. Quantile grid q_i = (2i-1)/200.

    Never raise. FAQ.md charges a non-answer (exception, timeout, or an output
    that fails validation) the 95th percentile of models that did answer.
    """
    global last_mu_px, last_anchor_px
    last_mu_px = float("nan")
    last_anchor_px = float("nan")
    try:
        prompt = payload.get("prompt") or {}
        venues = payload.get("venues") or {}
        spot = venues.get("spot") or {}
        f = extract(payload)
        closes = _venue_closes(spot)
        book = spot.get("book_ticker")
        # A missing spot hour used to leave the ladder on the book alone, which
        # is too tight once the book is live. Futures candles are the same
        # 10-second moves in log space; the anchor stays the spot microprice.
        if closes.size < 40:
            fut_closes = _venue_closes((venues.get("futures") or {}))
            if fut_closes.size >= 40:
                closes = fut_closes
        hour = _Hour(closes)
        now_ms = int(prompt.get("current_time_ms") or 0)
        trigger = str((prompt.get("trigger") or {}).get("kind") or "time")
        asset = str(prompt.get("asset") or "BTC")
        out = _forecast_from_parts(
            f,
            closes,
            _book_offsets(book),
            _quote_ofi(book),
            now_ms,
            trigger,
            hour,
            asset,
        )
        if _valid(out):
            try:
                out = _post_layers(out, payload, f, closes, now_ms, trigger, asset)
            except Exception:
                pass
            return out
    except Exception:
        pass
    return _fallback(payload)


def _valid(out: np.ndarray) -> bool:
    return out.shape == (NUM_PERCENTILES,) and out[0] > 0.0 and math.isfinite(float(out.sum()))


# The fitted ladder is a bit wide of the last few minutes once the regime has
# cooled, and a bit tight after it has heated up. Move 50% of the way toward
# the width of the latest 180 overlapping 10-second moves, and never by more
# than 10%. The stillness atom is put back on the anchor afterwards.
_LIVE_POW = 0.50
_LIVE_LO = 0.90
_LIVE_HI = 1.10
_LIVE_N = 180
_ATOM = 0.02


def _atom_band(asset: str, scale: float) -> float:
    """Prices this close to the anchor are the stillness point mass.

    BTC keeps the fitted two-cent band. On ETH, SOL, and XRP that band
    scaled by price falls through the tick, so the live-width step was
    smearing the atom off the touch.
    """
    if asset == "BTC":
        return _ATOM
    tick = asset_spec(asset).tick
    return max(_ATOM * scale, tick)


def _live_fm1_sd(closes: np.ndarray) -> float:
    """Scale of a 1-second log move from this asset's own candle tail.

    The fitted ``_FM1_SD`` is BTC's. ETH, SOL, and XRP run hotter, so the
    same constant treats an ordinary print as a four-sigma futures move.
    """
    c = closes.reshape(-1) if isinstance(closes, np.ndarray) else np.asarray(closes, dtype=np.float64)
    if c.size > 181:
        c = c[-181:]
    keep = np.isfinite(c) & (c > 0.0)
    if not bool(keep.all()):
        c = c[keep]
    if c.size < 40:
        return _FM1_SD
    r = np.diff(np.log(c))
    r = r[np.isfinite(r)]
    if r.size < 30:
        return _FM1_SD
    sd = float(r.std())
    if not math.isfinite(sd) or sd <= 0.0:
        return _FM1_SD
    return _clip(sd, 0.5 * _FM1_SD, 12.0 * _FM1_SD)


def _live_width(
    q: np.ndarray,
    closes: np.ndarray,
    anchor: float,
    scale: float = 1.0,
    atom: float = _ATOM,
) -> np.ndarray:
    """Rescale a finished ladder toward the payload's own recent 10s moves."""
    c = closes.reshape(-1) if isinstance(closes, np.ndarray) else np.asarray(closes, dtype=np.float64)
    if c.size < 40:
        return q
    keep = np.isfinite(c) & (c > 0.0)
    if not bool(keep.all()):
        c = c[keep]
    if c.size < 40:
        return q
    d = c[10:] - c[:-10]
    if d.size > _LIVE_N:
        d = d[-_LIVE_N:]
    d = d[np.isfinite(d)]
    if d.size < 40:
        return q
    emp = np.quantile(d, _TAU)
    em = 0.5 * (float(emp[49]) + float(emp[50]))
    ew = float(np.abs(emp - em).mean())
    med = 0.5 * (float(q[49]) + float(q[50]))
    width = float(np.abs(q - med).mean())
    if not math.isfinite(ew) or ew <= 0.0 or width < 0.5 * scale or not math.isfinite(med):
        return q
    alpha = _clip((ew / width) ** _LIVE_POW, _LIVE_LO, _LIVE_HI)
    if abs(alpha - 1.0) < 1e-4:
        return q
    x = med + alpha * (q - med)
    on = np.abs(q - anchor) <= atom
    if bool(on.any()):
        x[on] = q[on]
    np.maximum(x, _MIN_PRICE, out=x)
    np.maximum.accumulate(x, out=x)
    if not math.isfinite(float(x.sum())) or float(x[0]) <= 0.0:
        return q
    return x


# Residual scale after the BTC ladder is expressed in each asset's own units.
# Picked on the earlier half of the saved prompts; the later half agreed.
# ETH is closest to BTC's log-vol, SOL and XRP run wider than that fit.
_WIDTH_ASSET = {"ETH": 1.25, "SOL": 1.50, "XRP": 1.60}


def _asset_width(q: np.ndarray, anchor: float, asset: str, scale: float) -> np.ndarray:
    """Open a non-BTC ladder. The stillness prices on the touch stay put."""
    alpha = _WIDTH_ASSET.get(asset, 1.0)
    if alpha == 1.0:
        return q
    med = 0.5 * (float(q[49]) + float(q[50]))
    if not math.isfinite(med):
        return q
    x = med + alpha * (q - med)
    on = np.abs(q - anchor) <= _atom_band(asset, scale)
    if bool(on.any()):
        x[on] = q[on]
    np.maximum(x, _MIN_PRICE, out=x)
    np.maximum.accumulate(x, out=x)
    if not math.isfinite(float(x.sum())) or float(x[0]) <= 0.0:
        return q
    return x


def _post_layers(
    out: np.ndarray,
    payload: dict,
    f: dict[str, float],
    closes: np.ndarray,
    now_ms: int,
    trigger: str,
    asset: str = "BTC",
) -> np.ndarray:
    """Event location shift, stillness, then a width check against recent moves."""
    event = trigger != "time"
    q = out
    anchor = max(f["price"], _MIN_PRICE)
    scale = price_scale(asset, anchor)
    if event:
        fm1_sd = _FM1_SD if asset == "BTC" else _live_fm1_sd(closes)
        q = _event_shift(
            q,
            f,
            (payload.get("venues") or {}).get("futures"),
            now_ms,
            flow_sd=_FLOW_SD / scale,
            fm1_sd=fm1_sd,
        )
    seas = float(_SEAS_ABS[_slot_index(now_ms)]) if now_ms >= 0 else 0.0
    x = still_features(payload, closes, now_ms, seas, 0.0, asset=asset)
    q = apply_stillness(q, anchor, x, event, asset=asset)
    q = _live_width(q, closes, anchor, scale, _atom_band(asset, scale))
    q = _asset_width(q, anchor, asset, scale)
    q = _vol_blend(q, f, asset)
    q = _remap_quantiles(q, asset)
    return q if _valid(q) else out


# Median-centered mixture scaled by the asset's own 1-second vol, mixed into
# the finished ladder. k is the vol multiple, a is the mixture share. Chosen
# so every 24-hour window of the saved prompts scores lower. BTC is left on
# the feature ladder; a vol mix there raised the loud windows.
_VOL_BLEND = {"ETH": (0.80, 0.04), "SOL": (0.70, 0.35), "XRP": (0.80, 0.14)}


def _vol_blend(q: np.ndarray, f: dict[str, float], asset: str) -> np.ndarray:
    """Pull an alt ladder toward a short-horizon mixture, keeping its median."""
    spec = _VOL_BLEND.get(asset)
    if spec is None:
        return q
    k, share = spec
    vol = float(f.get("vol_1s") or 0.0)
    px = float(f.get("price") or 0.0)
    if not (vol > 0.0 and px > 0.0 and 0.0 < share < 1.0):
        return q
    sig = float(k) * px * vol * _SQRT_H
    if not math.isfinite(sig) or sig <= 0.0:
        return q
    med = 0.5 * (float(q[49]) + float(q[50]))
    if not math.isfinite(med):
        return q
    out = (1.0 - share) * q + share * (med + sig * _MIX_Z)
    np.maximum(out, _MIN_PRICE, out=out)
    np.maximum.accumulate(out, out=out)
    return out


# Probability warp of the finished BTC ladder. The new quantile at tau is the
# old quantile at g(tau), with g fit on the earlier saved prompts and checked
# on the later ones. It widens the body and shifts the center down a fraction
# of the width. ETH, SOL, and XRP had no warp that lowered every 24-hour window.
_REMAP_KNOTS = np.array([0.0, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 1.0])
_REMAP_G = {
    "BTC": np.array([0.0, 0.05, 0.13, 0.265, 0.465, 0.715, 0.87, 0.945, 1.0]),
}


def _remap_index(g_knots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.interp(_TAU, _REMAP_KNOTS, g_knots)
    g = np.clip(g, float(_TAU[0]), float(_TAU[-1]))
    pos = (g - _TAU[0]) / (_TAU[1] - _TAU[0])
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, NUM_PERCENTILES - 2)
    return i0, pos - i0


_REMAP_INDEX = {name: _remap_index(knots) for name, knots in _REMAP_G.items()}


def _remap_quantiles(q: np.ndarray, asset: str) -> np.ndarray:
    """Read the ladder at a warped probability grid."""
    index = _REMAP_INDEX.get(asset)
    if index is None:
        return q
    i0, frac = index
    out = q[i0] * (1.0 - frac) + q[i0 + 1] * frac
    np.maximum(out, _MIN_PRICE, out=out)
    np.maximum.accumulate(out, out=out)
    return out


def _warmup_payload(trigger: str) -> dict:
    """A full-size synthetic hour, so every array path is primed before scoring."""
    rng = np.random.default_rng(7)
    now = 1_790_000_000_000
    n = 3600
    close = 80_000.0 * np.exp(np.cumsum(rng.normal(0.0, 4e-5, n)))
    ohlcv = np.stack([close, close + 1.0, close - 1.0, close, rng.uniform(0.1, 3.0, n)], axis=1)
    candles = {
        "open_time_ms": now - 1000 * np.arange(n, 0, -1, dtype=np.int64),
        "ohlcv": ohlcv,
        "complete_history": True,
    }
    k = 4000
    recv = np.linspace(now - 59_900, now, k).astype(np.int64)
    mid = close[-1] + np.cumsum(rng.normal(0.0, 0.3, k))
    book = {
        "recv_ts_ms": recv,
        "bid_price": mid - 0.005,
        "bid_qty": rng.uniform(0.1, 5.0, k),
        "ask_price": mid + 0.005,
        "ask_qty": rng.uniform(0.1, 5.0, k),
    }
    m = 600
    trades = {
        "ts_ms": np.linspace(now - 59_000, now, m).astype(np.int64),
        "event_ts_ms": np.linspace(now - 59_000, now, m).astype(np.int64),
        "recv_ts_ms": np.linspace(now - 59_000, now, m).astype(np.int64),
        "price": mid[-m:],
        "qty": rng.uniform(0.001, 1.0, m),
        "buyer_is_maker": rng.random(m) < 0.5,
    }
    levels = np.arange(1.0, 21.0)
    depth = {
        "recv_ts_ms": now - 100,
        "bids": np.stack([mid[-1] - levels, rng.uniform(0.1, 3.0, 20)], axis=1),
        "asks": np.stack([mid[-1] + levels, rng.uniform(0.1, 3.0, 20)], axis=1),
    }
    venue = {
        "candles_1s": candles,
        "trades": trades,
        "book_ticker": book,
        "depth_latest": depth,
        "last_event_times": {"trade": now, "bookTicker": now},
    }
    spot = dict(venue)
    spot["depth_bands"] = {
        "anchor_price": float(mid[-1]),
        "bucket_usd": 10.0,
        "bids": rng.uniform(0.5, 5.0, 100),
        "asks": rng.uniform(0.5, 5.0, 100),
        "recv_ts_ms": now - 5_000,
    }
    fut = dict(venue)
    fut["liquidations"] = {
        "recv_ts_ms": np.array([now - 5_000, now - 600_000], dtype=np.int64),
        "price": np.array([mid[-1], mid[-1]]),
        "qty": np.array([0.2, 0.5]),
        "is_buy": np.array([True, False]),
    }
    return {
        "schema_version": 5,
        "prompt": {"current_time_ms": now, "trigger": {"kind": trigger, "venue": None}},
        "venues": {"spot": spot, "futures": fut},
    }


def _import_warmup() -> None:
    """Prime numpy paths at import. Synth does not count this against the 5 ms budget."""
    for trigger in ("time", "event"):
        payload = _warmup_payload(trigger)
        for _ in range(3):
            predict_percentiles(payload)
    predict_percentiles({"prompt": {"current_time_ms": 1_000}, "venues": {}})
    # Automatic GC pauses are a common source of multi-ms latency spikes.
    gc.collect()
    gc.disable()


_import_warmup()
