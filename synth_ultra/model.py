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
and futures quotes have been. On a trade print the moved part of the ladder
is then re-centred toward the last spot trade, which the quote can lag.
Constants and per-asset values live in ``synth_ultra.config``.
Never-raise fallback so a non-answer is not charged under FAQ.md.

Every hour-of-candles statistic is read from one pass over the 10-second
moves (``_Hour``), and scalar steps use ``math`` rather than NumPy wrappers,
whose per-call overhead dominates at these array sizes.
"""

from __future__ import annotations

import gc
import math

import numpy as np

from synth_ultra.config import ACT_EPS as _ACT_EPS
from synth_ultra.config import ACT_FAST as _ACT_FAST
from synth_ultra.config import ATOM as _ATOM
from synth_ultra.config import B_ABS as _B_ABS
from synth_ultra.config import B_MU as _B_MU
from synth_ultra.config import BAND_K as _BAND_K
from synth_ultra.config import BLEND_BETA as _BLEND_BETA
from synth_ultra.config import BLEND_PIVOT as _BLEND_PIVOT
from synth_ultra.config import BLEND_REALIZED_POW as _BLEND_REALIZED_POW
from synth_ultra.config import BLEND_TEMP as _BLEND_TEMP
from synth_ultra.config import BOOK_CAP as _BOOK_CAP
from synth_ultra.config import BOOK_MIN_GAP_MS as _BOOK_MIN_GAP_MS
from synth_ultra.config import BOOK_MIN_PAIRS as _BOOK_MIN_PAIRS
from synth_ultra.config import BOOK_MIN_TICKS as _BOOK_MIN_TICKS
from synth_ultra.config import BOOK_PAIRS as _BOOK_PAIRS
from synth_ultra.config import BOOK_PIVOT as _BOOK_PIVOT
from synth_ultra.config import BOOK_SLOPE as _BOOK_SLOPE
from synth_ultra.config import C1_TANH_SCALE as _C1_TANH_SCALE
from synth_ultra.config import CALM_CORE as _CALM_CORE
from synth_ultra.config import CANDLE_BINS as _N_BINS
from synth_ultra.config import CANDLE_MIN_RETURNS as _MIN_RETURNS
from synth_ultra.config import CANDLE_Q as _CANDLE_Q
from synth_ultra.config import CANDLE_WIN_S as _WIN_30_S
from synth_ultra.config import EPS as _EPS
from synth_ultra.config import EVENT_CENTER as _EVENT_CENTER
from synth_ultra.config import EVENT_TENT_KNOTS as _EVENT_TENT_KNOTS
from synth_ultra.config import EVENT_TRADE_ANCHOR as _EVENT_TRADE_ANCHOR
from synth_ultra.config import FALLBACK_SIGMA_PX as _FALLBACK_SIGMA_PX
from synth_ultra.config import FALLBACK_SIGMA_REL as _FALLBACK_SIGMA_REL
from synth_ultra.config import FL as _FL
from synth_ultra.config import FLEX_FADE_HI as _FLEX_FADE_HI
from synth_ultra.config import FLEX_FADE_LO as _FLEX_FADE_LO
from synth_ultra.config import FLEX_N as _FLEX_N
from synth_ultra.config import FLOW_CLIP as _FLOW_CLIP
from synth_ultra.config import FLOW_MS as _FLOW_MS
from synth_ultra.config import FLOW_SD as _FLOW_SD
from synth_ultra.config import FM1_CLIP as _FM1_CLIP
from synth_ultra.config import FM1_K as _FM1_K
from synth_ultra.config import FM1_LIVE_N as _FM1_LIVE_N
from synth_ultra.config import FM1_SD as _FM1_SD
from synth_ultra.config import FM1_SD_HI as _FM1_SD_HI
from synth_ultra.config import FM1_SD_LO as _FM1_SD_LO
from synth_ultra.config import FR as _FR
from synth_ultra.config import FV as _FV
from synth_ultra.config import HIST_JUMP as _HIST_JUMP
from synth_ultra.config import HIST_JUMP_Q as _HIST_JUMP_Q
from synth_ultra.config import HORIZON_SECONDS, NUM_PERCENTILES, REFERENCE_ASSET, asset_param
from synth_ultra.config import HOUR_EWMA_LAM as _HOUR_EWMA_LAM
from synth_ultra.config import IMB_EVENT as _IMB_EVENT
from synth_ultra.config import KA as _KA
from synth_ultra.config import KB as _KB
from synth_ultra.config import LIQ_HOUR_WIDEN as _LIQ_HOUR_WIDEN
from synth_ultra.config import LIQ_K as _LIQ_K
from synth_ultra.config import LIVE_HI as _LIVE_HI
from synth_ultra.config import LIVE_LO as _LIVE_LO
from synth_ultra.config import LIVE_MIN_WIDTH as _LIVE_MIN_WIDTH
from synth_ultra.config import LIVE_N as _LIVE_N
from synth_ultra.config import LOUD_BOOK as _LOUD_BOOK
from synth_ultra.config import LOUD_IMB as _LOUD_IMB
from synth_ultra.config import MAX_SIGMA_PX as _MAX_SIGMA_PX
from synth_ultra.config import MIN_PRICE as _MIN_PRICE
from synth_ultra.config import MIN_SIGMA_PX as _MIN_SIGMA_PX
from synth_ultra.config import MIX_Z as _MIX_Z
from synth_ultra.config import MU_ALPHA_CAP as _MU_ALPHA_CAP
from synth_ultra.config import MU_CLIP_SIGMAS as _MU_CLIP
from synth_ultra.config import MU_EXCESS as _MU_EXCESS
from synth_ultra.config import MU_FLOOR as _MU_FLOOR
from synth_ultra.config import MU_PULL as _MU_PULL
from synth_ultra.config import OFI_MS as _OFI_MS
from synth_ultra.config import QTY_LOG_DIV as _QTY_LOG_DIV
from synth_ultra.config import REG_B as _REG_B
from synth_ultra.config import REG_N as _REG_N
from synth_ultra.config import RV60_N as _RV60_N
from synth_ultra.config import SEAS_CLIP as _SEAS_CLIP
from synth_ultra.config import SEAS_QUIET_CAP as _SEAS_QUIET_CAP
from synth_ultra.config import SEAS_QUIET_PIVOT as _SEAS_QUIET_PIVOT
from synth_ultra.config import SEAS_QUIET_REL as _SEAS_QUIET_REL
from synth_ultra.config import SEAS_QUIET_SLOPE as _SEAS_QUIET_SLOPE
from synth_ultra.config import SH as _SH
from synth_ultra.config import SM as _SM
from synth_ultra.config import STALE_MS_CUT as _STALE_MS_CUT
from synth_ultra.config import TAU as _TAU
from synth_ultra.config import VOL_LOG_BIAS as _VOL_LOG_BIAS
from synth_ultra.config import VOL_LOG_FLOOR as _VOL_LOG_FLOOR
from synth_ultra.config import VOL_W as _VOL_W
from synth_ultra.config import WIDE_FLOOR as _WIDE_FLOOR
from synth_ultra.config import WIDE_POW as _WIDE_POW
from synth_ultra.constants import asset_spec, price_scale
from synth_ultra.features import extract, reference_price
from synth_ultra.season_table import SEAS_ABS as _SEAS_ABS
from synth_ultra.season_table import SEAS_REL as _SEAS_REL
from synth_ultra.stillness import apply as apply_stillness
from synth_ultra.stillness import still_features, taker_flow

_SQRT_H = math.sqrt(float(HORIZON_SECONDS))

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


# Weight vectors depend only on the series length, which is the same few
# thousand points on nearly every call.
_EWMA_W: dict[tuple[int, float], np.ndarray] = {}
_W_CACHE: dict[int, np.ndarray] = {}


def _ewma_abs(abs_r: np.ndarray, lam: float = _HOUR_EWMA_LAM) -> float:
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


_WIN_RET = _WIN_30_S - HORIZON_SECONDS
_BIN_W = 1.0 / _N_BINS
_BIN_EDGES = np.array([k * _BIN_W for k in range(_N_BINS)] + [1.0], dtype=np.float64)
_BIN_SPAN = np.diff(_BIN_EDGES)


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

    def vol_aware(self, fitted: float, px: float, log_floor: float = _VOL_LOG_FLOOR) -> float:
        """Log-blend the fitted scale with EWMA 10s moves, 60s vol and the 10s range."""
        if self.a10 is None or not math.isfinite(fitted) or fitted <= 0.0:
            return fitted
        fast = px * _ewma_abs(self.a10) / _Z_ABS
        if self.n > _RV60_N:
            tail = self.lc[-(_RV60_N + 1) :]
            rets = tail[1:] - tail[:-1]
            dev = rets - float(rets.sum()) / rets.size
            std = math.sqrt(float((dev * dev).sum()) / rets.size)
            rv60 = px * (std * _SQRT_H) / _Z_ABS
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
        pos = _HIST_JUMP_Q * (m - 1)
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
        pos = _CANDLE_Q * (m - 1)
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


def _regime_scale(
    sigma_px: float, close: np.ndarray, hour: _Hour | None = None, asset: str = REFERENCE_ASSET
) -> float:
    """Shrink a scale that is wide of the latest few minutes of 10s moves.

    ``s_recent`` is the mixture scale whose mean absolute deviation matches
    the last ``_REG_N`` overlapping 10-second dollar moves. The map
    ``clip(REG_A + REG_B * s_recent / sigma, REG_A, 1)`` was chosen so
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
    reg_a = asset_param(asset, "REG_A")
    return sigma * _clip(reg_a + _REG_B * ratio, reg_a, 1.0)




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
    fade = 1.0 / _clip(ratio, _FLEX_FADE_LO, _FLEX_FADE_HI)
    excess = max(abs(mu) / sigma - _MU_EXCESS, 0.0)
    gain = _clip(1.0 - _MU_PULL * excess * fade, _MU_FLOOR, 1.0)
    width = _clip(ratio**_WIDE_POW, _WIDE_FLOOR, 1.0)
    return mu * gain, sigma * width




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
    if ts.size < _BOOK_MIN_TICKS:
        return None
    bp = _f64_col(book.get("bid_price"))
    bq = _f64_col(book.get("bid_qty"))
    ap = _f64_col(book.get("ask_price"))
    aq = _f64_col(book.get("ask_qty"))
    n = int(min(ts.size, bp.size, bq.size, ap.size, aq.size))
    if n < _BOOK_MIN_TICKS:
        return None
    ts = ts[-n:]
    bp = bp[-n:]
    bq = bq[-n:]
    ap = ap[-n:]
    aq = aq[-n:]
    step = 1 if n <= _BOOK_PAIRS else (n + _BOOK_PAIRS - 1) // _BOOK_PAIRS
    ts_i = ts[::step]
    j = np.searchsorted(ts, ts_i + HORIZON_SECONDS * 1000, side="right") - 1
    ii = np.arange(0, n, step, dtype=np.int64)
    ok = (j > ii) & (ts[j] >= ts_i + _BOOK_MIN_GAP_MS)
    k = int(np.count_nonzero(ok))
    if k < _BOOK_MIN_PAIRS:
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
    return fitted * (realized / fitted) ** _BLEND_REALIZED_POW


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
            math.tanh(f["c1"] * _C1_TANH_SCALE),
            f["basis"] * vol10,
        ],
        dtype=np.float64,
    )


def _abs_vec(f: dict[str, float], vol10: float, vol_fast10: float) -> np.ndarray:
    rng = max(f["s_rng"], f["f_rng"])
    rv = max(f["s_rv"], f["f_rv"])
    log_fqty = math.log1p(max(f["f_qty2"], 0.0)) / _QTY_LOG_DIV
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
            math.log1p(max(f["s_qty2"], 0.0)) / _QTY_LOG_DIV,
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
            1.0 if f["stale_ms"] > _STALE_MS_CUT else 0.0,
        ],
        dtype=np.float64,
    )




def _slot_index(now_ms: int) -> int:
    """0..1007 index of the UTC weekday, hour, and 10-minute segment."""
    sec = int(now_ms) // 1000
    # 1970-01-01 was Thursday. datetime.weekday() is Monday=0.
    dow = ((sec // 86400) + 3) % 7
    sod = sec % 86400
    return (dow * 24 + sod // 3600) * 6 + (sod % 3600) // 600


def _season_scale(sigma_px: float, px: float, now_ms: int, asset: str = REFERENCE_ASSET) -> float:
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
    adj = _clip(asset_param(asset, "SEAS_W") * (math.log(seas) - math.log(sigma)), -_SEAS_CLIP, _SEAS_CLIP)
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


_EVENT_TENT = np.zeros(_TAU.shape, dtype=np.float64)
_T_LO, _T_Q1, _T_MID, _T_Q3, _T_HI = _EVENT_TENT_KNOTS
_left_band = (_TAU > _T_LO) & (_TAU < _T_MID)
_EVENT_TENT[_left_band] = np.where(
    _TAU[_left_band] <= _T_Q1,
    (_TAU[_left_band] - _T_LO) / (_T_Q1 - _T_LO),
    (_T_MID - _TAU[_left_band]) / (_T_MID - _T_Q1),
)
_right_band = (_TAU > _T_MID) & (_TAU < _T_HI)
_EVENT_TENT[_right_band] = np.where(
    _TAU[_right_band] <= _T_Q3,
    (_TAU[_right_band] - _T_MID) / (_T_Q3 - _T_MID),
    (_T_HI - _TAU[_right_band]) / (_T_HI - _T_Q3),
)
_EVENT_TENT[:11] = 0.0
_EVENT_TENT[49:51] = 0.0
_EVENT_TENT[89:] = 0.0
_EVENT_LEFT = _TAU < _T_MID


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
    asset: str = REFERENCE_ASSET,
) -> np.ndarray:
    """Mix the candle and mixture distributions on clock prompts.

    The mixture path scales the fitted drift down slightly and adds the last
    one-second move. Event prompts stay on that path. Clock prompts add the
    candle distribution in proportion to how quiet the recent hour has been,
    and how quiet the last minute is relative to that hour.
    """
    shift = (asset_param(asset, "MU_KEEP") - 1.0) * float(mu_px) + asset_param(asset, "C1_KEEP") * float(c1_px)
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
    sigma_px = h.vol_aware(sigma_px, px, log_floor=_VOL_LOG_FLOOR * scale)
    sigma_px = _clip(sigma_px, lo, hi)
    mu_px = _clip(mu_px, -_MU_CLIP * sigma_px, _MU_CLIP * sigma_px)
    # input.md: a buy liquidation pushes price up; depth bands must be
    # re-anchored to the live mid before they can shift the book.
    mu_px += (_LIQ_K * f["liq_dir"] + _BAND_K * f["band_imb"]) * sigma_px
    sigma_px *= 1.0 + _LIQ_HOUR_WIDEN * max(f["liq_hour"], 0.0)
    sigma_px = _clip(sigma_px, lo, hi)
    sigma_px = _regime_scale(sigma_px, closes, h, asset)
    sigma_px = _clip(sigma_px, lo, hi)
    mu_px = _clip(mu_px, -_MU_CLIP * sigma_px, _MU_CLIP * sigma_px)
    mu_px, sigma_px = _flex_linear(mu_px, sigma_px, closes, h)
    sigma_px = _clip(sigma_px, lo, hi)
    mu_px = _clip(mu_px, -_MU_CLIP * sigma_px, _MU_CLIP * sigma_px)
    sigma_px = _season_scale(sigma_px, px, now_ms, asset)
    sigma_px = _clip(sigma_px, lo, hi)
    mu_px += asset_param(asset, "OFI_K") * float(ofi) * sigma_px
    # Spot and futures top-of-book imbalance. On the prompts where the next
    # move is large, this sign is right about four times in five, and the
    # linear book coefficients were too small to use it.
    imb = 0.5 * (float(f["s_imb"]) + float(f["f_imb"]))
    imb_k = _IMB_EVENT if trigger != "time" else asset_param(asset, "IMB_K")
    mu_px += imb_k * imb * sigma_px
    mu_px = _clip(mu_px, -_MU_CLIP * sigma_px, _MU_CLIP * sigma_px)
    book_cap, book_pivot, book_slope = _season_book(now_ms)
    hist_w = asset_param(asset, "HIST_W")
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
    mixed = _combine(ladder, candle, mu_px, c1_px, p90, trigger, activity, scale, asset)
    if trigger != "time":
        mixed = _widen_event_center(mixed)
    last_mu_px = float(mu_px)
    last_anchor_px = float(px)
    return mixed




def _event_shift(
    q: np.ndarray,
    f: dict[str, float],
    fut: dict | None,
    now_ms: int,
    flow_sd: float = _FLOW_SD,
    fm1_sd: float = _FM1_SD,
    asset: str = REFERENCE_ASSET,
) -> np.ndarray:
    """Trade-print location: futures taker flow minus the spent 1s futures move."""
    flow = taker_flow((fut or {}).get("trades"), now_ms, _FLOW_MS)
    zf = _clip(flow / flow_sd, -_FLOW_CLIP, _FLOW_CLIP)
    denom = fm1_sd if math.isfinite(fm1_sd) and fm1_sd > 0.0 else _FM1_SD
    zm = _clip(float(f["f_m1"]) / denom, -_FM1_CLIP, _FM1_CLIP)
    delta = asset_param(asset, "FLOW_K") * zf + _FM1_K * zm
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
    asset = str(((payload or {}).get("prompt") or {}).get("asset") or REFERENCE_ASSET)
    return _ladder(px, 0.0, max(_FALLBACK_SIGMA_PX * price_scale(asset, px), _FALLBACK_SIGMA_REL * px))


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
        asset = str(prompt.get("asset") or REFERENCE_ASSET)
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




def _atom_band(asset: str, scale: float) -> float:
    """Prices this close to the anchor are the stillness point mass.

    BTC keeps the fitted two-cent band. On ETH, SOL, and XRP that band
    scaled by price falls through the tick, so the live-width step was
    smearing the atom off the touch.
    """
    if asset == REFERENCE_ASSET:
        return _ATOM
    tick = asset_spec(asset).tick
    return max(_ATOM * scale, tick)


def _live_fm1_sd(closes: np.ndarray) -> float:
    """Scale of a 1-second log move from this asset's own candle tail.

    The fitted ``_FM1_SD`` is BTC's. ETH, SOL, and XRP run hotter, so the
    same constant treats an ordinary print as a four-sigma futures move.
    """
    c = closes.reshape(-1) if isinstance(closes, np.ndarray) else np.asarray(closes, dtype=np.float64)
    if c.size > _FM1_LIVE_N + 1:
        c = c[-(_FM1_LIVE_N + 1) :]
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
    return _clip(sd, _FM1_SD_LO * _FM1_SD, _FM1_SD_HI * _FM1_SD)


def _live_width(
    q: np.ndarray,
    closes: np.ndarray,
    anchor: float,
    scale: float = 1.0,
    atom: float = _ATOM,
    asset: str = REFERENCE_ASSET,
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
    if not math.isfinite(ew) or ew <= 0.0 or width < _LIVE_MIN_WIDTH * scale or not math.isfinite(med):
        return q
    alpha = _clip((ew / width) ** asset_param(asset, "LIVE_POW"), _LIVE_LO, _LIVE_HI)
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




def _asset_width(q: np.ndarray, anchor: float, asset: str, scale: float) -> np.ndarray:
    """Open a non-BTC ladder. The stillness prices on the touch stay put."""
    alpha = asset_param(asset, "WIDTH")
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
        fm1_sd = _FM1_SD if asset == REFERENCE_ASSET else _live_fm1_sd(closes)
        q = _event_shift(
            q,
            f,
            (payload.get("venues") or {}).get("futures"),
            now_ms,
            flow_sd=_FLOW_SD / scale,
            fm1_sd=fm1_sd,
            asset=asset,
        )
    seas = float(_SEAS_ABS[_slot_index(now_ms)]) if now_ms >= 0 else 0.0
    x = still_features(payload, closes, now_ms, seas, 0.0, asset=asset)
    q = apply_stillness(q, anchor, x, event, asset=asset)
    q = _live_width(q, closes, anchor, scale, _atom_band(asset, scale), asset)
    q = _asset_width(q, anchor, asset, scale)
    if event:
        q = _trade_anchor(q, anchor, f["s_last_trade"], _atom_band(asset, scale))
    return q if _valid(q) else out


def _trade_anchor(q: np.ndarray, anchor: float, last_trade: float, band: float) -> np.ndarray:
    """Move the non-still prices of a trade-triggered ladder toward the last spot print.

    The quote can lag the print that fired the prompt. Prices farther than
    ``band`` from the anchor shift by ``EVENT_TRADE_ANCHOR`` of the print's gap
    to the anchor; the stillness mass stays on the quote.
    """
    k = _EVENT_TRADE_ANCHOR
    if k == 0.0 or not (last_trade > 0.0) or not math.isfinite(last_trade):
        return q
    shift = k * (float(last_trade) - float(anchor))
    if shift == 0.0 or not math.isfinite(shift):
        return q
    x = np.where(np.abs(q - anchor) > band, q + shift, q)
    x.sort()
    return x


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
