"""Quote stillness: the chance the spot microprice has not moved 10 s from now.

A third of scored prompts end within a cent of the current microprice: the
spot top of book simply does not change for ten seconds. When it does change,
it usually moves by dollars, not cents. The feature ladder is unimodal, so it
spreads that point mass over a band of several dollars around the center.

``still_features`` reads the live spot and futures quotes. Mid changes are
counted on a 100 ms grid over the last minute, together with the touch queue
against the taker flow hitting it and how often the price held inside one
tick on the 1-second candles. A logistic model turns these into P(still).
``mix_still`` then places that mass on the current microprice, moves the
remaining ladder out of the empty band around it, and keeps the mean shift.
"""

from __future__ import annotations

import math

import numpy as np

from synth_ultra.constants import BTC_REF_PX, NUM_PERCENTILES, asset_spec

_GRID_MS = 100
_GRID_N = 600
_GRID_OFF = np.arange(-(_GRID_N - 1) * _GRID_MS, 1, _GRID_MS, dtype=np.int64)
_WIN = 10_000 // _GRID_MS
# Changes of twice the mid. Spot moves in 0.01 ticks; futures in 0.1.
_SPOT_STEP = 0.008
_FUT_STEP = 0.08
_RANGE_TICK = 0.011
_RANGE_LONG = 1790
_RANGE_SHORT = 290

FEATURES = (
    "p0book",
    "pois60",
    "nchg10",
    "age",
    "minq",
    "q_flow",
    "spread",
    "fchg10",
    "fchg60",
    "p0range30m",
    "p0range5m",
    "seas",
    "m0",
)

_TAU = (2.0 * np.arange(1, NUM_PERCENTILES + 1) - 1.0) / 200.0
_M0 = FEATURES.index("m0")
_XBUF = np.empty(len(FEATURES), dtype=np.float64)
_R = np.empty(NUM_PERCENTILES, dtype=np.float64)
_OUT = np.empty(NUM_PERCENTILES, dtype=np.float64)
_POS = np.empty(NUM_PERCENTILES, dtype=np.float64)
_I0 = np.empty(NUM_PERCENTILES, dtype=np.int64)

# Logistic weights, intercept first, then FEATURES order. Fit by
# scripts/fit_stillness.py on the saved prompts; clock and trade-triggered
# prompts each have their own vector.
W_TIME = np.array(
    [
        -4.61846527451775,
        0.17034772984186247,
        0.2813703561617492,
        -0.15965766031065734,
        -0.04949323887833651,
        0.6605209888903941,
        0.13005700670299267,
        -20.758785373235124,
        -0.053866749363991946,
        -0.40563421464038446,
        1.9722942449642995,
        -0.21931392073370692,
        -0.4799491646634374,
        -1.0101467322622217,
    ],
    dtype=np.float64,
)
W_EVENT = np.array(
    [
        -5.293526458905508,
        1.2359207801555567,
        -0.7657115123577866,
        -0.35387194617541284,
        -0.2549507079896938,
        1.0323707288541988,
        0.19539287504904634,
        -1.8838370215557418,
        0.017228787450030474,
        -0.41789201765064954,
        -0.16554390432665644,
        1.3447109825428714,
        -0.48082099995511424,
        -8.983843865291522,
    ],
    dtype=np.float64,
)
# The added mass is GAIN * P(still) less the mass the ladder already puts
# within a cent of the price, capped at P_MAX. The remaining ladder widens by
# 1 + KAPPA * p, keeps SHIFT of its mean offset, and leaves a band of about
# HOLE widths around the price.
GAIN = 1.0
P_MAX = 0.9
KAPPA = 0.75
SHIFT = 1.0
HOLE = 0.05
# BTC's saved prompts were over-concentrated on the touch: a smaller gain and a
# wider residual ladder lowered every 24-hour window. ETH, SOL, and XRP keep
# the original fit; the same change raised at least one of their windows.
_GAIN_ASSET = {"BTC": 0.7}
_KAPPA_ASSET = {"BTC": 1.1}


def still_probability(x: np.ndarray, event: bool) -> float:
    w = W_EVENT if event else W_TIME
    z = float(w[0]) + float(np.dot(w[1:], x))
    if z >= 30.0:
        return 1.0
    if z <= -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def apply(
    q: np.ndarray, px: float, x: np.ndarray | None, event: bool, asset: str = "BTC"
) -> np.ndarray:
    """Add the stillness point mass to a finished ladder ``q``."""
    name = asset or "BTC"
    gain = _GAIN_ASSET.get(name, GAIN)
    kappa = _KAPPA_ASSET.get(name, KAPPA)
    if x is None or gain <= 0.0:
        return q
    near = asset_spec(name).tick
    m0 = float(np.count_nonzero(np.abs(q - px) < near)) / NUM_PERCENTILES
    x[_M0] = m0
    p = gain * still_probability(x, event) - m0
    if not (p > 0.0):
        return q
    p = min(p, P_MAX)
    med = 0.5 * (float(q[49]) + float(q[50]))
    width = float(np.abs(q - med).sum()) / NUM_PERCENTILES
    return mix_still(q, px, p, width, kappa, SHIFT, HOLE)


def _i64(x: object) -> np.ndarray:
    if isinstance(x, np.ndarray):
        a = x.reshape(-1)
        return a if a.dtype == np.int64 else a.astype(np.int64)
    if x is None:
        return np.empty(0, dtype=np.int64)
    return np.asarray(x, dtype=np.int64).reshape(-1)


def _f64(x: object) -> np.ndarray:
    if isinstance(x, np.ndarray):
        a = x.reshape(-1)
        return a if a.dtype == np.float64 else a.astype(np.float64)
    if x is None:
        return np.empty(0, dtype=np.float64)
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _mid_grid(book: dict | None, now_ms: int) -> np.ndarray | None:
    """Twice the mid at each 100 ms step of the last minute, oldest first.

    Steps before the first quote of the window are dropped.
    """
    if not book:
        return None
    ts = _i64(book.get("recv_ts_ms"))
    bp = _f64(book.get("bid_price"))
    ap = _f64(book.get("ask_price"))
    n = min(ts.size, bp.size, ap.size)
    if n < 2:
        return None
    ts = ts[-n:]
    idx = np.searchsorted(ts, now_ms + _GRID_OFF, side="right") - 1
    start = int(np.searchsorted(idx, 0, side="left"))
    if start >= idx.size - 1:
        return None
    idx = idx[start:]
    return bp[-n:][idx] + ap[-n:][idx]


def _changes(grid: np.ndarray, step: float) -> tuple[float, float, float]:
    """(changes in the minute, changes in the last 10 s, seconds since the last)."""
    chg = np.abs(grid[1:] - grid[:-1]) > step
    total = float(np.count_nonzero(chg))
    recent = float(np.count_nonzero(chg[-_WIN:]))
    if total > 0.0:
        last = int(chg.size - 1 - np.argmax(chg[::-1]))
        age = (chg.size - last) * (_GRID_MS / 1000.0)
    else:
        age = chg.size * (_GRID_MS / 1000.0)
    return total, recent, age


def taker_flow(trades: dict | None, now_ms: int, window_ms: int) -> float:
    """Taker buy minus taker sell quantity received in the last ``window_ms``."""
    buy, sell = _side_flow(trades, now_ms, window_ms)
    return buy - sell


def _side_flow(trades: dict | None, now_ms: int, window_ms: int) -> tuple[float, float]:
    """Taker buy and sell quantity received in the last ``window_ms``."""
    if not trades:
        return 0.0, 0.0
    ts = _i64(trades.get("recv_ts_ms"))
    q = _f64(trades.get("qty"))
    mk = trades.get("buyer_is_maker")
    mk = np.empty(0, dtype=bool) if mk is None else np.asarray(mk, dtype=bool).reshape(-1)
    n = min(ts.size, q.size, mk.size)
    if n == 0:
        return 0.0, 0.0
    i0 = int(np.searchsorted(ts[-n:], now_ms - window_ms, side="right"))
    q = q[-n:][i0:]
    if q.size == 0:
        return 0.0, 0.0
    sell = float(q[mk[-n:][i0:]].sum())
    return float(q.sum()) - sell, sell


def _qty_unit(asset: str, px: float) -> float:
    """Coin quantity to the notional units the stillness fit saw on BTC.

    A SOL or XRP queue is a large coin count and a small dollar count. Left
    in coins, ``log1p(qty)`` saturates the logistic and the point mass stays
    on the touch after the price has already left.
    """
    if str(asset or "BTC") == "BTC" or not math.isfinite(px) or px <= 0.0:
        return 1.0
    return px / BTC_REF_PX


def _asset_steps(asset: str) -> tuple[float, float, float, float]:
    """Spot step, futures step, candle-range tick, and spread scale.

    BTC keeps the fitted dollar thresholds. A one-tick move on ETH, SOL, or
    XRP is measured the same way, in that asset's own tick.
    """
    spec = asset_spec(asset or "BTC")
    tick_scale = spec.tick / 0.01
    fut_scale = spec.futures_tick / 0.1
    return _SPOT_STEP * tick_scale, _FUT_STEP * fut_scale, _RANGE_TICK * tick_scale, 0.01 / spec.tick


def _range_still(close: np.ndarray, tick: float = _RANGE_TICK) -> tuple[float, float]:
    """Share of 10-second candle windows whose closes stayed inside one tick."""
    c = close[-(_RANGE_LONG + 10) :]
    if c.size < 41:
        return 0.0, 0.0
    # Rolling 11-close max and min by doubling: windows of 2, 4, 8, then 8 + 3.
    hi = np.maximum(c[:-1], c[1:])
    lo = np.minimum(c[:-1], c[1:])
    for step in (2, 4, 3):
        hi = np.maximum(hi[:-step], hi[step:])
        lo = np.minimum(lo[:-step], lo[step:])
    held = (hi - lo) <= tick
    short = held[-_RANGE_SHORT:]
    return float(np.count_nonzero(held)) / held.size, float(np.count_nonzero(short)) / short.size


def _last(x: object) -> float:
    a = _f64(x)
    return float(a[-1]) if a.size else 0.0


def _qat(r: np.ndarray, t: np.ndarray, out: np.ndarray, sl: slice) -> None:
    """Write the 100-knot quantile function of ``r`` at probabilities ``t[sl]``."""
    pos = _POS[sl]
    np.subtract(t[sl], 0.005, out=pos)
    pos *= 100.0
    np.clip(pos, 0.0, 99.0, out=pos)
    i0 = _I0[sl]
    i0[:] = pos
    np.minimum(i0, 98, out=i0)
    np.subtract(pos, i0, out=pos)
    np.multiply(r[i0], 1.0 - pos, out=out[sl])
    out[sl] += r[i0 + 1] * pos


def mix_still(
    q: np.ndarray,
    px: float,
    p: float,
    width: float,
    kappa: float,
    shift: float,
    hole: float,
) -> np.ndarray:
    """Quantiles of ``p * delta(px) + (1 - p) * reshaped ladder``.

    The remaining ladder is moved away from ``px`` by ``shift * p / (1 - p)``
    of its center offset, so the mean shift survives the added point mass.
    It is widened by ``1 + kappa * p``, and pushed out of the band within
    about ``hole * width`` of ``px`` where moved prices rarely land.
    """
    if not (p > 0.0):
        return q
    med = 0.5 * (float(q[49]) + float(q[50]))
    sh = shift * p / (1.0 - p) * (med - px)
    r = _R
    np.add(q, sh, out=r)
    r += (kappa * p) * (r - (med + sh))
    h = hole * width
    if h > 0.0:
        d = r - px
        a = np.abs(d)
        r = px + np.sign(d) * (a + h * (1.0 - np.exp(-a / h)))
        _R[:] = r
        r = _R
    k = int(np.searchsorted(r, px, side="left"))
    if k <= 0:
        t_left = 0.0
    elif k >= NUM_PERCENTILES:
        t_left = 1.0
    else:
        lo = float(r[k - 1])
        hi = float(r[k])
        frac = (px - lo) / (hi - lo) if hi > lo else 0.0
        t_left = float(_TAU[k - 1]) + 0.01 * min(max(frac, 0.0), 1.0)
    keep = 1.0 - p
    t1 = keep * t_left
    t2 = t1 + p
    out = _OUT
    i1 = int(np.searchsorted(_TAU, t1, side="left"))
    i2 = int(np.searchsorted(_TAU, t2, side="right"))
    if i1:
        _qat(r, _TAU / keep, out, slice(0, i1))
    if i2 > i1:
        out[i1:i2] = px
    if i2 < NUM_PERCENTILES:
        _qat(r, (_TAU - p) / keep, out, slice(i2, NUM_PERCENTILES))
    np.maximum.accumulate(out, out=out)
    return out.copy()


def still_features(
    payload: dict,
    close: np.ndarray,
    now_ms: int,
    seas_abs: float,
    m0: float,
    asset: str = "BTC",
) -> np.ndarray | None:
    """Feature vector in ``FEATURES`` order. None without a minute of spot quotes."""
    venues = payload.get("venues") or {}
    spot = venues.get("spot") or {}
    fut = venues.get("futures") or {}
    book = spot.get("book_ticker")
    grid = _mid_grid(book, now_ms)
    if grid is None or grid.size <= _WIN + 1:
        return None
    spot_step, fut_step, range_tick, spread_scale = _asset_steps(asset)
    n60, n10, age = _changes(grid, spot_step)
    held = np.abs(grid[_WIN:] - grid[:-_WIN]) <= spot_step
    p0book = float(np.count_nonzero(held)) / held.size
    bq = _last(book.get("bid_qty"))
    aq = _last(book.get("ask_qty"))
    bid = _last(book.get("bid_price"))
    ask = _last(book.get("ask_price"))
    spread = ask - bid
    unit = _qty_unit(asset, 0.5 * (bid + ask))
    bq_u = max(bq, 0.0) * unit
    aq_u = max(aq, 0.0) * unit
    buy, sell = _side_flow(spot.get("trades"), now_ms, 10_000)
    buy_u = max(buy, 0.0) * unit
    sell_u = max(sell, 0.0) * unit
    q_flow = min(
        math.log((bq_u + 1e-3) / (sell_u + 0.05)),
        math.log((aq_u + 1e-3) / (buy_u + 0.05)),
    )
    fgrid = _mid_grid(fut.get("book_ticker"), now_ms)
    if fgrid is not None and fgrid.size > 2:
        f60, f10, _fage = _changes(fgrid, fut_step)
    else:
        f60 = f10 = 0.0
    r30, r5 = _range_still(close, range_tick)
    x = _XBUF
    x[0] = p0book
    x[1] = math.exp(-n60 / 6.0)
    x[2] = math.log1p(n10)
    x[3] = math.log1p(age)
    x[4] = math.log1p(max(min(bq_u, aq_u), 0.0))
    x[5] = q_flow
    x[6] = min(max(spread * spread_scale, 0.0), 1.0)
    x[7] = math.log1p(f10)
    x[8] = math.log1p(f60)
    x[9] = r30
    x[10] = r5
    x[11] = math.log(seas_abs) if seas_abs > 0.0 else 0.0
    x[12] = m0
    return x
