"""Extract cheap, vectorized features from a Synth Ultra payload.

Adds book-ticker momentum/range, short trade flow, and fast EWMA vol for
the CRPS-tuned location/scale model. Every column is read as a view or a
short tail, so the cost does not grow with a busy 60-second tape.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from synth_ultra.constants import HORIZON_SECONDS


def microprice(bid_p: float, bid_q: float, ask_p: float, ask_q: float) -> float:
    den = bid_q + ask_q
    if den <= 0.0 or not math.isfinite(den):
        return 0.5 * (bid_p + ask_p)
    return (bid_p * ask_q + ask_p * bid_q) / den


def _f64(x: Any) -> np.ndarray:
    """1-D float64 view of a payload column. Lists are converted once."""
    if isinstance(x, np.ndarray):
        a = x.reshape(-1)
        return a if a.dtype == np.float64 else a.astype(np.float64)
    if x is None:
        return np.empty(0, dtype=np.float64)
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _i64(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        a = x.reshape(-1)
        return a if a.dtype == np.int64 else a.astype(np.int64)
    if x is None:
        return np.empty(0, dtype=np.int64)
    return np.asarray(x, dtype=np.int64).reshape(-1)


def _bool(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        a = x.reshape(-1)
        return a if a.dtype == np.bool_ else a.astype(bool)
    if x is None:
        return np.empty(0, dtype=bool)
    return np.asarray(x, dtype=bool).reshape(-1)


def _last_f64(x: Any, default: float = 0.0) -> float:
    """Last element only — avoids materializing a full book-ticker column."""
    if x is None:
        return default
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return default
        return float(x.reshape(-1)[-1])
    try:
        n = len(x)  # type: ignore[arg-type]
    except TypeError:
        try:
            return float(x)
        except (TypeError, ValueError):
            return default
    if n == 0:
        return default
    return float(x[n - 1])  # type: ignore[index]


def last_book_micro(book_ticker: dict | None) -> tuple[float, float, float]:
    """Return (microprice, mid, spread) from the latest book-ticker row."""
    if not book_ticker:
        return 0.0, 0.0, 0.0
    bp = _last_f64(book_ticker.get("bid_price"))
    bq = _last_f64(book_ticker.get("bid_qty"))
    ap = _last_f64(book_ticker.get("ask_price"))
    aq = _last_f64(book_ticker.get("ask_qty"))
    if bp <= 0.0 and ap <= 0.0:
        return 0.0, 0.0, 0.0
    mid = 0.5 * (bp + ap)
    spread = max(ap - bp, 0.0)
    return microprice(bp, bq, ap, aq), mid, spread


def _side_sizes(levels: Any, n: int = 5) -> tuple[float, float] | None:
    """(all-level size, top-``n`` size) of one book side, or None when empty."""
    if levels is None:
        return None
    arr = levels if isinstance(levels, np.ndarray) else np.asarray(levels, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return None
    qty = arr[:, 1]
    return float(qty.sum()), float(qty[:n].sum())


def depth_imbalances(snapshot: dict | None, n: int = 5) -> tuple[float, float]:
    """All-level and top-``n`` resting-size imbalance of one depth snapshot."""
    if not snapshot:
        return 0.0, 0.0
    bids = _side_sizes(snapshot.get("bids"), n)
    asks = _side_sizes(snapshot.get("asks"), n)
    if bids is None or asks is None:
        return 0.0, 0.0
    tot = bids[0] + asks[0]
    full = (bids[0] - asks[0]) / tot if tot > 0.0 else 0.0
    top = bids[1] + asks[1]
    return full, ((bids[1] - asks[1]) / top if top > 0.0 else 0.0)


def depth_imbalance(snapshot: dict | None) -> float:
    return depth_imbalances(snapshot)[0]


def top_imbalance(snapshot: dict | None, n: int = 5) -> float:
    return depth_imbalances(snapshot, n)[1]


def trade_flows(trades: dict | None, last_n: int = 50, window_ms: int = 2000) -> tuple[float, float, float, float]:
    """Signed taker imbalance: whole tape, last ``last_n`` prints, last ``window_ms``.

    Returns ``(full, last_n, window, window_qty)``. ``buyer_is_maker`` True
    means the aggressor sold.
    """
    if not trades:
        return 0.0, 0.0, 0.0, 0.0
    q = _f64(trades.get("qty"))
    mk = _bool(trades.get("buyer_is_maker"))
    n = min(q.size, mk.size)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    q = q[-n:]
    signed = np.where(mk[-n:], -q, q)
    tot = float(q.sum())
    full = float(signed.sum()) / tot if tot > 0.0 else 0.0
    k = min(n, last_n)
    tot_k = float(q[-k:].sum())
    fast = float(signed[-k:].sum()) / tot_k if tot_k > 0.0 else 0.0
    ts = _i64(trades.get("recv_ts_ms"))
    m = min(n, ts.size)
    if m == 0:
        return full, fast, 0.0, 0.0
    ts = ts[-m:]
    i0 = int(np.searchsorted(ts, int(ts[-1]) - int(window_ms), side="left"))
    q_w = q[n - m + i0 :]
    tot_w = float(q_w.sum())
    if tot_w <= 0.0:
        return full, fast, 0.0, 0.0
    return full, fast, float(signed[n - m + i0 :].sum()) / tot_w, tot_w


def trade_imbalance(trades: dict | None, last_n: int | None = None) -> float:
    full, fast, _w, _q = trade_flows(trades, last_n=last_n or 1 << 62)
    return fast if last_n is not None else full


def trade_flow_window(trades: dict | None, window_ms: int = 2000) -> tuple[float, float]:
    """Signed flow imbalance and total qty over the last ``window_ms``."""
    _full, _fast, flow, qty = trade_flows(trades, window_ms=window_ms)
    return flow, qty


# Cached EWMA weights for the candle lengths we actually use (avoids alloc per call).
_EWMA_W94_120 = (1.0 - 0.94) * (0.94 ** np.arange(119, -1, -1, dtype=np.float64))
_EWMA_W94_120 /= _EWMA_W94_120.sum()
_EWMA_W85_90 = (1.0 - 0.85) * (0.85 ** np.arange(89, -1, -1, dtype=np.float64))
_EWMA_W85_90 /= _EWMA_W85_90.sum()


def ewma_vol(returns: np.ndarray, lam: float = 0.94) -> float:
    if returns.size == 0:
        return 0.0
    r2 = returns * returns
    n = r2.size
    if lam == 0.94 and n == _EWMA_W94_120.size:
        return math.sqrt(float(np.dot(_EWMA_W94_120, r2)))
    if lam == 0.85 and n == _EWMA_W85_90.size:
        return math.sqrt(float(np.dot(_EWMA_W85_90, r2)))
    w = (1.0 - lam) * lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
    w /= w.sum()
    return math.sqrt(float(np.dot(w, r2)))


def _close_tail(venue: dict | None, n: int) -> np.ndarray:
    """Last ``n`` closes only — skips scanning the full hour of candles."""
    if not venue:
        return np.empty(0, dtype=np.float64)
    candles = venue.get("candles_1s") or {}
    ohlcv = candles.get("ohlcv")
    if ohlcv is None:
        return np.empty(0, dtype=np.float64)
    arr = ohlcv if isinstance(ohlcv, np.ndarray) else np.asarray(ohlcv, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 4:
        return np.empty(0, dtype=np.float64)
    return np.asarray(arr[-n:, 3], dtype=np.float64)


_TAIL = max(HORIZON_SECONDS + 1, 121)


def _tail_stats(close: np.ndarray) -> tuple[float, float, float, float, float]:
    """(1s EWMA vol, fast 1s EWMA vol, 10s momentum, 1s and 3s log return)."""
    if close.size < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    lc = np.log(np.maximum(close, 1e-12))
    rets = lc[1:] - lc[:-1]
    vol = ewma_vol(rets[-120:])
    vol_fast = ewma_vol(rets[-90:], lam=0.85)
    mom = 0.0
    if close.size > HORIZON_SECONDS:
        c0 = float(close[-(HORIZON_SECONDS + 1)])
        c1 = float(close[-1])
        mom = math.log(c1 / c0) if c0 > 0.0 and c1 > 0.0 else 0.0
    last = float(close[-1])
    r1 = 0.0
    prev = float(close[-2])
    if prev > 0.0 and last > 0.0:
        r1 = math.log(last / prev)
    r3 = 0.0
    if close.size >= 4:
        prev3 = float(close[-4])
        if prev3 > 0.0 and last > 0.0:
            r3 = math.log(last / prev3)
    return vol, vol_fast, mom, r1, r3


def venue_vol_and_momentum(venue: dict | None) -> tuple[float, float]:
    """1s EWMA vol and last-horizon log return. Falls back to zeros."""
    vol, _fast, mom, _r1, _r3 = _tail_stats(_close_tail(venue, _TAIL))
    return vol, mom


_NO_BOOK = {"imb": 0.0, "m1": 0.0, "m3": 0.0, "m5": 0.0, "rv": 0.0, "rng": 0.0}


def book_ticker_stats(book_ticker: dict | None, lookback_ms: int = 12_000) -> dict[str, float]:
    """Microprice momentum, range, and realized vol from recent book ticker.

    Avoids building a full microprice series on busy 10k–60k-tick windows
    (FAQ.md): only timestamps are scanned, then a few indexed microprices.
    """
    if not book_ticker:
        return dict(_NO_BOOK)
    ts_all = _i64(book_ticker.get("recv_ts_ms"))
    bp = _f64(book_ticker.get("bid_price"))
    bq = _f64(book_ticker.get("bid_qty"))
    ap = _f64(book_ticker.get("ask_price"))
    aq = _f64(book_ticker.get("ask_qty"))
    n = min(ts_all.size, bp.size, bq.size, ap.size, aq.size)
    if n == 0:
        return dict(_NO_BOOK)
    ts_all = ts_all[-n:]
    now = int(ts_all[-1])
    i0 = int(np.searchsorted(ts_all, now - lookback_ms, side="right"))
    ts = ts_all[i0:]
    bp = bp[-n:][i0:]
    bq = bq[-n:][i0:]
    ap = ap[-n:][i0:]
    aq = aq[-n:][i0:]
    m = ts.size
    if m == 0:
        return dict(_NO_BOOK)

    def _micro(i: int) -> float:
        b = float(bp[i])
        bqi = float(bq[i])
        a = float(ap[i])
        aqi = float(aq[i])
        den = bqi + aqi
        if den <= 0.0:
            return 0.5 * (b + a)
        return (b * aqi + a * bqi) / den

    last = m - 1
    bq_l = float(bq[last])
    aq_l = float(aq[last])
    den_last = bq_l + aq_l
    out = dict(_NO_BOOK)
    out["imb"] = (bq_l - aq_l) / den_last if den_last > 0.0 else 0.0
    micro_last = _micro(last)
    marks = np.searchsorted(ts, np.array([now - 1000, now - 3000, now - 5000, now - 10_000]), side="right")
    for key, i in zip(("m1", "m3", "m5"), marks[:3].tolist()):
        i -= 1
        if i < 0:
            continue
        a = _micro(i)
        if a > 0.0 and micro_last > 0.0:
            out[key] = math.log(micro_last / a)

    i10 = int(marks[3])
    # At most ~512 microprices for rv/rng, evenly spaced in the 10s window.
    span = m - i10
    if span > 5:
        if span <= _N_SAMP:
            idxs = slice(i10, m)
        else:
            idxs = i10 + _even_index(span)
        sbq = bq[idxs]
        saq = aq[idxs]
        samples = (bp[idxs] * saq + ap[idxs] * sbq) / np.maximum(sbq + saq, 1e-12)
        lo = float(samples.min())
        if lo <= 0.0:
            samples = samples[samples > 0.0]
        if samples.size > 5:
            ls = np.log(samples)
            rets = ls[1:] - ls[:-1]
            k = rets.size
            dev = rets - float(rets.sum()) / k
            out["rv"] = math.sqrt(float((dev * dev).sum()) / k) * math.sqrt(max(k, 1))
            lo = float(samples.min())
            hi = float(samples.max())
            if lo > 0.0:
                out["rng"] = math.log(hi / lo)
    return out


_N_SAMP = 512
_SAMP_STEPS = np.arange(_N_SAMP, dtype=np.float64)


def _even_index(span: int) -> np.ndarray:
    """``np.linspace(0, span - 1, 512, dtype=np.int64)`` without its overhead."""
    y = _SAMP_STEPS * ((span - 1) / (_N_SAMP - 1))
    y[-1] = span - 1
    return y.astype(np.int64)


def _as_int_scalar(v: Any) -> int | None:
    """Coerce msgpack/numpy scalars (and size-1 arrays) to int. Live payloads vary."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        arr = np.asarray(v)
        if arr.size == 0:
            return None
        return int(arr.reshape(-1)[-1])
    except (TypeError, ValueError, OverflowError):
        return None


def liquidation_signals(liquidations: dict | None, now_ms: int) -> tuple[float, float]:
    """Signed 10s pressure and 1h activity from futures liquidations.

    ``is_buy`` is the liquidation order's side, not the trapped position.
    True means the exchange is buying to close a short, which pushes price up.
    Returns ``(direction, hour_activity)``, each in ``[-1, 1]`` and ``[0, 1]``.
    """
    if not liquidations or int(now_ms) <= 0:
        return 0.0, 0.0
    recv = _i64(liquidations.get("recv_ts_ms"))
    if recv.size == 0:
        return 0.0, 0.0
    qty = _f64(liquidations.get("qty"))
    buy = _bool(liquidations.get("is_buy"))
    n = min(recv.size, qty.size, buy.size)
    if n == 0:
        return 0.0, 0.0
    recv = recv[-n:]
    qty = qty[-n:]
    buy = buy[-n:]
    now = int(now_ms)
    hour = (recv >= now - 3_600_000) & (recv <= now) & np.isfinite(qty) & (qty > 0.0)
    if not hour.any():
        return 0.0, 0.0
    q = qty[hour]
    r = recv[hour]
    recent = r >= now - 10_000
    signed = 0.0
    if recent.any():
        qr = q[recent]
        signed = float(np.where(buy[hour][recent], qr, -qr).sum())
    return math.tanh(signed / 0.25), math.tanh(float(q.sum()) / 1.0)


_BAND_IDX: dict[int, np.ndarray] = {}


def depth_band_imbalance(
    bands: dict | None,
    mid: float,
    now_ms: int,
    reach_usd: float = 200.0,
) -> float:
    """Order-book imbalance from spot depth bands, re-anchored to the live mid.

    Band 0 is around ``anchor_price`` at snapshot time, not around the current
    price. A ``recv_ts_ms`` of 0, or a snapshot older than 90s, is unused.
    """
    if not bands or mid <= 0.0:
        return 0.0
    recv = int(bands.get("recv_ts_ms") or 0)
    if recv <= 0:
        return 0.0
    if int(now_ms) > 0 and int(now_ms) - recv > 90_000:
        return 0.0
    anchor = float(bands.get("anchor_price") or 0.0)
    bucket = float(bands.get("bucket_usd") or 0.0)
    if anchor <= 0.0 or bucket <= 0.0 or not math.isfinite(anchor):
        return 0.0
    bids = _f64(bands.get("bids"))
    asks = _f64(bands.get("asks"))
    if bids.size == 0 and asks.size == 0:
        return 0.0
    bid_sz = 0.0
    ask_sz = 0.0
    if bids.size:
        idx = _band_index(bids.size)
        bid_lo = anchor - (idx + 1.0) * bucket
        bid_hi = anchor - idx * bucket
        bid_sz = float(bids[(bid_hi > mid - reach_usd) & (bid_lo < mid)].sum())
    if asks.size:
        idx = _band_index(asks.size)
        ask_lo = anchor + idx * bucket
        ask_hi = anchor + (idx + 1.0) * bucket
        ask_sz = float(asks[(ask_lo < mid + reach_usd) & (ask_hi > mid)].sum())
    tot = bid_sz + ask_sz
    if tot <= 0.0 or not math.isfinite(tot):
        return 0.0
    return (bid_sz - ask_sz) / tot


def _band_index(n: int) -> np.ndarray:
    idx = _BAND_IDX.get(n)
    if idx is None:
        idx = np.arange(n, dtype=np.float64)
        if len(_BAND_IDX) < 4:
            _BAND_IDX[n] = idx
    return idx


def last_event_age_ms(venue: dict | None, now_ms: int) -> float:
    if not venue:
        return 0.0
    times = venue.get("last_event_times") or {}
    latest = None
    for v in times.values():
        t = _as_int_scalar(v)
        if t is not None and (latest is None or t > latest):
            latest = t
    if latest is None:
        return 0.0
    return float(max(0, int(now_ms) - latest))


def _top_px(levels: Any) -> float:
    """Best price on one depth side, or 0 when that side is missing."""
    if levels is None:
        return 0.0
    arr = levels if isinstance(levels, np.ndarray) else np.asarray(levels, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 1:
        return 0.0
    px = float(arr[0, 0])
    return px if math.isfinite(px) and px > 0.0 else 0.0


def _depth_mid(snapshot: dict | None) -> float:
    """Mid of the best depth quote. The book ticker is not the only top of book."""
    if not snapshot:
        return 0.0
    bid = _top_px(snapshot.get("bids"))
    ask = _top_px(snapshot.get("asks"))
    if bid > 0.0 and ask > 0.0:
        return 0.5 * (bid + ask)
    return bid if bid > 0.0 else ask


def _last_trade_px(trades: dict | None) -> float:
    if not trades:
        return 0.0
    px = trades.get("price")
    if px is None:
        return 0.0
    arr = px if isinstance(px, np.ndarray) else np.asarray(px, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    last = float(arr.reshape(-1)[-1])
    return last if math.isfinite(last) and last > 0.0 else 0.0


def _last_close(venue: dict | None) -> float:
    close = _close_tail(venue, 1)
    if close.size == 0:
        return 0.0
    last = float(close[-1])
    return last if math.isfinite(last) and last > 0.0 else 0.0


def reference_price(spot: dict | None, fut: dict | None) -> float:
    """Spot microprice, or the next quote that sits on the same scale.

    The scored price is the spot book. When that tape and the spot candles
    are empty, depth, the futures book, and the futures closes still name
    the market. A stand-in of 1 is used only when none of those exist.
    A candidate more than a factor of two from the candle close is a bad tick.
    """
    spot = spot or {}
    fut = fut or {}
    spot_micro, spot_mid, _ = last_book_micro(spot.get("book_ticker"))
    fut_micro, fut_mid, _ = last_book_micro(fut.get("book_ticker"))
    spot_close = _last_close(spot)
    fut_close = _last_close(fut)
    ref = spot_close if spot_close > 0.0 else fut_close
    candidates = (
        spot_micro,
        spot_mid,
        fut_micro,
        fut_mid,
        _depth_mid(spot.get("depth_latest")),
        _depth_mid(fut.get("depth_latest")),
        spot_close,
        fut_close,
        _last_trade_px(spot.get("trades")),
        _last_trade_px(fut.get("trades")),
    )
    for px in candidates:
        if not math.isfinite(px) or px <= 0.0:
            continue
        if ref > 0.0 and abs(math.log(px / ref)) > math.log(2.0):
            continue
        return float(px)
    return ref if ref > 0.0 else 1.0


def extract(payload: dict) -> dict[str, float]:
    prompt = payload.get("prompt") or {}
    venues = payload.get("venues") or {}
    spot = venues.get("spot") or {}
    fut = venues.get("futures") or {}
    now_ms = int(prompt.get("current_time_ms") or 0)

    spot_micro, spot_mid, spot_spread = last_book_micro(spot.get("book_ticker"))
    fut_micro, _fut_mid, _ = last_book_micro(fut.get("book_ticker"))

    spot_complete = bool((spot.get("candles_1s") or {}).get("complete_history", True))
    fut_complete = bool((fut.get("candles_1s") or {}).get("complete_history", False))

    close = _close_tail(spot, _TAIL)
    spot_vol, vol_fast, spot_mom, c1, c3 = _tail_stats(close)
    if close.size < 2:
        vol_fast = spot_vol
    if fut_complete:
        fut_vol, fut_mom = venue_vol_and_momentum(fut)
    else:
        fut_vol, fut_mom = spot_vol, spot_mom

    vol_1s = 0.6 * spot_vol + 0.4 * fut_vol if fut_vol > 0.0 else spot_vol
    mom = 0.5 * spot_mom + 0.5 * fut_mom

    s_obi, s_top = depth_imbalances(spot.get("depth_latest"))
    f_obi, f_top = depth_imbalances(fut.get("depth_latest"))
    obi = 0.5 * s_obi + 0.5 * f_obi
    top_obi = 0.5 * s_top + 0.5 * f_top
    s_full, s_fast, s_flow2, s_qty2 = trade_flows(spot.get("trades"))
    f_full, f_fast, f_flow2, f_qty2 = trade_flows(fut.get("trades"))
    flow = 0.5 * s_full + 0.5 * f_full
    flow_fast = 0.5 * s_fast + 0.5 * f_fast

    s_bt = book_ticker_stats(spot.get("book_ticker"))
    f_bt = book_ticker_stats(fut.get("book_ticker"))

    px = reference_price(spot, fut)

    ref = spot_mid if spot_mid > 0.0 else px
    band_imb = depth_band_imbalance(spot.get("depth_bands"), ref, now_ms)
    liq_dir, liq_hour = liquidation_signals(fut.get("liquidations"), now_ms)

    basis = 0.0
    if spot_micro > 0.0 and fut_micro > 0.0:
        basis = (fut_micro - spot_micro) / spot_micro

    spread_rel = (spot_spread / px) if px > 0.0 else 0.0
    stale_ms = max(last_event_age_ms(spot, now_ms), last_event_age_ms(fut, now_ms))

    return {
        "price": float(px),
        "mid": float(spot_mid if spot_mid > 0.0 else px),
        "vol_1s": float(vol_1s),
        "vol_fast": float(vol_fast),
        "momentum": float(mom),
        "c1": float(c1),
        "c3": float(c3),
        "obi": float(obi),
        "top_obi": float(top_obi),
        "flow": float(flow),
        "flow_fast": float(flow_fast),
        "s_flow2": float(s_flow2),
        "f_flow2": float(f_flow2),
        "s_qty2": float(s_qty2),
        "f_qty2": float(f_qty2),
        "s_imb": float(s_bt["imb"]),
        "f_imb": float(f_bt["imb"]),
        "s_m1": float(s_bt["m1"]),
        "f_m1": float(f_bt["m1"]),
        "s_m3": float(s_bt["m3"]),
        "f_m3": float(f_bt["m3"]),
        "s_m5": float(s_bt["m5"]),
        "f_m5": float(f_bt["m5"]),
        "s_rv": float(s_bt["rv"]),
        "f_rv": float(f_bt["rv"]),
        "s_rng": float(s_bt["rng"]),
        "f_rng": float(f_bt["rng"]),
        "basis": float(basis),
        "spread_rel": float(spread_rel),
        "stale_ms": float(stale_ms),
        "spot_complete": float(spot_complete),
        "fut_complete": float(fut_complete),
        "liq_dir": float(liq_dir),
        "liq_hour": float(liq_hour),
        "band_imb": float(band_imb),
    }
