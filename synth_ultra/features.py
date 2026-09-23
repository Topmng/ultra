"""Extract cheap, vectorized features from a Synth Ultra payload.

Adds book-ticker momentum/range, short trade flow, and fast EWMA vol for
the CRPS-tuned location/scale model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from synth_ultra.constants import HORIZON_SECONDS


def _as_f64(x: Any, default: float = 0.0) -> np.ndarray:
    if x is None:
        return np.array([default], dtype=np.float64)
    arr = np.asarray(x)
    if arr.size == 0:
        return np.array([default], dtype=np.float64)
    return arr.astype(np.float64, copy=False)


def _as_bool(x: Any) -> np.ndarray:
    if x is None:
        return np.array([], dtype=bool)
    return np.asarray(x, dtype=bool)


def microprice(bid_p: float, bid_q: float, ask_p: float, ask_q: float) -> float:
    den = bid_q + ask_q
    if den <= 0.0 or not np.isfinite(den):
        return 0.5 * (bid_p + ask_p)
    return (bid_p * ask_q + ask_p * bid_q) / den


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


def depth_imbalance(snapshot: dict | None) -> float:
    if not snapshot:
        return 0.0
    bids = np.asarray(snapshot.get("bids", []), dtype=np.float64)
    asks = np.asarray(snapshot.get("asks", []), dtype=np.float64)
    if bids.size == 0 or asks.size == 0:
        return 0.0
    bid_sz = float(np.sum(bids[:, 1])) if bids.ndim == 2 else 0.0
    ask_sz = float(np.sum(asks[:, 1])) if asks.ndim == 2 else 0.0
    tot = bid_sz + ask_sz
    if tot <= 0.0:
        return 0.0
    return (bid_sz - ask_sz) / tot


def top_imbalance(snapshot: dict | None, n: int = 5) -> float:
    if not snapshot:
        return 0.0
    bids = np.asarray(snapshot.get("bids", []), dtype=np.float64)
    asks = np.asarray(snapshot.get("asks", []), dtype=np.float64)
    if bids.ndim != 2 or asks.ndim != 2 or bids.size == 0 or asks.size == 0:
        return 0.0
    bid_sz = float(np.sum(bids[:n, 1]))
    ask_sz = float(np.sum(asks[:n, 1]))
    tot = bid_sz + ask_sz
    if tot <= 0.0:
        return 0.0
    return (bid_sz - ask_sz) / tot


def trade_imbalance(trades: dict | None, last_n: int | None = None) -> float:
    if not trades:
        return 0.0
    qty = trades.get("qty")
    maker = trades.get("buyer_is_maker")
    if qty is None or maker is None:
        return 0.0
    if isinstance(qty, np.ndarray):
        q = qty.reshape(-1)
    else:
        q = np.asarray(qty, dtype=np.float64).reshape(-1)
    if isinstance(maker, np.ndarray):
        mk = maker.reshape(-1)
    else:
        mk = np.asarray(maker, dtype=bool).reshape(-1)
    n = min(q.size, mk.size)
    if n == 0:
        return 0.0
    if last_n is not None:
        n = min(n, last_n)
    q = q[-n:]
    mk = mk[-n:]
    tot = float(np.sum(q))
    if tot <= 0.0:
        return 0.0
    signed = float(np.sum(np.where(mk, -q, q)))
    return signed / tot


def trade_flow_window(trades: dict | None, window_ms: int = 2000) -> tuple[float, float]:
    """Signed flow imbalance and total qty over the last ``window_ms``."""
    if not trades:
        return 0.0, 0.0
    ts = _as_f64(trades.get("recv_ts_ms"))
    qty = _as_f64(trades.get("qty"))
    maker = _as_bool(trades.get("buyer_is_maker"))
    if ts.size == 0 or qty.size == 0 or maker.size == 0:
        return 0.0, 0.0
    n = min(ts.size, qty.size, maker.size)
    ts = ts[-n:]
    qty = qty[-n:]
    maker = maker[-n:]
    cutoff = float(ts[-1]) - float(window_ms)
    mask = ts >= cutoff
    if not np.any(mask):
        return 0.0, 0.0
    q = qty[mask]
    mk = maker[mask]
    tot = float(np.sum(q))
    if tot <= 0.0:
        return 0.0, 0.0
    signed = float(np.sum(np.where(mk, -q, q)))
    return signed / tot, tot


def log_returns(close: np.ndarray) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    close = close[np.isfinite(close) & (close > 0.0)]
    if close.size < 2:
        return np.empty(0, dtype=np.float64)
    return np.diff(np.log(close))


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
        return float(np.sqrt(np.dot(_EWMA_W94_120, r2)))
    if lam == 0.85 and n == _EWMA_W85_90.size:
        return float(np.sqrt(np.dot(_EWMA_W85_90, r2)))
    w = (1.0 - lam) * lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
    w /= w.sum()
    return float(np.sqrt(np.dot(w, r2)))


def candle_close(venue: dict | None) -> np.ndarray:
    if not venue:
        return np.empty(0, dtype=np.float64)
    candles = venue.get("candles_1s") or {}
    ohlcv = candles.get("ohlcv")
    if ohlcv is None:
        return np.empty(0, dtype=np.float64)
    arr = ohlcv if isinstance(ohlcv, np.ndarray) else np.asarray(ohlcv, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 4:
        return np.empty(0, dtype=np.float64)
    return arr[:, 3]


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


def venue_vol_and_momentum(venue: dict | None) -> tuple[float, float]:
    """1s EWMA vol and last-horizon log return. Falls back to zeros."""
    # Need HORIZON+1 for momentum and 121 closes for 120 returns.
    close = _close_tail(venue, max(HORIZON_SECONDS + 1, 121))
    if close.size < 2:
        return 0.0, 0.0
    # Trust candle closes; avoid a full finite-mask copy on the hot path.
    rets = np.diff(np.log(np.maximum(close[-(121):], 1e-12)))
    vol_1s = ewma_vol(rets[-120:] if rets.size else rets)
    if close.size > HORIZON_SECONDS:
        c0 = float(close[-(HORIZON_SECONDS + 1)])
        c1 = float(close[-1])
        mom = float(np.log(c1 / c0)) if c0 > 0.0 and c1 > 0.0 else 0.0
    else:
        mom = 0.0
    return vol_1s, mom


def _col_tail(arr: Any, n: int, i0: int) -> np.ndarray:
    """Aligned ``arr[-n:][i0:]`` view/copy as float64."""
    if isinstance(arr, np.ndarray):
        a = arr.reshape(-1)[-n:]
        sl = a[i0:]
        return sl if sl.dtype == np.float64 else np.asarray(sl, dtype=np.float64)
    a = np.asarray(arr, dtype=np.float64).reshape(-1)[-n:]
    return a[i0:]


def book_ticker_stats(book_ticker: dict | None, lookback_ms: int = 12_000) -> dict[str, float]:
    """Microprice momentum, range, and realized vol from recent book ticker.

    Avoids building a full microprice series on busy 10k–60k-tick windows
    (FAQ.md): only timestamps are scanned, then a few indexed microprices.
    """
    out = {
        "imb": 0.0,
        "m1": 0.0,
        "m3": 0.0,
        "m5": 0.0,
        "rv": 0.0,
        "rng": 0.0,
    }
    if not book_ticker:
        return out
    ts_raw = book_ticker.get("recv_ts_ms")
    bp_raw = book_ticker.get("bid_price")
    bq_raw = book_ticker.get("bid_qty")
    ap_raw = book_ticker.get("ask_price")
    aq_raw = book_ticker.get("ask_qty")
    if ts_raw is None or bp_raw is None or bq_raw is None or ap_raw is None or aq_raw is None:
        return out
    ts_all = ts_raw if isinstance(ts_raw, np.ndarray) else np.asarray(ts_raw, dtype=np.int64)
    if ts_all.size == 0:
        return out
    n = min(ts_all.size, len(bp_raw), len(bq_raw), len(ap_raw), len(aq_raw))
    if n == 0:
        return out
    ts_all = ts_all.reshape(-1)[-n:]
    now = int(ts_all[-1])
    i0 = int(np.searchsorted(ts_all, now - lookback_ms, side="right"))
    # Restrict columns to the lookback window only.
    bp = _col_tail(bp_raw, n, i0)
    bq = _col_tail(bq_raw, n, i0)
    ap = _col_tail(ap_raw, n, i0)
    aq = _col_tail(aq_raw, n, i0)
    ts = ts_all[i0:]
    m = ts.size
    if m == 0:
        return out

    def _micro(i: int) -> float:
        den = float(bq[i]) + float(aq[i])
        if den <= 0.0:
            return 0.5 * (float(bp[i]) + float(ap[i]))
        return (float(bp[i]) * float(aq[i]) + float(ap[i]) * float(bq[i])) / den

    last = m - 1
    den_last = float(bq[last]) + float(aq[last])
    out["imb"] = (
        (float(bq[last]) - float(aq[last])) / den_last if den_last > 0.0 else 0.0
    )
    micro_last = _micro(last)

    def _mom(ms: int) -> float:
        i = int(np.searchsorted(ts, now - ms, side="right") - 1)
        if i < 0:
            return 0.0
        a = _micro(i)
        if a <= 0.0 or micro_last <= 0.0:
            return 0.0
        return float(np.log(micro_last / a))

    out["m1"] = _mom(1000)
    out["m3"] = _mom(3000)
    out["m5"] = _mom(5000)

    i10 = int(np.searchsorted(ts, now - 10_000, side="right"))
    # At most ~512 microprices for rv/rng, evenly spaced in the 10s window.
    span = m - i10
    if span > 5:
        n_samp = min(span, 512)
        idxs = i10 + np.linspace(0, span - 1, n_samp, dtype=np.int64)
        den = bq[idxs] + aq[idxs]
        samples = (bp[idxs] * aq[idxs] + ap[idxs] * bq[idxs]) / np.maximum(den, 1e-12)
        samples = samples[samples > 0.0]
        if samples.size > 5:
            rets = np.diff(np.log(samples))
            out["rv"] = float(np.std(rets) * np.sqrt(max(rets.size, 1)))
            lo = float(np.min(samples))
            hi = float(np.max(samples))
            if lo > 0.0:
                out["rng"] = float(np.log(hi / lo))
    return out


def _as_int_scalar(v: Any) -> int | None:
    """Coerce msgpack/numpy scalars (and size-1 arrays) to int. Live payloads vary."""
    if v is None:
        return None
    try:
        arr = np.asarray(v)
        if arr.size == 0:
            return None
        return int(arr.reshape(-1)[-1])
    except (TypeError, ValueError, OverflowError):
        return None


def last_event_age_ms(venue: dict | None, now_ms: int) -> float:
    if not venue:
        return 0.0
    times = venue.get("last_event_times") or {}
    if not times:
        return 0.0
    vals = [_as_int_scalar(v) for v in times.values()]
    vals = [v for v in vals if v is not None]
    if not vals:
        return 0.0
    return float(max(0, int(now_ms) - max(vals)))


def extract(payload: dict) -> dict[str, float]:
    prompt = payload.get("prompt") or {}
    venues = payload.get("venues") or {}
    spot = venues.get("spot") or {}
    fut = venues.get("futures") or {}
    now_ms = int(prompt.get("current_time_ms") or 0)

    spot_micro, spot_mid, spot_spread = last_book_micro(spot.get("book_ticker"))
    fut_micro, fut_mid, _ = last_book_micro(fut.get("book_ticker"))

    spot_complete = bool((spot.get("candles_1s") or {}).get("complete_history", True))
    fut_complete = bool((fut.get("candles_1s") or {}).get("complete_history", False))

    spot_vol, spot_mom = venue_vol_and_momentum(spot)
    if fut_complete:
        fut_vol, fut_mom = venue_vol_and_momentum(fut)
    else:
        fut_vol, fut_mom = spot_vol, spot_mom

    close = _close_tail(spot, max(HORIZON_SECONDS + 1, 121))
    if close.size >= 2:
        rets_fast = np.diff(np.log(np.maximum(close[-91:], 1e-12)))
        vol_fast = ewma_vol(rets_fast[-90:], lam=0.85) if rets_fast.size else spot_vol
    else:
        vol_fast = spot_vol
    c1 = 0.0
    c3 = 0.0
    if close.size >= 2 and close[-2] > 0.0:
        c1 = float(np.log(close[-1] / close[-2]))
    if close.size >= 4 and close[-4] > 0.0:
        c3 = float(np.log(close[-1] / close[-4]))

    vol_1s = 0.6 * spot_vol + 0.4 * fut_vol if fut_vol > 0.0 else spot_vol
    mom = 0.5 * spot_mom + 0.5 * fut_mom

    spot_depth = spot.get("depth_latest")
    fut_depth = fut.get("depth_latest")
    obi = 0.5 * depth_imbalance(spot_depth) + 0.5 * depth_imbalance(fut_depth)
    top_obi = 0.5 * top_imbalance(spot_depth) + 0.5 * top_imbalance(fut_depth)
    spot_trades = spot.get("trades")
    fut_trades = fut.get("trades")
    flow = 0.5 * trade_imbalance(spot_trades) + 0.5 * trade_imbalance(fut_trades)
    flow_fast = 0.5 * trade_imbalance(spot_trades, last_n=50) + 0.5 * trade_imbalance(
        fut_trades, last_n=50
    )
    s_flow2, s_qty2 = trade_flow_window(spot_trades, 2000)
    f_flow2, f_qty2 = trade_flow_window(fut_trades, 2000)

    s_bt = book_ticker_stats(spot.get("book_ticker"))
    f_bt = book_ticker_stats(fut.get("book_ticker"))

    px = spot_micro if spot_micro > 0.0 else (spot_mid if spot_mid > 0.0 else fut_micro)
    if px <= 0.0:
        px = float(close[-1]) if close.size else 1.0

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
    }
