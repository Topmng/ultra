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


def last_book_micro(book_ticker: dict | None) -> tuple[float, float, float]:
    """Return (microprice, mid, spread) from the latest book-ticker row."""
    if not book_ticker:
        return 0.0, 0.0, 0.0
    bid_p = _as_f64(book_ticker.get("bid_price"))
    bid_q = _as_f64(book_ticker.get("bid_qty"))
    ask_p = _as_f64(book_ticker.get("ask_price"))
    ask_q = _as_f64(book_ticker.get("ask_qty"))
    bp, bq, ap, aq = bid_p[-1], bid_q[-1], ask_p[-1], ask_q[-1]
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
    qty = _as_f64(trades.get("qty"))
    maker = _as_bool(trades.get("buyer_is_maker"))
    if qty.size == 0 or maker.size == 0:
        return 0.0
    n = min(qty.size, maker.size)
    if last_n is not None:
        n = min(n, last_n)
        qty = qty[-n:]
        maker = maker[-n:]
    else:
        qty = qty[-n:]
        maker = maker[-n:]
    signed = np.where(maker, -qty, qty)
    tot = float(np.sum(qty))
    if tot <= 0.0:
        return 0.0
    return float(np.sum(signed) / tot)


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


def ewma_vol(returns: np.ndarray, lam: float = 0.94) -> float:
    if returns.size == 0:
        return 0.0
    r2 = returns * returns
    n = r2.size
    w = (1.0 - lam) * lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
    w /= w.sum()
    return float(np.sqrt(np.dot(w, r2)))


def candle_close(venue: dict | None) -> np.ndarray:
    if not venue:
        return np.empty(0, dtype=np.float64)
    candles = venue.get("candles_1s") or {}
    ohlcv = np.asarray(candles.get("ohlcv", []), dtype=np.float64)
    if ohlcv.ndim != 2 or ohlcv.shape[0] == 0 or ohlcv.shape[1] < 4:
        return np.empty(0, dtype=np.float64)
    return ohlcv[:, 3]


def venue_vol_and_momentum(venue: dict | None) -> tuple[float, float]:
    """1s EWMA vol and last-horizon log return. Falls back to zeros."""
    close = candle_close(venue)
    rets = log_returns(close)
    vol_1s = ewma_vol(rets[-120:] if rets.size else rets)
    if close.size > HORIZON_SECONDS:
        c0 = close[-(HORIZON_SECONDS + 1)]
        c1 = close[-1]
        mom = float(np.log(c1 / c0)) if c0 > 0.0 and c1 > 0.0 else 0.0
    else:
        mom = 0.0
    return vol_1s, mom


def book_ticker_stats(book_ticker: dict | None, lookback_ms: int = 12_000) -> dict[str, float]:
    """Microprice momentum, range, and realized vol from recent book ticker."""
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
    bp = _as_f64(book_ticker.get("bid_price"))
    bq = _as_f64(book_ticker.get("bid_qty"))
    ap = _as_f64(book_ticker.get("ask_price"))
    aq = _as_f64(book_ticker.get("ask_qty"))
    ts = np.asarray(book_ticker.get("recv_ts_ms", []), dtype=np.int64)
    n = min(bp.size, bq.size, ap.size, aq.size, ts.size)
    if n == 0:
        return out
    bp, bq, ap, aq, ts = bp[-n:], bq[-n:], ap[-n:], aq[-n:], ts[-n:]
    now = int(ts[-1])
    i0 = int(np.searchsorted(ts, now - lookback_ms, side="right"))
    bp, bq, ap, aq, ts = bp[i0:], bq[i0:], ap[i0:], aq[i0:], ts[i0:]
    if bp.size == 0:
        return out
    micro = (bp * aq + ap * bq) / np.maximum(bq + aq, 1e-12)
    imb = (bq - aq) / np.maximum(bq + aq, 1e-12)
    out["imb"] = float(imb[-1])

    def _mom(ms: int) -> float:
        i = int(np.searchsorted(ts, now - ms, side="right") - 1)
        if i < 0:
            return 0.0
        a, b = float(micro[i]), float(micro[-1])
        if a <= 0.0 or b <= 0.0:
            return 0.0
        return float(np.log(b / a))

    out["m1"] = _mom(1000)
    out["m3"] = _mom(3000)
    out["m5"] = _mom(5000)

    i10 = int(np.searchsorted(ts, now - 10_000, side="right"))
    window = micro[i10:]
    if window.size > 5:
        rets = np.diff(np.log(np.maximum(window, 1e-12)))
        out["rv"] = float(np.std(rets) * np.sqrt(max(rets.size, 1)))
        lo = float(np.min(window))
        hi = float(np.max(window))
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

    close = candle_close(spot)
    rets = log_returns(close)
    vol_fast = ewma_vol(rets[-90:], lam=0.85) if rets.size else spot_vol
    c1 = 0.0
    c3 = 0.0
    if close.size >= 2 and close[-2] > 0.0:
        c1 = float(np.log(close[-1] / close[-2]))
    if close.size >= 4 and close[-4] > 0.0:
        c3 = float(np.log(close[-1] / close[-4]))

    vol_1s = 0.6 * spot_vol + 0.4 * fut_vol if fut_vol > 0.0 else spot_vol
    mom = 0.5 * spot_mom + 0.5 * fut_mom

    obi = 0.5 * depth_imbalance(spot.get("depth_latest")) + 0.5 * depth_imbalance(
        fut.get("depth_latest")
    )
    top_obi = 0.5 * top_imbalance(spot.get("depth_latest")) + 0.5 * top_imbalance(
        fut.get("depth_latest")
    )
    flow = 0.5 * trade_imbalance(spot.get("trades")) + 0.5 * trade_imbalance(fut.get("trades"))
    flow_fast = 0.5 * trade_imbalance(spot.get("trades"), last_n=50) + 0.5 * trade_imbalance(
        fut.get("trades"), last_n=50
    )
    s_flow2, s_qty2 = trade_flow_window(spot.get("trades"), 2000)
    f_flow2, f_qty2 = trade_flow_window(fut.get("trades"), 2000)

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
