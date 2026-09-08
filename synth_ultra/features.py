"""Extract cheap, vectorized features from a Synth Ultra payload."""

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


def _trade_times(trades: dict) -> np.ndarray:
    recv = trades.get("recv_ts_ms")
    if recv is not None:
        arr = np.asarray(recv)
        if arr.size:
            return arr.astype(np.int64, copy=False)
    ts = trades.get("ts_ms")
    if ts is None:
        return np.empty(0, dtype=np.int64)
    return np.asarray(ts, dtype=np.int64)


def trade_imbalance_window(trades: dict | None, now_ms: int, dt_ms: int) -> float:
    """Signed aggressor flow over the trailing dt_ms (recv clock)."""
    if not trades or dt_ms <= 0:
        return 0.0
    ts = _trade_times(trades)
    qty = _as_f64(trades.get("qty"))
    maker = _as_bool(trades.get("buyer_is_maker"))
    n = min(ts.size, qty.size, maker.size)
    if n == 0:
        return 0.0
    i0 = int(np.searchsorted(ts[:n], int(now_ms) - int(dt_ms), side="left"))
    if i0 >= n:
        return 0.0
    qty = qty[i0:n]
    maker = maker[i0:n]
    tot = float(np.sum(qty))
    if tot <= 0.0:
        return 0.0
    signed = np.where(maker, -qty, qty)
    return float(np.sum(signed) / tot)


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


def rms_vol(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(returns * returns)))


def har_vol(returns: np.ndarray) -> float:
    """Short EWMA mixed with longer RMS so post-jump width does not collapse."""
    if returns.size == 0:
        return 0.0
    v30 = ewma_vol(returns[-30:])
    v120 = rms_vol(returns[-120:])
    v300 = rms_vol(returns[-300:])
    parts = []
    weights = []
    if v30 > 0.0:
        parts.append(v30)
        weights.append(0.50)
    if v120 > 0.0:
        parts.append(v120)
        weights.append(0.30)
    if v300 > 0.0:
        parts.append(v300)
        weights.append(0.20)
    if not parts:
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    return float(np.dot(w / w.sum(), np.asarray(parts, dtype=np.float64)))


def candle_close(venue: dict | None) -> np.ndarray:
    if not venue:
        return np.empty(0, dtype=np.float64)
    candles = venue.get("candles_1s") or {}
    ohlcv = np.asarray(candles.get("ohlcv", []), dtype=np.float64)
    if ohlcv.ndim != 2 or ohlcv.shape[0] == 0 or ohlcv.shape[1] < 4:
        return np.empty(0, dtype=np.float64)
    return ohlcv[:, 3]


def last_book_obi(book_ticker: dict | None) -> float:
    """Signed qty imbalance at the last book-ticker quote."""
    if not book_ticker:
        return 0.0
    bid_q = _as_f64(book_ticker.get("bid_qty"))
    ask_q = _as_f64(book_ticker.get("ask_qty"))
    bq, aq = float(bid_q[-1]), float(ask_q[-1])
    tot = bq + aq
    if tot <= 0.0 or not np.isfinite(tot):
        return 0.0
    return (bq - aq) / tot


def book_obi_mean(book_ticker: dict | None, last_n: int = 8) -> float:
    """Mean BBO qty imbalance over the last few book-ticker rows."""
    if not book_ticker or last_n <= 0:
        return 0.0
    bid_q = _as_f64(book_ticker.get("bid_qty"))
    ask_q = _as_f64(book_ticker.get("ask_qty"))
    n = min(int(last_n), bid_q.size, ask_q.size)
    if n == 0:
        return 0.0
    bq = bid_q[-n:]
    aq = ask_q[-n:]
    tot = bq + aq
    mask = tot > 0.0
    if not np.any(mask):
        return 0.0
    return float(np.mean((bq[mask] - aq[mask]) / tot[mask]))


def log_ret_n(close: np.ndarray, n: int) -> float:
    if close.size <= n:
        return 0.0
    c0 = float(close[-(n + 1)])
    c1 = float(close[-1])
    if c0 <= 0.0 or c1 <= 0.0:
        return 0.0
    return float(np.log(c1 / c0))


def venue_vol_and_momentum(venue: dict | None) -> tuple[float, float]:
    """HAR 1s vol and last-horizon log return. Falls back to zeros."""
    close = candle_close(venue)
    rets = log_returns(close)
    vol_1s = har_vol(rets)
    mom = log_ret_n(close, HORIZON_SECONDS)
    return vol_1s, mom


def last_event_age_ms(venue: dict | None, now_ms: int) -> float:
    if not venue:
        return 0.0
    times = venue.get("last_event_times") or {}
    if not times:
        return 0.0
    try:
        latest = max(int(v) for v in times.values() if v is not None)
    except ValueError:
        return 0.0
    return float(max(0, now_ms - latest))


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

    vol_1s = 0.6 * spot_vol + 0.4 * fut_vol if fut_vol > 0.0 else spot_vol
    mom = 0.5 * spot_mom + 0.5 * fut_mom
    spot_close = candle_close(spot)
    ret_1s = log_ret_n(spot_close, 1)
    ret_10s = log_ret_n(spot_close, HORIZON_SECONDS)
    fut_ret_1s = log_ret_n(candle_close(fut), 1) if fut_complete else ret_1s
    lead_1s = fut_ret_1s - ret_1s

    obi = 0.5 * depth_imbalance(spot.get("depth_latest")) + 0.5 * depth_imbalance(
        fut.get("depth_latest")
    )
    top_obi = 0.5 * top_imbalance(spot.get("depth_latest")) + 0.5 * top_imbalance(
        fut.get("depth_latest")
    )
    top1_obi = 0.5 * top_imbalance(spot.get("depth_latest"), n=1) + 0.5 * top_imbalance(
        fut.get("depth_latest"), n=1
    )
    book_obi = 0.6 * last_book_obi(spot.get("book_ticker")) + 0.4 * last_book_obi(
        fut.get("book_ticker")
    )
    book_obi_avg = 0.6 * book_obi_mean(spot.get("book_ticker")) + 0.4 * book_obi_mean(
        fut.get("book_ticker")
    )
    flow = 0.5 * trade_imbalance(spot.get("trades")) + 0.5 * trade_imbalance(fut.get("trades"))
    flow_fast = 0.5 * trade_imbalance(spot.get("trades"), last_n=50) + 0.5 * trade_imbalance(
        fut.get("trades"), last_n=50
    )
    flow20 = 0.5 * trade_imbalance(spot.get("trades"), last_n=20) + 0.5 * trade_imbalance(
        fut.get("trades"), last_n=20
    )
    flow1s = 0.5 * trade_imbalance_window(spot.get("trades"), now_ms, 1000) + 0.5 * trade_imbalance_window(
        fut.get("trades"), now_ms, 1000
    )

    px = spot_micro if spot_micro > 0.0 else (spot_mid if spot_mid > 0.0 else fut_micro)
    if px <= 0.0:
        close = candle_close(spot)
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
        "momentum": float(mom),
        "obi": float(obi),
        "top_obi": float(top_obi),
        "flow": float(flow),
        "flow_fast": float(flow_fast),
        "basis": float(basis),
        "spread_rel": float(spread_rel),
        "stale_ms": float(stale_ms),
        "spot_complete": float(spot_complete),
        "fut_complete": float(fut_complete),
        "ret_1s": float(ret_1s),
        "ret_10s": float(ret_10s),
        "lead_1s": float(lead_1s),
        "top1_obi": float(top1_obi),
        "book_obi": float(book_obi),
        "book_obi_avg": float(book_obi_avg),
        "flow20": float(flow20),
        "flow1s": float(flow1s),
    }
