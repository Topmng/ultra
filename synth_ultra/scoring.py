"""Scoring helper: pinball-loss CRPS from SPECIFICATION.md."""

from __future__ import annotations

import time

import numpy as np

from synth_ultra.constants import (
    HORIZON_SECONDS,
    NUM_PERCENTILES,
    QUANTILE_GRID,
    SPOT_FEED_LIVE_MS,
    SPOT_FEED_REST_LIVE_MS,
)


def spot_microprice(bid_price: float, bid_qty: float, ask_price: float, ask_qty: float) -> float:
    """Scoring target: last spot book-ticker microprice at the horizon instant."""
    den = bid_qty + ask_qty
    if den <= 0.0:
        return 0.5 * (bid_price + ask_price)
    return (bid_price * ask_qty + ask_price * bid_qty) / den


def _finite_positive_book(bp: float, bq: float, ap: float, aq: float) -> bool:
    vals = (bp, bq, ap, aq)
    return bool(np.all(np.isfinite(vals))) and bp > 0.0 and ap > 0.0 and bq >= 0.0 and aq >= 0.0


def _spot_microprice_from_csv(target_ms: int) -> float | None:
    from synth_ultra.payload import SPOT_BOOK_TICKER_CSV, _load_book_ticker_csv

    table = _load_book_ticker_csv(str(SPOT_BOOK_TICKER_CSV))
    if table is None or table["recv_ts_ms"].size == 0:
        return None
    times = table["recv_ts_ms"]
    i = int(np.searchsorted(times, int(target_ms), side="right") - 1)
    if i < 0:
        return None
    recv = int(times[i])
    if int(target_ms) - recv > SPOT_FEED_LIVE_MS:
        return None
    bp = float(table["bid_price"][i])
    bq = float(table["bid_qty"][i])
    ap = float(table["ask_price"][i])
    aq = float(table["ask_qty"][i])
    if not _finite_positive_book(bp, bq, ap, aq):
        return None
    return spot_microprice(bp, bq, ap, aq)


def _spot_microprice_from_rest(target_ms: int) -> float | None:
    """Live spot bookTicker only — Binance REST has no historical book-ticker series."""
    import json
    import urllib.request

    now = int(time.time() * 1000)
    if abs(now - int(target_ms)) > SPOT_FEED_REST_LIVE_MS:
        return None
    url = "https://api.binance.com/api/v3/ticker/bookTicker?symbol=BTCUSDT"
    request = urllib.request.Request(url, headers={"User-Agent": "synth-ultra-validate/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            row = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None
    try:
        bp = float(row["bidPrice"])
        bq = float(row["bidQty"])
        ap = float(row["askPrice"])
        aq = float(row["askQty"])
    except (KeyError, TypeError, ValueError):
        return None
    if not _finite_positive_book(bp, bq, ap, aq):
        return None
    return spot_microprice(bp, bq, ap, aq)


def realized_spot_microprice(
    current_time_ms: int,
    horizon_seconds: int = HORIZON_SECONDS,
    *,
    allow_rest: bool = False,
) -> float | None:
    """Spot book-ticker microprice at current_time_ms + horizon.

    SPECIFICATION.md: last spot book-ticker at or before the target instant.
    Returns None (prediction dropped, no penalty) if that feed is not live.
    """
    target = int(current_time_ms) + int(horizon_seconds) * 1000
    y = _spot_microprice_from_csv(target)
    if y is None and allow_rest:
        y = _spot_microprice_from_rest(target)
    return y


def pinball_crps(percentiles: np.ndarray, realized_price: float) -> float:
    """CRPS = (2/N) · Σ_i ρ_τ_i(y − x_i), τ_i = (2i−1)/200.

    Lower is better, in price units.
    """
    x = np.asarray(percentiles, dtype=np.float64).reshape(NUM_PERCENTILES)
    y = float(realized_price)
    u = y - x
    tau = QUANTILE_GRID
    rho = u * (tau - (u < 0.0).astype(np.float64))
    return float((2.0 / NUM_PERCENTILES) * np.sum(rho))
