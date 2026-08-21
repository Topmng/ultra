"""Scoring helper: pinball-loss CRPS from SPECIFICATION.md."""

from __future__ import annotations

import numpy as np

from synth_ultra.constants import HORIZON_SECONDS, NUM_PERCENTILES, QUANTILE_GRID


def _spot_close_from_csv(target_ms: int) -> float | None:
    from synth_ultra.payload import SPOT_CANDLES_CSV, _load_candles_csv

    table = _load_candles_csv(str(SPOT_CANDLES_CSV))
    if table is None or table["open_time_ms"].size == 0:
        return None
    times = table["open_time_ms"]
    i = int(np.searchsorted(times, target_ms, side="left"))
    if i >= times.size or int(times[i]) != target_ms:
        return None
    return float(table["ohlcv"][i, 3])


def _spot_close_from_rest(target_ms: int) -> float | None:
    """One Binance spot 1s kline close when the local CSV does not cover the target."""
    import json
    import urllib.parse
    import urllib.request

    query = urllib.parse.urlencode(
        {
            "symbol": "BTCUSDT",
            "interval": "1s",
            "startTime": int(target_ms),
            "endTime": int(target_ms) + 999,
            "limit": 1,
        }
    )
    url = f"https://api.binance.com/api/v3/klines?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "synth-ultra-validate/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            rows = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, json.JSONDecodeError):
        return None
    if not rows:
        return None
    open_ms = int(rows[0][0])
    if open_ms >= 10**15:
        open_ms //= 1000
    if open_ms != int(target_ms):
        return None
    return float(rows[0][4])


def realized_spot_close(
    current_time_ms: int,
    horizon_seconds: int = HORIZON_SECONDS,
    *,
    allow_rest: bool = False,
) -> float | None:
    """Spot 1s close at current_time_ms + horizon.

    Prefers database/btc_spot_candles.csv. With allow_rest=True, falls back to a single
    Binance REST kline (needed for env_payload.json snapshots).
    """
    target = (int(current_time_ms) + int(horizon_seconds) * 1000) // 1000 * 1000
    y = _spot_close_from_csv(target)
    if y is None and allow_rest:
        y = _spot_close_from_rest(target)
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


def spot_microprice(bid_price: float, bid_qty: float, ask_price: float, ask_qty: float) -> float:
    """Scoring target: last spot book-ticker microprice at the horizon instant."""
    den = bid_qty + ask_qty
    if den <= 0.0:
        return 0.5 * (bid_price + ask_price)
    return (bid_price * ask_qty + ask_price * bid_qty) / den
