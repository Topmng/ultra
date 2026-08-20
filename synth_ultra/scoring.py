"""Scoring helper: pinball-loss CRPS from SPECIFICATION.md."""

from __future__ import annotations

import numpy as np

from synth_ultra.constants import HORIZON_SECONDS, NUM_PERCENTILES, QUANTILE_GRID


def realized_spot_close(current_time_ms: int, horizon_seconds: int = HORIZON_SECONDS) -> float | None:
    """Spot 1s close at current_time_ms + horizon, from btc_spot_candles.csv."""
    from synth_ultra.payload import SPOT_CANDLES_CSV, _load_candles_csv

    table = _load_candles_csv(str(SPOT_CANDLES_CSV))
    if table is None or table["open_time_ms"].size == 0:
        return None
    target = (int(current_time_ms) + int(horizon_seconds) * 1000) // 1000 * 1000
    times = table["open_time_ms"]
    i = int(np.searchsorted(times, target, side="left"))
    if i >= times.size or int(times[i]) != target:
        return None
    return float(table["ohlcv"][i, 3])


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
