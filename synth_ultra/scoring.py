"""Scoring helper: pinball-loss CRPS from SPECIFICATION.md and FAQ.md.

Per prediction, SPECIFICATION.md grades the 100 percentiles by pinball CRPS
against the spot microprice at ``current_time_ms + 10_000``. That prompt is
dropped (no penalty) when the spot book-ticker feed is not live at the target.

FAQ.md then ranks models on a prompt: a non-answer is charged the 95th
percentile of the CRPS values from models that did answer, and the best score
on that prompt is subtracted so the winner is 0. ``prompt_field_scores``
implements that adjustment. A single local backtest has no field, so it
reports the raw pinball CRPS only.
"""

from __future__ import annotations

import numpy as np

from synth_ultra.constants import (
    HORIZON_SECONDS,
    NUM_PERCENTILES,
    QUANTILE_GRID,
    SPOT_FEED_LIVE_MS,
)


def spot_microprice(bid_price: float, bid_qty: float, ask_price: float, ask_qty: float) -> float:
    """SPECIFICATION.md microprice. Undefined when both sizes are zero."""
    den = float(bid_qty) + float(ask_qty)
    if den <= 0.0 or not np.isfinite(den):
        raise ValueError("microprice denominator must be positive")
    return (float(bid_price) * float(ask_qty) + float(ask_price) * float(bid_qty)) / den


def _finite_positive_book(bp: float, bq: float, ap: float, aq: float) -> bool:
    vals = (bp, bq, ap, aq)
    return bool(np.all(np.isfinite(vals))) and bp > 0.0 and ap > 0.0 and bq >= 0.0 and aq >= 0.0 and (bq + aq) > 0.0


def spot_microprice_at(times_ms: np.ndarray, bid_price, bid_qty, ask_price, ask_qty, target_ms: int) -> float | None:
    """Last spot book-ticker microprice at or before ``target_ms``.

    Returns None when no tick exists, the tick is stale, or the quote cannot
    form the spec microprice. ``times_ms`` must be the spot ``recv_ts_ms``
    column, oldest-first.
    """
    times = np.asarray(times_ms, dtype=np.int64)
    if times.size == 0:
        return None
    target = int(target_ms)
    i = int(np.searchsorted(times, target, side="right") - 1)
    if i < 0:
        return None
    recv = int(times[i])
    if target - recv > SPOT_FEED_LIVE_MS:
        return None
    bp = float(np.asarray(bid_price, dtype=np.float64)[i])
    bq = float(np.asarray(bid_qty, dtype=np.float64)[i])
    ap = float(np.asarray(ask_price, dtype=np.float64)[i])
    aq = float(np.asarray(ask_qty, dtype=np.float64)[i])
    if not _finite_positive_book(bp, bq, ap, aq):
        return None
    return spot_microprice(bp, bq, ap, aq)


def realized_from_saved_book(
    row: dict,
    current_time_ms: int,
    horizon_seconds: int = HORIZON_SECONDS,
) -> float | None:
    """Microprice of a spot book tick saved for one payload's horizon.

    The tick must be the last spot book update in
    ``(current_time_ms, current_time_ms + horizon]`` and still live at the
    target (within ``SPOT_FEED_LIVE_MS``). Missing or stale quotes are dropped.
    """
    recv = row.get("recv_ts_ms") if row else None
    if recv is None or recv == "":
        return None
    target = int(current_time_ms) + int(horizon_seconds) * 1000
    recv_i = int(recv)
    if recv_i > target or target - recv_i > SPOT_FEED_LIVE_MS:
        return None
    bp = float(row["bid_price"])
    bq = float(row["bid_qty"])
    ap = float(row["ask_price"])
    aq = float(row["ask_qty"])
    if not _finite_positive_book(bp, bq, ap, aq):
        return None
    return spot_microprice(bp, bq, ap, aq)


def pinball_crps(percentiles: np.ndarray, realized_price: float) -> float:
    """CRPS = (2/N) · Σ_i ρ_τ_i(y − x_i), τ_i = (2i−1)/200.

    ρ_τ(u) = u · (τ − 1{u < 0}). Lower is better, in price units.
    """
    x = np.asarray(percentiles, dtype=np.float64).reshape(NUM_PERCENTILES)
    y = float(realized_price)
    if not np.isfinite(y):
        raise ValueError("realized price must be finite")
    u = y - x
    tau = QUANTILE_GRID
    rho = u * (tau - (u < 0.0).astype(np.float64))
    return float((2.0 / NUM_PERCENTILES) * np.sum(rho))


def prompt_field_scores(crps: np.ndarray, answered: np.ndarray) -> np.ndarray | None:
    """FAQ.md adjustment for one prompt that was actually scored.

    ``crps`` holds raw pinball CRPS per model. ``answered`` is False for a
    non-answer (timeout or output that failed validation). Non-answers are
    charged the 95th percentile of the models that answered, then the best
    score on the prompt is subtracted so the winner is 0.

    Returns None when nobody answered. A prompt dropped because the spot feed
    was not live is not passed here — it has no penalty and is left out of
    the average.
    """
    raw = np.asarray(crps, dtype=np.float64)
    ok = np.asarray(answered, dtype=bool)
    if raw.shape != ok.shape:
        raise ValueError("crps and answered must have the same shape")
    if not np.any(ok):
        return None
    if not np.all(np.isfinite(raw[ok])):
        raise ValueError("answered CRPS values must be finite")
    filled = raw.copy()
    penalty = float(np.quantile(raw[ok], 0.95))
    filled[~ok] = penalty
    return filled - float(np.min(filled))
