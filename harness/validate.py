"""Local harness: schema load, output contract, latency budget.

Run the CLI from the repo root:

    python test.py
"""

from __future__ import annotations

import csv
import os
import shutil
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

from harness.constants import (
    HORIZON_SECONDS,
    LATENCY_BUDGET_S,
    NUM_PERCENTILES,
)
from harness.payload import load_scoring_pair, load_payload_scoring, make_sample_payload
from harness.saved_pairs import BookReadError, iter_pairs, load_book
from harness.scoring import pinball_crps, realized_from_saved_book
import synth_ultra.model as forecast_model
from synth_ultra.model import predict_percentiles

PLOT_DIR = Path("plot")
CRPS_DIR = Path("crps")


class ValidationError(Exception):
    pass


def ms_to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat(
        sep=" ", timespec="milliseconds"
    )


def forecast_svg_path(target_ms: int) -> Path:
    """plot/<target UTC>.svg, e.g. plot/2026-08-20_06-40-34.000Z.svg."""
    dt = datetime.fromtimestamp(target_ms / 1000.0, tz=timezone.utc)
    name = dt.strftime("%Y-%m-%d_%H-%M-%S") + f".{dt.microsecond // 1000:03d}Z.svg"
    return PLOT_DIR / name


def crps_csv_path(asset: str, now: datetime | None = None) -> Path:
    """crps/{asset}_{datetime.now}.csv, filesystem-safe."""
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
    return CRPS_DIR / f"{asset}_{stamp}.csv"


def write_crps_csv(path: Path, reports: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("start", "crps", "predict_ms"))
        for one in reports:
            crps = one.get("crps")
            if crps is None:
                continue
            writer.writerow(
                (
                    one["current_time_utc"],
                    f"{float(crps):.6f}",
                    f"{float(one['max_ms']):.3f}",
                )
            )


def check_output(arr: Any) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise ValidationError(f"output must be np.ndarray, got {type(arr)!r}")
    if arr.shape != (NUM_PERCENTILES,):
        raise ValidationError(f"shape must be ({NUM_PERCENTILES},), got {arr.shape}")
    if arr.dtype != np.float64:
        raise ValidationError(f"dtype must be float64, got {arr.dtype}")
    if not np.all(np.isfinite(arr)):
        raise ValidationError("output contains non-finite values")
    if not np.all(arr > 0.0):
        raise ValidationError("output must be strictly positive")
    if np.any(np.diff(arr) < 0.0):
        raise ValidationError("output must be non-decreasing")
    return arr


def run_once(payload: dict) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    out = predict_percentiles(payload)
    elapsed = time.perf_counter() - t0
    check_output(out)
    return out, elapsed


def backtest_anchors(start_ms: int, interval_s: int, length: int) -> list[int]:
    """Half-open window: start .. start+length*interval seconds, `length` tests.

    Example: start=1787208024000, interval_s=1, length=60 → 60 anchors from
    1787208024000 through 1787208083000 (end exclusive 1787208084000).
    """
    if interval_s <= 0:
        raise ValidationError("time-interval must be > 0 seconds")
    if length <= 0:
        raise ValidationError("time-length must be > 0")
    step_ms = interval_s * 1000
    return [int(start_ms) + i * step_ms for i in range(length)]


_HISTORY_POINTS = 100
_HOUR_SECONDS = 3600
_FONT = "Segoe UI, Helvetica, sans-serif"


def recent_spot_closes(
    payload: dict,
    current_time_ms: int,
    n: int = _HISTORY_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Last ``n`` spot 1-second closes at or before ``current_time_ms``.

    Returns ``(offset_s, close)``, oldest first. ``offset_s`` is seconds
    relative to now and is 0 on the latest bar.
    """
    empty = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))
    spot = ((payload.get("venues") or {}).get("spot") or {})
    candles = spot.get("candles_1s") or {}
    ohlcv = candles.get("ohlcv")
    if ohlcv is None:
        return empty
    arr = np.asarray(ohlcv, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 4:
        return empty
    closes = arr[:, 3]
    now = int(current_time_ms)
    times = candles.get("open_time_ms")

    def _tail(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        good = np.isfinite(values) & (values > 0.0)
        kept = values[good]
        if kept.size == 0:
            return empty
        if kept.size > n:
            kept = kept[-n:]
        offset = np.arange(kept.size, dtype=np.float64) - float(kept.size - 1)
        return offset, np.ascontiguousarray(kept, dtype=np.float64)

    if times is None:
        return _tail(closes)
    open_ms = np.asarray(times, dtype=np.int64).reshape(-1)
    if open_ms.shape[0] != closes.shape[0]:
        return _tail(closes)
    good = (open_ms <= now) & np.isfinite(closes) & (closes > 0.0)
    open_ms = open_ms[good]
    closes = closes[good]
    if open_ms.size == 0:
        return empty
    if open_ms.size > n:
        open_ms = open_ms[-n:]
        closes = closes[-n:]
    plot_ms = np.minimum(open_ms.astype(np.float64) + 1000.0, float(now))
    offset = (plot_ms - float(now)) / 1000.0
    return (
        np.ascontiguousarray(offset, dtype=np.float64),
        np.ascontiguousarray(closes, dtype=np.float64),
    )


def _finite_positive(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr) & (arr > 0.0)]


def _price_text(price: float) -> str:
    return f"{price:,.2f}"


def _time_text(seconds: float) -> str:
    if abs(seconds) < 1e-9:
        return "now"
    if seconds > 0.0:
        return f"+{seconds:.0f}s"
    return f"{seconds:.0f}s"


def _hour_text(seconds: float) -> str:
    if abs(seconds) < 1e-6:
        return "now"
    minutes = seconds / 60.0
    whole = int(round(minutes))
    if abs(minutes - whole) < 0.05:
        return f"+{whole}m" if whole > 0 else f"{whole}m"
    return _time_text(seconds)


def _paired_series(
    prices: np.ndarray | None,
    offsets: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(prices if prices is not None else [], dtype=np.float64).reshape(-1)
    times = np.asarray(offsets if offsets is not None else [], dtype=np.float64).reshape(-1)
    if times.size != values.size:
        times = (
            np.empty(0, dtype=np.float64)
            if values.size == 0
            else np.arange(values.size, dtype=np.float64) - float(values.size - 1)
        )
    keep = np.isfinite(values) & (values > 0.0) & np.isfinite(times)
    return values[keep], times[keep]


def write_forecast_picture(
    path: Path,
    percentiles: np.ndarray,
    realized: float,
    current_utc: str,
    target_utc: str,
    crps: float,
    history_prices: np.ndarray | None = None,
    history_offset_s: np.ndarray | None = None,
    hour_prices: np.ndarray | None = None,
    hour_offset_s: np.ndarray | None = None,
    horizon_seconds: int = HORIZON_SECONDS,
    mu_px: float | None = None,
    anchor_px: float | None = None,
) -> None:
    """Two equal panels. Left is the past hour of spot closes. Right is the
    last 100 seconds, the 100 predictions, and the aim price.

    The hour panel uses the same pixel size and a wider time axis, so one
    second occupies less width than on the forecast panel.
    """
    history, offsets = _paired_series(history_prices, history_offset_s)
    hour, hour_times = _paired_series(hour_prices, hour_offset_s)
    predictions = _finite_positive(percentiles)
    horizon = float(horizon_seconds)
    aim = float(realized)
    mu = float(mu_px) if mu_px is not None else float("nan")
    anchor = float(anchor_px) if anchor_px is not None else float("nan")
    mu_price = anchor + mu if np.isfinite(anchor) and np.isfinite(mu) else float("nan")
    if predictions.size == 0:
        center = float("nan")
    elif predictions.size % 2:
        center = float(predictions[predictions.size // 2])
    else:
        mid = predictions.size // 2
        center = float(0.5 * (predictions[mid - 1] + predictions[mid]))

    axis, top, bottom = 112, 78, 88
    plot_w, plot_h = 872, 914
    gap = 28
    hour_x = axis
    left = hour_x + plot_w + gap + axis
    w = left + plot_w + 36
    h = top + plot_h + bottom
    x_samples = [0.0, horizon]
    if offsets.size:
        x_samples.extend((float(offsets.min()), float(offsets.max())))
    x_lo = min(x_samples) - 4.0
    x_hi = max(x_samples) + 6.0
    y_parts = [predictions]
    if history.size:
        y_parts.append(history)
    if np.isfinite(aim) and aim > 0.0:
        y_parts.append(np.array([aim], dtype=np.float64))
    if np.isfinite(mu_price) and mu_price > 0.0:
        y_parts.append(np.array([mu_price], dtype=np.float64))
    y_all = np.concatenate(y_parts) if y_parts and any(part.size for part in y_parts) else np.array([1.0])
    y_lo = float(y_all.min())
    y_hi = float(y_all.max())
    pad = (y_hi - y_lo) * 0.08 or max(abs(y_lo) * 1e-4, 1.0)
    y_lo -= pad
    y_hi += pad

    def px(seconds: float) -> float:
        return left + (seconds - x_lo) / (x_hi - x_lo) * plot_w

    def py(price: float) -> float:
        return top + (y_hi - price) / (y_hi - y_lo) * plot_h

    x_ticks = [tick for tick in (-100.0, -75.0, -50.0, -25.0, 0.0, horizon) if x_lo < tick < x_hi]
    y_ticks = [y_lo + (y_hi - y_lo) * i / 8.0 for i in range(9)]
    title = f"Last {history.size or _HISTORY_POINTS}s price, +{horizon_seconds:g}s aim, 100 predictions"
    subtitle = f"{escape(current_utc)}  →  {escape(target_utc)}"
    pred_lo = float(predictions[0]) if predictions.size else float("nan")
    pred_hi = float(predictions[-1]) if predictions.size else float("nan")
    caption = (
        f"CRPS={crps:.6f}   aim={aim:,.2f}   center={center:,.2f}   "
        f"predictions {pred_lo:,.2f} … {pred_hi:,.2f}   "
        f"history={history.size}"
    )
    mu_y = py(mu_price) if np.isfinite(mu_price) and mu_price > 0.0 else None

    grid: list[str] = []
    for price in y_ticks:
        y = py(price)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            f'stroke="#e6e6e6" stroke-width="1"/>'
        )
        if mu_y is not None and abs(y - mu_y) < 16.0:
            continue
        grid.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="{_FONT}" font-size="12" fill="#444">{_price_text(price)}</text>'
        )
    if mu_y is not None:
        grid.append(
            f'<line x1="{left - 8}" y1="{mu_y:.2f}" x2="{left}" y2="{mu_y:.2f}" '
            f'stroke="#111" stroke-width="1.6"/>'
        )
        grid.append(
            f'<text x="{left - 12}" y="{mu_y + 4:.2f}" text-anchor="end" '
            f'font-family="{_FONT}" font-size="12" font-weight="600" fill="#111">'
            f"μ {_price_text(mu_price)}</text>"
        )
    xlabels: list[str] = []
    for seconds in x_ticks:
        x = px(seconds)
        grid.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" '
            f'stroke="#e6e6e6" stroke-width="1"/>'
        )
        xlabels.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 22:.2f}" text-anchor="middle" '
            f'font-family="{_FONT}" font-size="12" fill="#444">{_time_text(seconds)}</text>'
        )

    green = ""
    if history.size:
        pts = " ".join(
            f"{px(float(t)):.2f},{py(float(price)):.2f}"
            for t, price in zip(offsets, history)
        )
        if history.size > 1:
            green += (
                f'<polyline points="{pts}" fill="none" stroke="#188038" '
                f'stroke-width="1.4" stroke-linejoin="round"/>'
            )
        green += "".join(
            f'<circle cx="{px(float(t)):.2f}" cy="{py(float(price)):.2f}" r="3.1" fill="#188038"/>'
            for t, price in zip(offsets, history)
        )
    x_horizon = px(horizon)
    blue = "".join(
        f'<circle cx="{x_horizon:.2f}" cy="{py(float(price)):.2f}" r="3.1" fill="#1a73e8"/>'
        for price in predictions
    )
    yellow = ""
    if np.isfinite(center) and center > 0.0:
        yellow = (
            f'<circle cx="{x_horizon:.2f}" cy="{py(center):.2f}" r="6.2" '
            f'fill="#f9ab00" stroke="#ffffff" stroke-width="1.6"/>'
        )
    red = ""
    if np.isfinite(aim) and aim > 0.0:
        red = (
            f'<circle cx="{x_horizon:.2f}" cy="{py(aim):.2f}" r="6.2" '
            f'fill="#d93025" stroke="#ffffff" stroke-width="1.6"/>'
        )
    hx_lo = -float(_HOUR_SECONDS) - 40.0
    hx_hi = 40.0
    if hour.size:
        hy_lo = float(hour.min())
        hy_hi = float(hour.max())
    else:
        hy_lo, hy_hi = y_lo, y_hi
    hpad = (hy_hi - hy_lo) * 0.08 or max(abs(hy_lo) * 1e-4, 1.0)
    hy_lo -= hpad
    hy_hi += hpad

    def hpx(seconds: float) -> float:
        return hour_x + (seconds - hx_lo) / (hx_hi - hx_lo) * plot_w

    def hpy(price: float) -> float:
        return top + (hy_hi - price) / (hy_hi - hy_lo) * plot_h

    hour_grid: list[str] = []
    for price in (hy_lo + (hy_hi - hy_lo) * i / 8.0 for i in range(9)):
        y = hpy(price)
        hour_grid.append(
            f'<line x1="{hour_x}" y1="{y:.2f}" x2="{hour_x + plot_w}" y2="{y:.2f}" '
            f'stroke="#e6e6e6" stroke-width="1"/>'
        )
        hour_grid.append(
            f'<text x="{hour_x - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="{_FONT}" font-size="12" fill="#444">{_price_text(price)}</text>'
        )
    hour_xlabels: list[str] = []
    for seconds in (-3600.0, -2700.0, -1800.0, -900.0, 0.0):
        x = hpx(seconds)
        hour_grid.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" '
            f'stroke="#e6e6e6" stroke-width="1"/>'
        )
        hour_xlabels.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 22:.2f}" text-anchor="middle" '
            f'font-family="{_FONT}" font-size="12" fill="#444">{_hour_text(seconds)}</text>'
        )
    def _hour_stroke(times: np.ndarray, prices: np.ndarray, color: str) -> str:
        if times.size > 1:
            pts = " ".join(
                f"{hpx(float(t)):.2f},{hpy(float(price)):.2f}"
                for t, price in zip(times, prices)
            )
            return (
                f'<polyline points="{pts}" fill="none" stroke="{color}" '
                f'stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        if times.size == 1:
            return (
                f'<circle cx="{hpx(float(times[0])):.2f}" cy="{hpy(float(prices[0])):.2f}" '
                f'r="3.1" fill="{color}"/>'
            )
        return ""

    hour_line = ""
    if hour.size:
        # Share the boundary point so the red last-100s segment meets the green path.
        cut = int(np.searchsorted(hour_times, -100.0, side="left"))
        hour_line = _hour_stroke(hour_times[: cut + 1], hour[: cut + 1], "#188038")
        hour_line += _hour_stroke(hour_times[cut:], hour[cut:], "#d93025")
    hour_caption = (
        f"past hour  n={hour.size}  "
        f"{float(hour.min()) if hour.size else float('nan'):,.2f} … "
        f"{float(hour.max()) if hour.size else float('nan'):,.2f}"
    )
    legend = (
        f'<circle cx="{left + 6}" cy="58" r="4.5" fill="#188038"/>'
        f'<text x="{left + 16}" y="62" font-family="{_FONT}" font-size="12" fill="#333">'
        f"last {history.size}s</text>"
        f'<circle cx="{left + 118}" cy="58" r="4.5" fill="#1a73e8"/>'
        f'<text x="{left + 128}" y="62" font-family="{_FONT}" font-size="12" fill="#333">'
        f"100 predictions</text>"
        f'<circle cx="{left + 258}" cy="58" r="4.5" fill="#f9ab00"/>'
        f'<text x="{left + 268}" y="62" font-family="{_FONT}" font-size="12" fill="#333">'
        f"center</text>"
        f'<circle cx="{left + 340}" cy="58" r="4.5" fill="#d93025"/>'
        f'<text x="{left + 350}" y="62" font-family="{_FONT}" font-size="12" fill="#333">'
        f"aim +{horizon_seconds:g}s</text>"
    )

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="{w}" height="{h}" fill="#fafafa"/>
  <text x="{hour_x}" y="24" font-family="{_FONT}" font-size="16" font-weight="600" fill="#111">Past hour</text>
  <text x="{left}" y="24" font-family="{_FONT}" font-size="16" font-weight="600" fill="#111">{escape(title)}</text>
  <text x="{hour_x}" y="42" font-family="{_FONT}" font-size="12" fill="#555">{subtitle}</text>
  {legend}
  <rect x="{hour_x}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff"/>
  {"".join(hour_grid)}
  <rect x="{hour_x}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#bbb"/>
  <line x1="{hpx(0.0):.2f}" y1="{top}" x2="{hpx(0.0):.2f}" y2="{top + plot_h}" stroke="#bdbdbd" stroke-width="1" stroke-dasharray="4 4"/>
  {hour_line}
  {"".join(hour_xlabels)}
  <text x="{hour_x + plot_w / 2:.2f}" y="{top + plot_h + 46:.2f}" text-anchor="middle" font-family="{_FONT}" font-size="12" fill="#333">minutes from now</text>
  <text x="{hour_x}" y="{top + plot_h + 70:.2f}" font-family="{_FONT}" font-size="12" fill="#333">{escape(hour_caption)}</text>
  <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff"/>
  {"".join(grid)}
  <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#bbb"/>
  <line x1="{px(0.0):.2f}" y1="{top}" x2="{px(0.0):.2f}" y2="{top + plot_h}" stroke="#bdbdbd" stroke-width="1" stroke-dasharray="4 4"/>
  <line x1="{x_horizon:.2f}" y1="{top}" x2="{x_horizon:.2f}" y2="{top + plot_h}" stroke="#1a73e8" stroke-width="1" stroke-dasharray="4 4" opacity="0.45"/>
  {green}
  {blue}
  {yellow}
  {red}
  {"".join(xlabels)}
  <text x="{left + plot_w / 2:.2f}" y="{top + plot_h + 46:.2f}" text-anchor="middle" font-family="{_FONT}" font-size="12" fill="#333">seconds from now</text>
  <text x="16" y="{top + plot_h / 2:.2f}" transform="rotate(-90 16 {top + plot_h / 2:.2f})" text-anchor="middle" font-family="{_FONT}" font-size="12" fill="#333">BTC price</text>
  <text x="{left - 118}" y="{top + plot_h / 2:.2f}" transform="rotate(-90 {left - 118} {top + plot_h / 2:.2f})" text-anchor="middle" font-family="{_FONT}" font-size="12" fill="#333">BTC price</text>
  <text x="{left}" y="{top + plot_h + 70:.2f}" font-family="{_FONT}" font-size="12" fill="#333">{escape(caption)}</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


_LOOKUP_REALIZED = object()


def _validate_one(
    payload: dict,
    *,
    rounds: int,
    warmup: int = 0,
    realized_price: Any = _LOOKUP_REALIZED,
) -> dict[str, Any]:
    times: list[float] = []
    first = None
    crps: float | None = None
    anchor = int(payload["prompt"]["current_time_ms"])
    horizon = int(payload["prompt"].get("horizon_seconds", HORIZON_SECONDS))
    target = anchor + horizon * 1000
    # SPECIFICATION.md: drop (no penalty) when the spot book is not live at the
    # target. FAQ.md: the container's one warm-up call is not scored or timed.
    realized = None if realized_price is _LOOKUP_REALIZED else realized_price
    for _ in range(max(0, warmup)):
        predict_percentiles(payload)
    for _ in range(rounds):
        out, elapsed = run_once(payload)
        if realized is not None:
            crps = pinball_crps(out, realized)
        if first is None:
            first = out
        elif not np.array_equal(first, out):
            raise ValidationError("model is not deterministic on the same payload")
        times.append(elapsed)

    median = statistics.median(times)
    p95 = statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times)
    hour_offset_s, hour_prices = recent_spot_closes(payload, anchor, n=_HOUR_SECONDS)
    if hour_prices.size > _HISTORY_POINTS:
        history_offset_s = hour_offset_s[-_HISTORY_POINTS:]
        history_prices = hour_prices[-_HISTORY_POINTS:]
    else:
        history_offset_s = hour_offset_s
        history_prices = hour_prices
    return {
        "ok": True,
        "median_ms": median * 1000.0,
        "p95_ms": p95 * 1000.0,
        "max_ms": max(times) * 1000.0,
        "budget_ms": LATENCY_BUDGET_S * 1000.0,
        "within_budget": median <= LATENCY_BUDGET_S,
        "head": first[:5].tolist() if first is not None else [],
        "tail": first[-5:].tolist() if first is not None else [],
        "percentiles": first.copy() if first is not None else np.array([], dtype=np.float64),
        "current_time_ms": anchor,
        "target_ms": target,
        "current_time_utc": ms_to_utc(anchor),
        "target_utc": ms_to_utc(target),
        "realized_price": realized,
        "crps": crps,
        "scored": realized is not None,
        "predict_times_s": times,
        "history_offset_s": history_offset_s,
        "history_prices": history_prices,
        "hour_offset_s": hour_offset_s,
        "hour_prices": hour_prices,
        "mu_px": float(forecast_model.last_mu_px),
        "anchor_px": float(forecast_model.last_anchor_px),
    }


def validate(
    *,
    warmup: int = 8,
    rounds: int = 64,
    seed: int = 0,
    strict: bool = False,
    current_time_ms: int | None = None,
    time_interval: int = 1,
    time_length: int = 1,
    payload: dict | None = None,
    realized_price: Any = _LOOKUP_REALIZED,
    on_test: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    t_prep = time.perf_counter()
    prepare_s = time.perf_counter() - t_prep
    if on_test is not None:
        print(f"prepare={prepare_s:.3f} s")
        sys.stdout.flush()
    if payload is not None:
        t1 = time.perf_counter()
        one = _validate_one(payload, rounds=rounds, warmup=warmup, realized_price=realized_price)
        one["elapsed_s"] = time.perf_counter() - t1
        if on_test is not None:
            on_test(1, one)
        reports = [one]
        all_times = list(one["predict_times_s"])
        asset = str(payload["prompt"].get("asset") or "BTC")
        start_ms = int(one["current_time_ms"])
        time_interval = 0
        time_length = 1
    else:
        if current_time_ms is None:
            start_ms = int(make_sample_payload(seed)["prompt"]["current_time_ms"])
        else:
            start_ms = int(current_time_ms)
        anchors = backtest_anchors(start_ms, time_interval, time_length)
        reports = []
        all_times = []
        asset = "BTC"
        for i, anchor in enumerate(anchors, start=1):
            t1 = time.perf_counter()
            point_payload = make_sample_payload(seed, current_time_ms=anchor)
            if i == 1:
                asset = str(point_payload["prompt"]["asset"])
            one = _validate_one(
                point_payload,
                rounds=rounds,
                warmup=warmup if i == 1 else 0,
                realized_price=realized_price,
            )
            one["elapsed_s"] = time.perf_counter() - t1
            reports.append(one)
            all_times.extend(one["predict_times_s"])
            if on_test is not None:
                on_test(i, one)

    median = statistics.median(all_times)
    p95 = statistics.quantiles(all_times, n=20)[18] if len(all_times) >= 20 else max(all_times)
    scored = [r for r in reports if r["crps"] is not None]
    crps_vals = [float(r["crps"]) for r in scored]
    first = reports[0]
    last = reports[-1]
    display = scored[0] if scored else first
    report: dict[str, Any] = {
        "ok": True,
        "n": len(reports),
        "n_scored": len(scored),
        "n_dropped": len(reports) - len(scored),
        "time_interval": time_interval,
        "time_length": time_length,
        "median_ms": median * 1000.0,
        "p95_ms": p95 * 1000.0,
        "max_ms": max(all_times) * 1000.0,
        "budget_ms": LATENCY_BUDGET_S * 1000.0,
        "within_budget": median <= LATENCY_BUDGET_S,
        "head": first["head"],
        "tail": first["tail"],
        "percentiles": first["percentiles"],
        "current_time_ms": first["current_time_ms"],
        "target_ms": first["target_ms"],
        "current_time_utc": first["current_time_utc"],
        "target_utc": first["target_utc"],
        "last_current_time_ms": last["current_time_ms"],
        "last_target_ms": last["target_ms"],
        "last_current_time_utc": last["current_time_utc"],
        "last_target_utc": last["target_utc"],
        "end_ms": start_ms + time_length * time_interval * 1000,
        "end_utc": ms_to_utc(start_ms + time_length * time_interval * 1000),
        "realized_price": display["realized_price"],
        "crps": float(statistics.mean(crps_vals)) if crps_vals else None,
        "crps_median": float(statistics.median(crps_vals)) if crps_vals else None,
        "crps_min": min(crps_vals) if crps_vals else None,
        "crps_max": max(crps_vals) if crps_vals else None,
        "prepare_s": prepare_s,
        "asset": asset,
        "reports": reports,
    }
    if strict and not report["within_budget"]:
        raise ValidationError(
            f"median latency {report['median_ms']:.3f} ms exceeds "
            f"{report['budget_ms']:.1f} ms budget"
        )
    return report


def _scoring_workers(n_pairs: int) -> int:
    """Parallel JSON loads for a large saved-pair run. Small runs stay in-process."""
    if n_pairs < 32:
        return 1
    cpus = os.cpu_count() or 2
    # Past four processes the JSON parse stops getting faster.
    return max(2, min(4, cpus - 1, n_pairs))


def _iter_scoring_inputs(
    pairs: list[tuple[Path, Path, Path]],
    *,
    workers: int,
) -> Iterator[tuple[int, Path, dict | None, Any]]:
    """Yield ``(index, folder, payload, book)``. ``payload`` is None on a bad book."""
    if workers <= 1:
        for i, (folder, payload_file, book_file) in enumerate(pairs, start=1):
            payload = load_payload_scoring(payload_file)
            try:
                book = load_book(book_file)
            except BookReadError as exc:
                yield i, folder, None, exc
                continue
            yield i, folder, payload, book
        return

    jobs = list(enumerate(pairs, start=1))
    # Finish a batch of loads before yielding it. The caller scores that batch
    # before pulling the next one, so JSON parsing does not run during the
    # timed predict.
    batch_size = max(workers * 4, 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(jobs), batch_size):
            futures = [
                (
                    i,
                    folder,
                    pool.submit(load_scoring_pair, str(payload_file), str(book_file)),
                )
                for i, (folder, payload_file, book_file) in jobs[start : start + batch_size]
            ]
            ready: list[tuple[int, Path, dict | None, Any]] = []
            for i, folder, fut in futures:
                try:
                    payload, book = fut.result()
                except BookReadError as exc:
                    ready.append((i, folder, None, exc))
                else:
                    ready.append((i, folder, payload, book))
            yield from ready


def _delete_pair_folder(folder: Path, *, reason: str) -> None:
    """Remove a ``database/{current_time_ms}`` capture that cannot be scored."""
    if not folder.is_dir():
        return
    shutil.rmtree(folder)
    print(f"deleted {folder.resolve()}  ({reason})", flush=True)


def validate_saved_pairs(
    database: Path,
    *,
    warmup: int = 1,
    rounds: int = 1,
    strict: bool = False,
    on_test: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Score every ``database/{current_time_ms}`` payload against its saved book tick."""
    pairs = list(iter_pairs(database))
    if not pairs:
        raise ValidationError(
            f"no payload/book pairs under {database}. "
            "Capture them with: python fetch_payload.py"
        )
    t_prep = time.perf_counter()
    prepare_s = time.perf_counter() - t_prep
    reports: list[dict[str, Any]] = []
    all_times: list[float] = []
    asset = "BTC"
    workers = _scoring_workers(len(pairs))
    for _i, folder, payload, book in _iter_scoring_inputs(pairs, workers=workers):
        if payload is None:
            _delete_pair_folder(folder, reason="unreadable spot book")
            continue
        asset = str(payload["prompt"].get("asset") or asset)
        anchor = int(payload["prompt"]["current_time_ms"])
        horizon = int(payload["prompt"].get("horizon_seconds", HORIZON_SECONDS))
        realized = realized_from_saved_book(book, anchor, horizon)
        if realized is None:
            _delete_pair_folder(
                folder, reason="spot book-ticker not live at horizon"
            )
            continue
        # Touch arrays once outside the timed window so first-page faults from
        # freshly loaded payloads are not counted as model latency.
        predict_percentiles(payload)
        t1 = time.perf_counter()
        one = _validate_one(
            payload,
            rounds=rounds,
            warmup=warmup if len(reports) == 0 else 0,
            realized_price=realized,
        )
        one["elapsed_s"] = time.perf_counter() - t1
        one["book_recv_ts_ms"] = book.get("recv_ts_ms")
        reports.append(one)
        all_times.extend(one["predict_times_s"])
        if on_test is not None:
            on_test(len(reports), one)

    if not reports:
        raise ValidationError(f"no scorable payload/book pairs under {database}")

    median = statistics.median(all_times)
    p95 = statistics.quantiles(all_times, n=20)[18] if len(all_times) >= 20 else max(all_times)
    crps_vals = [float(item["crps"]) for item in reports]
    first = reports[0]
    last = reports[-1]
    display = first
    report: dict[str, Any] = {
        "ok": True,
        "n": len(reports),
        "n_scored": len(reports),
        "n_dropped": 0,
        "n_unreadable": 0,
        "time_interval": 0,
        "time_length": len(reports),
        "median_ms": median * 1000.0,
        "p95_ms": p95 * 1000.0,
        "max_ms": max(all_times) * 1000.0,
        "budget_ms": LATENCY_BUDGET_S * 1000.0,
        "within_budget": median <= LATENCY_BUDGET_S,
        "head": first["head"],
        "tail": first["tail"],
        "percentiles": first["percentiles"],
        "current_time_ms": first["current_time_ms"],
        "target_ms": first["target_ms"],
        "current_time_utc": first["current_time_utc"],
        "target_utc": first["target_utc"],
        "last_current_time_ms": last["current_time_ms"],
        "last_target_ms": last["target_ms"],
        "last_current_time_utc": last["current_time_utc"],
        "last_target_utc": last["target_utc"],
        "end_ms": last["target_ms"],
        "end_utc": last["target_utc"],
        "realized_price": display["realized_price"],
        "crps": float(statistics.mean(crps_vals)) if crps_vals else None,
        "crps_median": float(statistics.median(crps_vals)) if crps_vals else None,
        "crps_min": min(crps_vals) if crps_vals else None,
        "crps_max": max(crps_vals) if crps_vals else None,
        "prepare_s": prepare_s,
        "asset": asset,
        "reports": reports,
    }
    if strict and not report["within_budget"]:
        raise ValidationError(
            f"median latency {report['median_ms']:.3f} ms exceeds "
            f"{report['budget_ms']:.1f} ms budget"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI lives in the repo-root ``test.py`` (`python test.py`)."""
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "test.py"
    spec = importlib.util.spec_from_file_location("synth_ultra_test_cli", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main(argv))
