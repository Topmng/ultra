"""Local harness: schema load, output contract, latency budget.

Run the CLI from the repo root:

    python test.py
"""

from __future__ import annotations

import csv
import statistics
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

from synth_ultra.constants import (
    HORIZON_SECONDS,
    LATENCY_BUDGET_S,
    NUM_PERCENTILES,
    QUANTILE_GRID,
)
from synth_ultra.model import predict_percentiles
from synth_ultra.payload import make_sample_payload, preload_venue_csvs
from synth_ultra.scoring import pinball_crps, realized_spot_close

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
        writer.writerow(("test", "start", "target", "crps"))
        for i, one in enumerate(reports, start=1):
            writer.writerow(
                (i, one["current_time_utc"], one["target_utc"], f"{one['crps']:.6f}")
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


def write_forecast_picture(
    path: Path,
    percentiles: np.ndarray,
    realized: float,
    current_utc: str,
    target_utc: str,
    crps: float,
) -> None:
    """Draw the 100 percentile points: BTC price on X, quantile on Y."""
    prices = np.asarray(percentiles, dtype=np.float64)
    quantiles = np.asarray(QUANTILE_GRID, dtype=np.float64)
    w, h = 920, 560
    left, right, top, bottom = 56, 28, 56, 72
    plot_w = w - left - right
    plot_h = h - top - bottom
    x_lo = min(float(prices.min()), realized)
    x_hi = max(float(prices.max()), realized)
    pad = (x_hi - x_lo) * 0.08 or 1.0
    x_lo -= pad
    x_hi += pad
    q_lo, q_hi = 0.0, 1.0

    def px(price: float) -> float:
        return left + (price - x_lo) / (x_hi - x_lo) * plot_w

    def py(q: float) -> float:
        return top + (q_hi - q) / (q_hi - q_lo) * plot_h

    x_real = px(realized)
    ticks_y = [0.005, 0.25, 0.5, 0.75, 0.995]
    n_xticks = 6
    x_ticks = [x_lo + (x_hi - x_lo) * i / (n_xticks - 1) for i in range(n_xticks)]
    title = "Output contract  (100,) float64  finite  positive  non-decreasing"
    subtitle = f"{escape(current_utc)}  →  {escape(target_utc)}"
    caption = (
        f"CRPS={crps:.6f}   realized_close={realized:.4f}   "
        f"q=0.005 {prices[0]:.2f}   q=0.995 {prices[-1]:.2f}"
    )

    grid = []
    for q in ticks_y:
        y = py(q)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            f'stroke="#e6e6e6" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Segoe UI, Helvetica, sans-serif" font-size="12" '
            f'fill="#444">{q:.3f}</text>'
        )
    xlabels = []
    for price in x_ticks:
        x = px(price)
        grid.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" '
            f'stroke="#e6e6e6" stroke-width="1"/>'
        )
        xlabels.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 22:.2f}" text-anchor="middle" '
            f'font-family="Segoe UI, Helvetica, sans-serif" font-size="12" '
            f'fill="#444">{price:.2f}</text>'
        )
    dots = [
        f'<circle cx="{px(float(price)):.2f}" cy="{py(float(q)):.2f}" r="3.2" '
        f'fill="#1f4e79"/>'
        for q, price in zip(quantiles, prices)
    ]
    label_x = px(float(prices[49])) + 10
    label_y = py(0.5)

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="{w}" height="{h}" fill="#fafafa"/>
  <text x="{left}" y="26" font-family="Segoe UI, Helvetica, sans-serif" font-size="16" font-weight="600" fill="#111">{escape(title)}</text>
  <text x="{left}" y="46" font-family="Segoe UI, Helvetica, sans-serif" font-size="12" fill="#555">{subtitle}</text>
  {"".join(grid)}
  <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#bbb"/>
  <line x1="{x_real:.2f}" y1="{top}" x2="{x_real:.2f}" y2="{top + plot_h}" stroke="#c0392b" stroke-width="1.5" stroke-dasharray="6 4"/>
  {"".join(dots)}
  {"".join(xlabels)}
  <text x="{left + plot_w / 2:.2f}" y="{h - 18}" text-anchor="middle" font-family="Segoe UI, Helvetica, sans-serif" font-size="12" fill="#333">BTC price</text>
  <text x="16" y="{top + plot_h / 2:.2f}" transform="rotate(-90 16 {top + plot_h / 2:.2f})" text-anchor="middle" font-family="Segoe UI, Helvetica, sans-serif" font-size="12" fill="#333">quantile q_i = (2i-1)/200</text>
  <text x="{min(x_real + 8, left + plot_w - 160):.2f}" y="{top + 16}" font-family="Segoe UI, Helvetica, sans-serif" font-size="12" fill="#c0392b">realized close {realized:.2f}</text>
  <text x="{label_x:.2f}" y="{label_y:.2f}" font-family="Segoe UI, Helvetica, sans-serif" font-size="12" fill="#1f4e79">100 predicted percentiles</text>
  <text x="{left}" y="{h - 8}" font-family="Segoe UI, Helvetica, sans-serif" font-size="12" fill="#333">{escape(caption)}</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def _validate_one(
    payload: dict,
    *,
    rounds: int,
    allow_rest: bool = False,
) -> dict[str, Any]:
    times: list[float] = []
    first = None
    crps: float | None = None
    realized: float | None = None
    anchor = int(payload["prompt"]["current_time_ms"])
    target = anchor + HORIZON_SECONDS * 1000
    for _ in range(rounds):
        out, elapsed = run_once(payload)
        realized = realized_spot_close(anchor, allow_rest=allow_rest)
        if realized is None:
            raise ValidationError(
                f"no spot candle close at current_time+{HORIZON_SECONDS}s "
                f"({ms_to_utc(target)}); need database/btc_spot_candles.csv or a live REST kline"
            )
        crps = pinball_crps(out, realized)
        if first is None:
            first = out
        elif not np.array_equal(first, out):
            raise ValidationError("model is not deterministic on the same payload")
        times.append(elapsed)

    median = statistics.median(times)
    p95 = statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times)
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
        "realized_close": realized,
        "crps": crps,
        "predict_times_s": times,
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
    allow_rest: bool = False,
    on_test: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    t_prep = time.perf_counter()
    if payload is None:
        preload_venue_csvs()
    prepare_s = time.perf_counter() - t_prep
    if on_test is not None:
        print(f"prepare={prepare_s:.3f} s")
        sys.stdout.flush()
    if payload is not None:
        t1 = time.perf_counter()
        one = _validate_one(payload, rounds=rounds, allow_rest=allow_rest)
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
            one = _validate_one(point_payload, rounds=rounds, allow_rest=allow_rest)
            one["elapsed_s"] = time.perf_counter() - t1
            reports.append(one)
            all_times.extend(one["predict_times_s"])
            if on_test is not None:
                on_test(i, one)

    median = statistics.median(all_times)
    p95 = statistics.quantiles(all_times, n=20)[18] if len(all_times) >= 20 else max(all_times)
    crps_vals = [float(r["crps"]) for r in reports]
    first = reports[0]
    last = reports[-1]
    report: dict[str, Any] = {
        "ok": True,
        "n": len(reports),
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
        "realized_close": first["realized_close"],
        "crps": float(statistics.mean(crps_vals)),
        "crps_median": float(statistics.median(crps_vals)),
        "crps_min": min(crps_vals),
        "crps_max": max(crps_vals),
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
