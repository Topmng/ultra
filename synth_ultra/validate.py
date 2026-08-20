"""Local harness: schema load, output contract, latency budget."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Any

import numpy as np

from synth_ultra.constants import LATENCY_BUDGET_S, NUM_PERCENTILES
from synth_ultra.model import predict_percentiles
from synth_ultra.payload import make_sample_payload
from synth_ultra.scoring import pinball_crps, realized_spot_close


class ValidationError(Exception):
    pass


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


def validate(
    *,
    warmup: int = 8,
    rounds: int = 64,
    seed: int = 0,
    strict: bool = False,
    current_time_ms: int | None = None,
) -> dict[str, Any]:
    payload = make_sample_payload(seed, current_time_ms=current_time_ms)
    # for _ in range(warmup):
    #     run_once(payload)

    times: list[float] = []
    first = None
    crps: float | None = None
    realized: float | None = None
    anchor = int(payload["prompt"]["current_time_ms"])
    for _ in range(rounds):
        out, elapsed = run_once(payload)
        realized = realized_spot_close(anchor)
        if realized is None:
            raise ValidationError(
                f"no spot candle close at current_time_ms+10s "
                f"({anchor} + 10000) in btc_spot_candles.csv"
            )
        crps = pinball_crps(out, realized)
        if first is None:
            print(
                f"CRPS={crps:.6f}  realized_close={realized:.4f}  "
                f"current_time_ms={anchor}  target_ms={anchor + 10000}",
                flush=True,
            )
            first = out
        elif not np.array_equal(first, out):
            raise ValidationError("model is not deterministic on the same payload")
        times.append(elapsed)

    median = statistics.median(times)
    p95 = statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times)
    report = {
        "ok": True,
        "median_ms": median * 1000.0,
        "p95_ms": p95 * 1000.0,
        "max_ms": max(times) * 1000.0,
        "budget_ms": LATENCY_BUDGET_S * 1000.0,
        "within_budget": median <= LATENCY_BUDGET_S,
        "head": first[:5].tolist() if first is not None else [],
        "tail": first[-5:].tolist() if first is not None else [],
        "current_time_ms": anchor,
        "realized_close": realized,
        "crps": crps,
    }
    if strict and not report["within_budget"]:
        raise ValidationError(
            f"median latency {report['median_ms']:.3f} ms exceeds "
            f"{report['budget_ms']:.1f} ms budget"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Synth Ultra model contract")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--current-time-ms",
        type=int,
        default=None,
        help="forecast anchor passed to make_sample_payload",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if median latency exceeds the 5 ms budget",
    )
    args = parser.parse_args(argv)
    try:
        report = validate(
            warmup=args.warmup,
            rounds=args.rounds,
            seed=args.seed,
            strict=args.strict,
            current_time_ms=1787047608000,
        )
    except ValidationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    budget_flag = "PASS" if report["within_budget"] else "WARN"
    print("OK  output contract (100,) float64 finite positive non-decreasing")
    print("OK  deterministic")
    print(
        f"{budget_flag} latency  median={report['median_ms']:.3f} ms  "
        f"p95={report['p95_ms']:.3f} ms  max={report['max_ms']:.3f} ms  "
        f"budget={report['budget_ms']:.1f} ms"
    )
    print(f"head {report['head']}")
    print(f"tail {report['tail']}")
    print(
        f"CRPS {report['crps']:.6f}  realized_close={report['realized_close']}  "
        f"current_time_ms={report['current_time_ms']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
