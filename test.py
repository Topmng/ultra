"""Local model backtest: latency, output contract, and CRPS.

    python test.py
    python test.py --time-length 20 --time-interval 11
    python test.py --payload examples/env_payload.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from synth_ultra.payload import ENV_PAYLOAD_JSON, load_payload
from synth_ultra.validate import (
    ValidationError,
    crps_csv_path,
    forecast_svg_path,
    validate,
    write_crps_csv,
    write_forecast_picture,
)

# Flip to False to skip writing plot/*.svg during a backtest.
WRITE_SVG = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Synth Ultra model contract")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--current-time-ms",
        type=int,
        default=1787743363000,
        help="first forecast anchor (ms); backtest starts here",
    )
    parser.add_argument(
        "--time-interval",
        type=int,
        default=120,
        help="seconds between consecutive anchors",
    )
    parser.add_argument(
        "--time-length",
        type=int,
        default=700,
        help="number of backtest anchors (e.g. 60 tests of 1s = 60s window)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if median latency exceeds the 5 ms budget",
    )
    parser.add_argument(
        "--payload",
        type=Path,
        nargs="?",
        const=ENV_PAYLOAD_JSON,
        default=None,
        help="score a saved payload JSON (bare flag: examples/env_payload.json)",
    )
    args = parser.parse_args(argv)

    def on_test(i: int, one: dict[str, Any]) -> None:
        print(f"=== test {i} ===")
        if one["crps"] is None:
            print(
                f"start={one['current_time_utc']}  "
                f"target={one['target_utc']}  "
                f"CRPS=dropped (spot book-ticker not live at horizon)"
            )
            print(
                f"OK  output contract  "
                f"predict={one['max_ms']:.3f} ms  "
                f"elapsed={one['elapsed_s']:.3f} s"
            )
            sys.stdout.flush()
            return
        print(
            f"start={one['current_time_utc']}  "
            f"target={one['target_utc']}  "
            f"CRPS={one['crps']:.6f}"
        )
        print(
            f"OK  output contract  "
            f"predict={one['max_ms']:.3f} ms  "
            f"elapsed={one['elapsed_s']:.3f} s"
        )
        sys.stdout.flush()
        if WRITE_SVG:
            write_forecast_picture(
                forecast_svg_path(int(one["target_ms"])),
                one["percentiles"],
                float(one["realized_price"]),
                one["current_time_utc"],
                one["target_utc"],
                float(one["crps"]),
            )

    try:
        loaded = load_payload(args.payload) if args.payload is not None else None
        report = validate(
            warmup=args.warmup,
            rounds=args.rounds,
            seed=args.seed,
            strict=args.strict,
            current_time_ms=args.current_time_ms,
            time_interval=args.time_interval,
            time_length=args.time_length,
            payload=loaded,
            allow_rest=args.payload is not None,
            on_test=on_test,
        )
    except ValidationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print()
    dropped = int(report.get("n_dropped") or 0)
    scored = int(report.get("n_scored") or 0)
    if report["crps"] is None:
        print(f"average CRPS=dropped  scored=0  dropped={dropped}")
        print(f"maximum predict time={report['max_ms']:.3f} ms")
        print(
            "FAIL: no prediction scored — spot book-ticker is not live at "
            "current_time_ms + 10s (SPECIFICATION.md).",
            file=sys.stderr,
        )
        return 1
    extra = f"  scored={scored}  dropped={dropped}" if dropped else ""
    print(f"average CRPS={report['crps']:.6f}{extra}")
    print(f"maximum predict time={report['max_ms']:.3f} ms")
    crps_path = crps_csv_path(str(report["asset"]))
    write_crps_csv(crps_path, report["reports"])
    print(f"wrote {crps_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
