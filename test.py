"""Score every saved payload / spot-book pair under database/.

    python fetch_payload.py --interval 57.171
    python test.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from synth_ultra.payload import load_payload
from synth_ultra.saved_pairs import book_path
from synth_ultra.scoring import realized_from_saved_book
from synth_ultra.validate import (
    ValidationError,
    crps_csv_path,
    validate_saved_pairs,
    write_crps_csv,
    _validate_one,
)

DEFAULT_DATABASE = Path(__file__).resolve().parent / "database"


def _print_one(i: int, one: dict[str, Any]) -> None:
    print(f"=== test {i} ===")
    if one["crps"] is None:
        print(
            f"start={one['current_time_utc']}  "
            f"target={one['target_utc']}  "
            f"CRPS=dropped (spot book-ticker not live at horizon)"
        )
    else:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CRPS for every database/{current_time_ms} payload and spot book pair"
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if median latency exceeds the 5 ms budget",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help=f"pair root (default {DEFAULT_DATABASE})",
    )
    parser.add_argument(
        "--payload",
        type=Path,
        default=None,
        help="score one payload JSON; uses spot_book_ticker.json beside it when present",
    )
    args = parser.parse_args(argv)

    try:
        if args.payload is not None:
            payload = load_payload(args.payload)
            book_file = book_path(args.payload.parent)
            book = None
            realized = None
            if book_file.is_file():
                from synth_ultra.saved_pairs import load_book

                book = load_book(book_file)
                anchor = int(payload["prompt"]["current_time_ms"])
                horizon = int(payload["prompt"].get("horizon_seconds", 10))
                realized = realized_from_saved_book(book, anchor, horizon)
            t1 = time.perf_counter()
            one = _validate_one(
                payload,
                rounds=args.rounds,
                warmup=args.warmup,
                **({} if book is None else {"realized_price": realized}),
            )
            one["elapsed_s"] = time.perf_counter() - t1
            one["book_recv_ts_ms"] = None if book is None else book.get("recv_ts_ms")
            _print_one(1, one)
            report = {
                "asset": str(payload["prompt"].get("asset") or "BTC"),
                "crps": one["crps"],
                "n_scored": 0 if one["crps"] is None else 1,
                "n_dropped": 1 if one["crps"] is None else 0,
                "max_ms": one["max_ms"],
                "reports": [one],
            }
        else:
            report = validate_saved_pairs(
                args.database,
                warmup=args.warmup,
                rounds=args.rounds,
                strict=args.strict,
                on_test=_print_one,
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
    else:
        extra = f"  scored={scored}  dropped={dropped}" if dropped else ""
        print(f"average CRPS={report['crps']:.6f}{extra}")
        print(f"maximum predict time={report['max_ms']:.3f} ms")
    crps_path = crps_csv_path(str(report["asset"]))
    write_crps_csv(crps_path, report["reports"])
    print(f"wrote {crps_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
