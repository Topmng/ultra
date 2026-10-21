"""Score every saved payload / spot-book pair under database/.

    python fetch_payload.py
    python test.py
    python test.py --plot
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TextIO

from harness.payload import load_payload
from harness.saved_pairs import BookReadError, book_path
from harness.scoring import realized_from_saved_book
from harness.validate import (
    ValidationError,
    crps_csv_path,
    forecast_svg_path,
    validate_saved_pairs,
    write_crps_csv,
    write_forecast_picture,
    _validate_one,
)

DEFAULT_DATABASE = Path(__file__).resolve().parent / "database"
LOG_DIR = Path(__file__).resolve().parent / "logs"


class _Tee:
    """Write each chunk to the terminal and a log file."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


@contextmanager
def _terminal_log() -> Iterator[Path]:
    """Copy stdout and stderr into ``logs/test_<utc>.log`` for the run."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / datetime.now(timezone.utc).strftime("test_%Y-%m-%d_%H-%M-%S.log")
    stdout, stderr = sys.stdout, sys.stderr
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        sys.stdout = _Tee(stdout, handle)
        sys.stderr = _Tee(stderr, handle)
        try:
            print(f"log={path.resolve()}")
            yield path
        finally:
            sys.stdout = stdout
            sys.stderr = stderr


def _print_one(i: int, one: dict[str, Any], *, plot: bool = False) -> None:
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
    if plot and one["crps"] is not None and one["realized_price"] is not None:
        path = forecast_svg_path(int(one["target_ms"]))
        horizon_s = int(one["target_ms"]) - int(one["current_time_ms"])
        write_forecast_picture(
            path,
            one["percentiles"],
            float(one["realized_price"]),
            str(one["current_time_utc"]),
            str(one["target_utc"]),
            float(one["crps"]),
            history_prices=one.get("history_prices"),
            history_offset_s=one.get("history_offset_s"),
            hour_prices=one.get("hour_prices"),
            hour_offset_s=one.get("hour_offset_s"),
            horizon_seconds=max(0, horizon_s // 1000),
            mu_px=one.get("mu_px"),
            anchor_px=one.get("anchor_px"),
        )
        print(f"wrote {path.resolve()}")
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
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "write an SVG under plot/ for each scored pair: the past hour on "
            "the left, and the last 100s, +10s aim, and 100 predictions on "
            "the right (default off)"
        ),
    )
    args = parser.parse_args(argv)

    with _terminal_log():
        return _run(args)


def _run(args: argparse.Namespace) -> int:
    try:
        if args.payload is not None:
            payload = load_payload(args.payload)
            book_file = book_path(args.payload.parent)
            book = None
            realized = None
            if book_file.is_file():
                from harness.saved_pairs import load_book

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
            _print_one(1, one, plot=args.plot)
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
                on_test=lambda i, one: _print_one(i, one, plot=args.plot),
            )
    except (ValidationError, BookReadError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print()
    dropped = int(report.get("n_dropped") or 0)
    scored = int(report.get("n_scored") or 0)
    unreadable = int(report.get("n_unreadable") or 0)
    if report["crps"] is None:
        extra = f"  unreadable={unreadable}" if unreadable else ""
        print(f"average CRPS=dropped  scored=0  dropped={dropped}{extra}")
        print(f"maximum predict time={report['max_ms']:.3f} ms")
    else:
        extra = f"  scored={scored}  dropped={dropped}" if dropped else ""
        if unreadable:
            extra += f"  unreadable={unreadable}"
        print(f"average CRPS={report['crps']:.6f}{extra}")
        print(f"maximum predict time={report['max_ms']:.3f} ms")
    crps_path = crps_csv_path(str(report["asset"]))
    write_crps_csv(crps_path, report["reports"])
    print(f"wrote {crps_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
