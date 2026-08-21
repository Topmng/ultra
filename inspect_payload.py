"""Inspect a sample or saved payload and the model output.

    python inspect_payload.py
    python inspect_payload.py --payload examples/env_payload.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from synth_ultra.model import predict_percentiles
from synth_ultra.payload import ENV_PAYLOAD_JSON, load_payload, make_sample_payload, payload_to_jsonable
from synth_ultra.scoring import pinball_crps, realized_spot_close
from synth_ultra.validate import check_output


def summarize_venue(name: str, venue: dict) -> None:
    candles = venue["candles_1s"]
    trades = venue["trades"]
    bt = venue["book_ticker"]
    print(f"  {name}  symbol={venue['symbol']}")
    print(
        f"    candles_1s     n={len(candles['open_time_ms'])}  "
        f"complete_history={candles['complete_history']}  "
        f"ohlcv[-1]={np.asarray(candles['ohlcv'])[-1].tolist()}"
    )
    n_trades = len(trades["ts_ms"])
    if n_trades:
        print(
            f"    trades         n={n_trades}  "
            f"last_price={float(np.asarray(trades['price'])[-1]):.2f}"
        )
    else:
        print("    trades         n=0")
    print(
        f"    book_ticker    n={len(bt['recv_ts_ms'])}  "
        f"bid={float(np.asarray(bt['bid_price'])[-1]):.2f}  "
        f"ask={float(np.asarray(bt['ask_price'])[-1]):.2f}"
    )
    latest = venue["depth_latest"]
    bids = np.asarray(latest["bids"])
    asks = np.asarray(latest["asks"])
    print(
        f"    depth_latest   levels={bids.shape[0]}  "
        f"best_bid={bids[0, 0]:.2f} x {bids[0, 1]:.3f}  "
        f"best_ask={asks[0, 0]:.2f} x {asks[0, 1]:.3f}"
    )
    print(f"    depth_updates  n={len(venue['depth_updates'])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show sample input and model output")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compact",
        action="store_true",
        default=True,
        help="short arrays (default); easier to inspect",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="full-size windows (1h candles, 60s trades/book)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("examples/sample_payload.json"),
        help="write JSON payload to this path",
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--payload",
        type=Path,
        nargs="?",
        const=ENV_PAYLOAD_JSON,
        default=None,
        help="load this JSON instead of a synthetic sample (bare flag: examples/env_payload.json)",
    )
    args = parser.parse_args(argv)

    if args.payload is not None:
        payload = load_payload(args.payload)
    else:
        payload = make_sample_payload(args.seed, compact=not args.full)
    prompt = payload["prompt"]
    print("input prompt")
    print(json.dumps(prompt, indent=2))
    print(f"schema_version = {payload['schema_version']}")
    summarize_venue("spot", payload["venues"]["spot"])
    summarize_venue("futures", payload["venues"]["futures"])

    if not args.no_save and args.payload is None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(
            json.dumps(payload_to_jsonable(payload), indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.save}")

    out = check_output(predict_percentiles(payload))
    print("\noutput predict_percentiles  shape", out.shape, out.dtype)
    print("q=0.005 (first)", f"{out[0]:.4f}")
    print("q=0.495 (mid)  ", f"{out[49]:.4f}")
    print("q=0.995 (last) ", f"{out[-1]:.4f}")
    np.set_printoptions(precision=4, suppress=True, linewidth=100, threshold=100)
    print(out)

    if args.payload is not None:
        from synth_ultra.constants import HORIZON_SECONDS

        anchor = int(prompt["current_time_ms"])
        y = realized_spot_close(anchor, allow_rest=True)
        if y is None:
            print(
                f"\nCRPS skipped: no spot 1s close at current_time+{HORIZON_SECONDS}s. "
                "Wait 10s after capture, or keep database/btc_spot_candles.csv covering that instant."
            )
        else:
            print(f"\nrealized_close={y:.4f}  CRPS={pinball_crps(out, y):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
