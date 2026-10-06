"""Shape-faithful sample payload for local validation (input.md schema_version 5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from synth_ultra.constants import (
    BOOK_TICKER_WINDOW_S,
    CANDLE_WINDOW_S,
    DEPTH_BAND_USD,
    DEPTH_BANDS,
    DEPTH_LEVELS,
    HORIZON_SECONDS,
    LIQUIDATION_WINDOW_S,
    NUM_PERCENTILES,
    SCHEMA_VERSION,
    TRADE_WINDOW_S,
    TRIGGER_KINDS,
    TRIGGER_VENUES,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CURRENT_TIME_MS = 1_700_000_000_000

def _candles(
    n: int,
    t0: int,
    px: float,
    rng: np.random.Generator,
    complete: bool,
    *,
    compact: bool = False,
) -> dict:
    if not complete:
        n = max(n // 2, 4) if compact else max(n // 6, 30)
    rets = rng.normal(0.0, 3e-5, size=n)
    close = px * np.exp(np.cumsum(rets))
    high = close * (1.0 + rng.uniform(0.0, 8e-5, size=n))
    low = close * (1.0 - rng.uniform(0.0, 8e-5, size=n))
    open_ = np.empty(n, dtype=np.float64)
    open_[0] = px
    open_[1:] = close[:-1]
    vol = rng.uniform(0.1, 4.0, size=n)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    ohlcv = np.stack([open_, high, low, close, vol], axis=1)
    open_time_ms = t0 + np.arange(n, dtype=np.int64) * 1000
    return {
        "open_time_ms": open_time_ms,
        "ohlcv": ohlcv,
        "complete_history": bool(complete) and history_is_complete(open_time_ms),
    }


def history_is_complete(open_time_ms: np.ndarray) -> bool:
    """True when the candle array holds a full trailing hour of 1s bars.

    input.md: ``complete_history`` is whether that hour is populated. Spot
    evaluation payloads are complete; futures is false while its trade-built
    history is still filling and then contains only the seconds available.
    """
    times = np.asarray(open_time_ms)
    if times.size < CANDLE_WINDOW_S:
        return False
    return int(times[-1]) - int(times[0]) >= (CANDLE_WINDOW_S - 1) * 1000


def empty_liquidations() -> dict:
    """Futures-only stream. Spot is present and always empty (input.md)."""
    return {
        "ts_ms": np.array([], dtype=np.int64),
        "recv_ts_ms": np.array([], dtype=np.int64),
        "price": np.array([], dtype=np.float64),
        "qty": np.array([], dtype=np.float64),
        "is_buy": np.array([], dtype=bool),
    }


def empty_depth_bands() -> dict:
    """No snapshot yet: recv_ts_ms 0 and zero-length sides (input.md)."""
    return {
        "anchor_price": 0.0,
        "bucket_usd": float(DEPTH_BAND_USD),
        "bids": np.array([], dtype=np.float64),
        "asks": np.array([], dtype=np.float64),
        "recv_ts_ms": 0,
    }


def depth_bands_from_book(
    bids,
    asks,
    recv_ms: int,
    *,
    bucket_usd: float = DEPTH_BAND_USD,
    n_bands: int = DEPTH_BANDS,
) -> dict:
    """Fold a spot book into absolute $10 bands around the snapshot mid.

    ``bids[i]`` is size in ``[anchor-(i+1)*bucket, anchor-i*bucket)`` and
    ``asks[i]`` is the mirror above the anchor. The anchor is that snapshot's
    mid, not the live mid at prediction time.
    """
    bid_book = np.asarray(bids, dtype=np.float64)
    ask_book = np.asarray(asks, dtype=np.float64)
    if (
        bid_book.ndim != 2
        or ask_book.ndim != 2
        or bid_book.shape[0] == 0
        or ask_book.shape[0] == 0
        or bid_book.shape[1] < 2
        or ask_book.shape[1] < 2
    ):
        return empty_depth_bands()
    anchor = 0.5 * (float(bid_book[0, 0]) + float(ask_book[0, 0]))
    bucket = float(bucket_usd)
    if not np.isfinite(anchor) or anchor <= 0.0 or bucket <= 0.0:
        return empty_depth_bands()
    bid_qty = np.zeros(int(n_bands), dtype=np.float64)
    ask_qty = np.zeros(int(n_bands), dtype=np.float64)
    dist = (anchor - bid_book[:, 0]) / bucket
    idx = np.ceil(dist - 1e-9).astype(np.int64) - 1
    bq = bid_book[:, 1]
    ok = (idx >= 0) & (idx < n_bands) & np.isfinite(bq) & (bq > 0.0)
    np.add.at(bid_qty, idx[ok], bq[ok])
    dist_a = (ask_book[:, 0] - anchor) / bucket
    idx_a = np.floor(dist_a + 1e-9).astype(np.int64)
    aq = ask_book[:, 1]
    ok_a = (idx_a >= 0) & (idx_a < n_bands) & np.isfinite(aq) & (aq > 0.0)
    np.add.at(ask_qty, idx_a[ok_a], aq[ok_a])
    return {
        "anchor_price": float(anchor),
        "bucket_usd": bucket,
        "bids": bid_qty,
        "asks": ask_qty,
        "recv_ts_ms": int(recv_ms),
    }


def _trades(n: int, t0: int, px: float, rng: np.random.Generator) -> dict:
    # Trailing 60s is (t0, t0 + 60_000], t0 = current_time_ms - 60_000.
    ts = t0 + rng.integers(1, TRADE_WINDOW_S * 1000 + 1, size=n, dtype=np.int64)
    price = px * np.exp(rng.normal(0.0, 4e-5, size=n))
    qty = rng.uniform(0.001, 0.8, size=n)
    maker = rng.random(n) < 0.5
    event = ts + rng.integers(0, 3, size=n, dtype=np.int64)
    recv = event + rng.integers(0, 8, size=n, dtype=np.int64)
    order = np.argsort(recv, kind="mergesort")
    ts, event, recv = ts[order], event[order], recv[order]
    price, qty, maker = price[order], qty[order], maker[order]
    return {
        "ts_ms": ts,
        "event_ts_ms": event,
        "recv_ts_ms": recv,
        "price": price.astype(np.float64),
        "qty": qty.astype(np.float64),
        "buyer_is_maker": maker,
    }


def _book_ticker(
    n: int, t0: int, px: float, rng: np.random.Generator, *, futures: bool
) -> dict:
    recv = np.linspace(t0 + 1, t0 + BOOK_TICKER_WINDOW_S * 1000, n, dtype=np.int64)
    mid = px * np.exp(np.cumsum(rng.normal(0.0, 8e-6, size=n)))
    half = 0.5 * rng.uniform(0.5, 2.5, size=n)
    bid_p = mid - half
    ask_p = mid + half
    bid_q = rng.uniform(0.05, 3.0, size=n)
    ask_q = rng.uniform(0.05, 3.0, size=n)
    out: dict[str, Any] = {
        "recv_ts_ms": recv,
        "bid_price": bid_p.astype(np.float64),
        "bid_qty": bid_q.astype(np.float64),
        "ask_price": ask_p.astype(np.float64),
        "ask_qty": ask_q.astype(np.float64),
    }
    if futures:
        out["event_ts_ms"] = recv - 1
        out["transaction_ts_ms"] = recv - 2
    return out


def _book_side(px: float, rng: np.random.Generator, *, bids: bool) -> np.ndarray:
    step = rng.uniform(0.5, 2.0, size=DEPTH_LEVELS)
    if bids:
        prices = px - np.cumsum(step)
    else:
        prices = px + np.cumsum(step)
    qty = rng.uniform(0.05, 5.0, size=DEPTH_LEVELS)
    return np.stack([prices, qty], axis=1).astype(np.float64)


def _depth_snapshot(t_ms: int, px: float, rng: np.random.Generator, *, futures: bool) -> dict:
    return {
        "recv_ts_ms": np.int64(t_ms),
        "update_id": int(rng.integers(1_000_000, 9_000_000)),
        "bids": _book_side(px, rng, bids=True),
        "asks": _book_side(px, rng, bids=False),
        "event_ts_ms": int(t_ms - 1) if futures else None,
        "transaction_ts_ms": int(t_ms - 2) if futures else None,
    }


def _depth_updates(
    n: int, t0: int, px: float, rng: np.random.Generator, *, futures: bool, start_id: int
) -> list[dict]:
    updates = []
    uid = start_id
    for i in range(n):
        uid += 1
        first = uid
        final = uid + int(rng.integers(0, 4))
        uid = final
        recv = t0 + int((i + 1) * (TRADE_WINDOW_S * 1000 / max(n, 1)))
        u: dict[str, Any] = {
            "recv_ts_ms": np.int64(recv),
            "event_ts_ms": np.int64(recv - 1),
            "first_id": int(first),
            "final_id": int(final),
            "prev_final_id": int(first - 1),
            "bids": _book_side(px, rng, bids=True)[: rng.integers(1, 6)],
            "asks": _book_side(px, rng, bids=False)[: rng.integers(1, 6)],
        }
        if futures:
            u["transaction_ts_ms"] = np.int64(recv - 2)
        updates.append(u)
    return updates


def _liquidations(
    n: int,
    current_time_ms: int,
    px: float,
    rng: np.random.Generator,
    *,
    futures: bool,
) -> dict:
    """Trailing hour of forced closes. ``is_buy`` is the order side.

    True means the exchange is buying to close a short (price pushed up).
    Spot has no liquidations.
    """
    if not futures or n <= 0:
        return empty_liquidations()
    recv = np.linspace(
        current_time_ms - LIQUIDATION_WINDOW_S * 1000 + 1000,
        current_time_ms - 200,
        n,
        dtype=np.int64,
    )
    price = px + rng.normal(0.0, 4.0, size=n)
    qty = rng.uniform(0.001, 0.3, size=n)
    is_buy = rng.random(n) < 0.5
    return {
        "ts_ms": recv - 40,
        "recv_ts_ms": recv,
        "price": price.astype(np.float64),
        "qty": qty.astype(np.float64),
        "is_buy": is_buy,
    }


def _depth_bands(
    px: float,
    rng: np.random.Generator,
    current_time_ms: int,
    *,
    futures: bool,
) -> dict:
    """Spot-only coarse book. Anchor is a past mid, not the live price."""
    if futures:
        return empty_depth_bands()
    anchor = float(px) - 35.0
    return {
        "anchor_price": anchor,
        "bucket_usd": float(DEPTH_BAND_USD),
        "bids": rng.uniform(0.05, 3.0, size=DEPTH_BANDS).astype(np.float64),
        "asks": rng.uniform(0.05, 3.0, size=DEPTH_BANDS).astype(np.float64),
        "recv_ts_ms": int(current_time_ms) - 15_000,
    }


def _venue(
    *,
    current_time_ms: int,
    t0: int,
    px: float,
    rng: np.random.Generator,
    futures: bool,
    complete_candles: bool,
    compact: bool = False,
) -> dict:
    n_trades = 6 if compact else (400 if futures else 350)
    n_bt = 8 if compact else (800 if futures else 700)
    n_upd = 2 if compact else 40
    n_liq = 3 if compact else 12
    n_candles = 8 if compact else CANDLE_WINDOW_S
    candles = _candles(
        n_candles,
        current_time_ms - (n_candles - 1) * 1000,
        px,
        rng,
        complete_candles,
        compact=compact,
    )
    trades = _trades(n_trades, t0, px, rng)
    book_ticker = _book_ticker(n_bt, t0, px, rng, futures=futures)
    if candles["ohlcv"].size:
        px = float(candles["ohlcv"][-1, 3])
    start = _depth_snapshot(t0, px, rng, futures=futures)
    latest = _depth_snapshot(current_time_ms, px, rng, futures=futures)
    updates = _depth_updates(n_upd, t0, px, rng, futures=futures, start_id=start["update_id"])
    last_event_times: dict[str, np.int64] = {
        "trade": np.int64(int(trades["recv_ts_ms"][-1])),
        "bookTicker": np.int64(int(book_ticker["recv_ts_ms"][-1])),
        "kline_1s": np.int64(int(candles["open_time_ms"][-1])),
        "depth": np.int64(int(latest["recv_ts_ms"])),
    }
    return {
        "symbol": "BTCUSDT",
        "candles_1s": candles,
        "trades": trades,
        "book_ticker": book_ticker,
        "depth_start": start,
        "depth_updates": updates,
        "depth_latest": latest,
        "liquidations": _liquidations(n_liq, current_time_ms, px, rng, futures=futures),
        "depth_bands": _depth_bands(px, rng, current_time_ms, futures=futures),
        "last_event_times": last_event_times,
    }


def make_sample_payload(
    seed: int = 0,
    *,
    price: float = 97_500.0,
    current_time_ms: int | None = None,
    compact: bool = False,
) -> dict:
    """Build a schema_version-5 payload with NumPy arrays, oldest-first.

    ``current_time_ms`` is the forecast anchor. Candles are a trailing hour and
    trades / book ticker a trailing 60s of shape-faithful synthetic data.
    ``compact=True`` keeps the same keys and dtypes but short arrays.
    """
    rng = np.random.default_rng(seed)
    now = int(current_time_ms if current_time_ms is not None else _DEFAULT_CURRENT_TIME_MS)
    t0 = now - TRADE_WINDOW_S * 1000
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt": {
            "asset": "BTC",
            "horizon_seconds": HORIZON_SECONDS,
            "num_percentiles": NUM_PERCENTILES,
            "quantile_grid": "centered-100",
            "current_time_ms": now,
            "trigger": {"kind": "time", "venue": None},
        },
        "venues": {
            "spot": _venue(
                current_time_ms=now,
                t0=t0,
                px=price,
                rng=rng,
                futures=False,
                complete_candles=True,
                compact=compact,
            ),
            "futures": _venue(
                current_time_ms=now,
                t0=t0,
                px=price + 8.0,
                rng=rng,
                futures=True,
                complete_candles=False,
                compact=compact,
            ),
        },
    }


def payload_to_jsonable(obj: Any) -> Any:
    """Convert NumPy arrays / scalars so the payload can be written as JSON."""
    if isinstance(obj, dict):
        return {str(k): payload_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [payload_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _as_i64(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.int64)


def _as_f64(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _depth_snapshot_from_jsonable(snap: dict) -> dict:
    bids = _as_f64(snap.get("bids", []))
    asks = _as_f64(snap.get("asks", []))
    if bids.ndim != 2:
        bids = bids.reshape(-1, 2) if bids.size else np.zeros((0, 2), dtype=np.float64)
    if asks.ndim != 2:
        asks = asks.reshape(-1, 2) if asks.size else np.zeros((0, 2), dtype=np.float64)
    event = snap.get("event_ts_ms")
    tx = snap.get("transaction_ts_ms")
    return {
        "recv_ts_ms": np.int64(snap["recv_ts_ms"]),
        "update_id": int(snap["update_id"]),
        "bids": np.ascontiguousarray(bids, dtype=np.float64),
        "asks": np.ascontiguousarray(asks, dtype=np.float64),
        "event_ts_ms": None if event is None else int(event),
        "transaction_ts_ms": None if tx is None else int(tx),
    }


def _depth_update_from_jsonable(upd: dict, *, futures: bool) -> dict:
    bids = _as_f64(upd.get("bids", []))
    asks = _as_f64(upd.get("asks", []))
    if bids.ndim != 2:
        bids = bids.reshape(-1, 2) if bids.size else np.zeros((0, 2), dtype=np.float64)
    if asks.ndim != 2:
        asks = asks.reshape(-1, 2) if asks.size else np.zeros((0, 2), dtype=np.float64)
    prev = upd.get("prev_final_id")
    out: dict[str, Any] = {
        "recv_ts_ms": np.int64(upd["recv_ts_ms"]),
        "event_ts_ms": np.int64(upd["event_ts_ms"]),
        "first_id": int(upd["first_id"]),
        "final_id": int(upd["final_id"]),
        # Spot has no `pu`, so live prompts send null here.
        "prev_final_id": None if prev is None else int(prev),
        "bids": np.ascontiguousarray(bids, dtype=np.float64),
        "asks": np.ascontiguousarray(asks, dtype=np.float64),
    }
    if futures or "transaction_ts_ms" in upd:
        tx = upd.get("transaction_ts_ms")
        out["transaction_ts_ms"] = None if tx is None else np.int64(tx)
    return out


def _liquidations_from_jsonable(raw: dict | None) -> dict:
    if not raw:
        return empty_liquidations()
    return {
        "ts_ms": _as_i64(raw.get("ts_ms") or []),
        "recv_ts_ms": _as_i64(raw.get("recv_ts_ms") or []),
        "price": _as_f64(raw.get("price") or []),
        "qty": _as_f64(raw.get("qty") or []),
        "is_buy": np.asarray(raw.get("is_buy") or [], dtype=bool).reshape(-1),
    }


def _depth_bands_from_jsonable(raw: dict | None) -> dict:
    if not raw:
        return empty_depth_bands()
    bids = _as_f64(raw.get("bids") or []).reshape(-1)
    asks = _as_f64(raw.get("asks") or []).reshape(-1)
    return {
        "anchor_price": float(raw.get("anchor_price") or 0.0),
        "bucket_usd": float(raw.get("bucket_usd") or DEPTH_BAND_USD),
        "bids": np.ascontiguousarray(bids, dtype=np.float64),
        "asks": np.ascontiguousarray(asks, dtype=np.float64),
        "recv_ts_ms": int(raw.get("recv_ts_ms") or 0),
    }


def _venue_from_jsonable(venue: dict, *, futures: bool) -> dict:
    candles = venue["candles_1s"]
    trades = venue["trades"]
    bt = venue["book_ticker"]
    book: dict[str, Any] = {
        "recv_ts_ms": _as_i64(bt["recv_ts_ms"]),
        "bid_price": _as_f64(bt["bid_price"]),
        "bid_qty": _as_f64(bt["bid_qty"]),
        "ask_price": _as_f64(bt["ask_price"]),
        "ask_qty": _as_f64(bt["ask_qty"]),
    }
    if futures:
        book["event_ts_ms"] = _as_i64(bt["event_ts_ms"])
        book["transaction_ts_ms"] = _as_i64(bt["transaction_ts_ms"])
    last_events = {
        str(k): np.int64(v) for k, v in (venue.get("last_event_times") or {}).items()
    }
    return {
        "symbol": str(venue.get("symbol") or "BTCUSDT"),
        "candles_1s": {
            "open_time_ms": _as_i64(candles["open_time_ms"]),
            "ohlcv": np.ascontiguousarray(_as_f64(candles["ohlcv"]).reshape(-1, 5), dtype=np.float64),
            "complete_history": bool(candles["complete_history"]),
        },
        "trades": {
            "ts_ms": _as_i64(trades["ts_ms"]),
            "event_ts_ms": _as_i64(trades["event_ts_ms"]),
            "recv_ts_ms": _as_i64(trades["recv_ts_ms"]),
            "price": _as_f64(trades["price"]),
            "qty": _as_f64(trades["qty"]),
            "buyer_is_maker": np.asarray(trades["buyer_is_maker"], dtype=bool),
        },
        "book_ticker": book,
        "depth_start": _depth_snapshot_from_jsonable(venue["depth_start"]),
        "depth_updates": [
            _depth_update_from_jsonable(u, futures=futures) for u in venue.get("depth_updates") or []
        ],
        "depth_latest": _depth_snapshot_from_jsonable(venue["depth_latest"]),
        "liquidations": _liquidations_from_jsonable(venue.get("liquidations")),
        "depth_bands": _depth_bands_from_jsonable(venue.get("depth_bands")),
        "last_event_times": last_events,
    }


ENV_PAYLOAD_JSON = REPO_ROOT / "examples" / "env_payload.json"
SAMPLE_PAYLOAD_JSON = REPO_ROOT / "examples" / "sample_payload.json"


def payload_from_jsonable(obj: dict) -> dict:
    """Restore NumPy arrays / scalars from a JSON-decoded environment payload."""
    prompt = dict(obj["prompt"])
    prompt["current_time_ms"] = int(prompt["current_time_ms"])
    prompt["horizon_seconds"] = int(prompt.get("horizon_seconds", HORIZON_SECONDS))
    prompt["num_percentiles"] = int(prompt.get("num_percentiles", NUM_PERCENTILES))
    return {
        "schema_version": int(obj["schema_version"]),
        "prompt": prompt,
        "venues": {
            "spot": _venue_from_jsonable(obj["venues"]["spot"], futures=False),
            "futures": _venue_from_jsonable(obj["venues"]["futures"], futures=True),
        },
    }


def load_payload(path: Path | None = None) -> dict:
    """Load a saved environment payload JSON into the live dict-of-arrays shape."""
    target = Path(path) if path is not None else ENV_PAYLOAD_JSON
    return payload_from_jsonable(json.loads(target.read_text(encoding="utf-8")))


def _assert_time_order(times: Any, label: str) -> None:
    arr = np.asarray(times)
    if arr.size >= 2 and np.any(arr[1:] < arr[:-1]):
        raise ValueError(label)


def _assert_book_best_first(levels: Any, label: str, *, bids: bool) -> None:
    book = np.asarray(levels)
    if book.ndim != 2 or book.shape[0] < 2:
        return
    prices = book[:, 0]
    if bids and np.any(prices[1:] > prices[:-1]):
        raise ValueError(label)
    if not bids and np.any(prices[1:] < prices[:-1]):
        raise ValueError(label)


def _assert_same_len(rows: dict, keys: tuple[str, ...], label: str) -> int:
    n = len(np.asarray(rows[keys[0]]))
    for key in keys[1:]:
        if len(np.asarray(rows[key])) != n:
            raise ValueError(f"{label}.{key}")
    return n


def assert_payload_schema(payload: dict) -> None:
    """Fail if the payload does not match input.md schema_version 5."""
    if int(payload["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("schema_version")
    prompt = payload["prompt"]
    for key in (
        "asset",
        "horizon_seconds",
        "num_percentiles",
        "quantile_grid",
        "current_time_ms",
        "trigger",
    ):
        if key not in prompt:
            raise ValueError(f"prompt.{key}")
    trigger = prompt["trigger"] or {}
    kind = trigger.get("kind")
    if kind not in TRIGGER_KINDS:
        raise ValueError("prompt.trigger.kind")
    venue = trigger.get("venue")
    if kind == "time":
        if venue is not None:
            raise ValueError("prompt.trigger.venue")
    elif venue not in TRIGGER_VENUES:
        raise ValueError("prompt.trigger.venue")
    if int(prompt["horizon_seconds"]) != HORIZON_SECONDS:
        raise ValueError("prompt.horizon_seconds")
    if int(prompt["num_percentiles"]) != NUM_PERCENTILES:
        raise ValueError("prompt.num_percentiles")
    if str(prompt["quantile_grid"]) != "centered-100":
        raise ValueError("prompt.quantile_grid")
    for name, futures in (("spot", False), ("futures", True)):
        venue = payload["venues"][name]
        if venue["symbol"] != "BTCUSDT":
            raise ValueError(f"{name}.symbol")
        candles = venue["candles_1s"]
        ohlcv = np.asarray(candles["ohlcv"])
        if ohlcv.ndim != 2 or ohlcv.shape[-1] != 5:
            raise ValueError(f"{name}.candles_1s.ohlcv")
        if "complete_history" not in candles:
            raise ValueError(f"{name}.candles_1s.complete_history")
        trades = venue["trades"]
        for key in ("ts_ms", "event_ts_ms", "recv_ts_ms", "price", "qty", "buyer_is_maker"):
            if key not in trades:
                raise ValueError(f"{name}.trades.{key}")
        bt = venue["book_ticker"]
        for key in ("recv_ts_ms", "bid_price", "bid_qty", "ask_price", "ask_qty"):
            if key not in bt:
                raise ValueError(f"{name}.book_ticker.{key}")
        if futures:
            if "event_ts_ms" not in bt or "transaction_ts_ms" not in bt:
                raise ValueError("futures.book_ticker exchange times")
        elif "event_ts_ms" in bt or "transaction_ts_ms" in bt:
            raise ValueError("spot.book_ticker must omit E/T")
        for snap_name in ("depth_start", "depth_latest"):
            snap = venue[snap_name]
            bids = np.asarray(snap["bids"])
            asks = np.asarray(snap["asks"])
            if bids.ndim != 2 or bids.shape[1] != 2:
                raise ValueError(f"{name}.{snap_name}.bids")
            if asks.ndim != 2 or asks.shape[1] != 2:
                raise ValueError(f"{name}.{snap_name}.asks")
            if futures:
                if snap["event_ts_ms"] is None or snap["transaction_ts_ms"] is None:
                    raise ValueError(f"futures.{snap_name} exchange times")
            elif snap["event_ts_ms"] is not None or snap["transaction_ts_ms"] is not None:
                raise ValueError(f"spot.{snap_name} E/T must be None")
        for upd in venue["depth_updates"]:
            if "event_ts_ms" not in upd:
                raise ValueError(f"{name}.depth_updates.event_ts_ms")
            tx = upd.get("transaction_ts_ms", None)
            if futures:
                if tx is None or upd.get("prev_final_id") is None:
                    raise ValueError("futures.depth_updates exchange ids")
            elif tx is not None:
                raise ValueError("spot.depth_updates T must be null")
        liq = venue.get("liquidations")
        if not isinstance(liq, dict):
            raise ValueError(f"{name}.liquidations")
        for key in ("ts_ms", "recv_ts_ms", "price", "qty", "is_buy"):
            if key not in liq:
                raise ValueError(f"{name}.liquidations.{key}")
        n_liq = _assert_same_len(
            liq, ("ts_ms", "recv_ts_ms", "price", "qty", "is_buy"), f"{name}.liquidations"
        )
        if not futures and n_liq != 0:
            raise ValueError("spot.liquidations must be empty")
        _assert_time_order(liq["recv_ts_ms"], f"{name}.liquidations.recv_ts_ms")
        bands = venue.get("depth_bands")
        if not isinstance(bands, dict):
            raise ValueError(f"{name}.depth_bands")
        for key in ("anchor_price", "bucket_usd", "bids", "asks", "recv_ts_ms"):
            if key not in bands:
                raise ValueError(f"{name}.depth_bands.{key}")
        band_bids = np.asarray(bands["bids"]).reshape(-1)
        band_asks = np.asarray(bands["asks"]).reshape(-1)
        band_recv = int(bands["recv_ts_ms"])
        if futures:
            if band_recv != 0 or band_bids.size or band_asks.size:
                raise ValueError("futures.depth_bands must be empty")
        elif band_recv == 0:
            if band_bids.size or band_asks.size:
                raise ValueError(f"{name}.depth_bands")
        else:
            if band_bids.shape != band_asks.shape or float(bands["bucket_usd"]) <= 0.0:
                raise ValueError(f"{name}.depth_bands")
            if float(bands["anchor_price"]) <= 0.0:
                raise ValueError(f"{name}.depth_bands.anchor_price")
        last = venue.get("last_event_times")
        if not isinstance(last, dict):
            raise ValueError(f"{name}.last_event_times")
        for stream, ts in last.items():
            if isinstance(ts, bool) or not isinstance(ts, (int, np.integer)):
                raise ValueError(f"{name}.last_event_times.{stream}")
        _assert_time_order(candles["open_time_ms"], f"{name}.candles_1s.open_time_ms")
        _assert_time_order(trades["recv_ts_ms"], f"{name}.trades.recv_ts_ms")
        _assert_time_order(bt["recv_ts_ms"], f"{name}.book_ticker.recv_ts_ms")
        if history_is_complete(candles["open_time_ms"]) != bool(candles["complete_history"]):
            raise ValueError(f"{name}.candles_1s.complete_history")
        n_tr = len(np.asarray(trades["ts_ms"]))
        for key in ("event_ts_ms", "recv_ts_ms", "price", "qty", "buyer_is_maker"):
            if len(np.asarray(trades[key])) != n_tr:
                raise ValueError(f"{name}.trades.{key}")
        n_bt = len(np.asarray(bt["recv_ts_ms"]))
        for key in ("bid_price", "bid_qty", "ask_price", "ask_qty"):
            if len(np.asarray(bt[key])) != n_bt:
                raise ValueError(f"{name}.book_ticker.{key}")
        for snap_name in ("depth_start", "depth_latest"):
            _assert_book_best_first(venue[snap_name]["bids"], f"{name}.{snap_name}.bids", bids=True)
            _assert_book_best_first(venue[snap_name]["asks"], f"{name}.{snap_name}.asks", bids=False)


def write_sample_payload(path: Path | None = None, **kwargs: Any) -> Path:
    """Write a shape-faithful compact sample matching input.md."""
    kwargs.setdefault("seed", 0)
    kwargs.setdefault("compact", True)
    kwargs.setdefault("current_time_ms", 1_700_000_000_000)
    payload = make_sample_payload(**kwargs)
    assert_payload_schema(payload)
    target = Path(path) if path is not None else SAMPLE_PAYLOAD_JSON
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload_to_jsonable(payload), indent=2), encoding="utf-8")
    return target
