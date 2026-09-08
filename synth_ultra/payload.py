"""Shape-faithful sample payload for local validation (input.md schema_version 3)."""

from __future__ import annotations

import csv
import functools
import json
from pathlib import Path
from typing import Any

import numpy as np

from synth_ultra.constants import (
    BOOK_TICKER_WINDOW_S,
    CANDLE_WINDOW_S,
    DEPTH_LEVELS,
    HORIZON_SECONDS,
    NUM_PERCENTILES,
    SCHEMA_VERSION,
    TRADE_WINDOW_S,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = REPO_ROOT / "database"


def _data_csv(name: str) -> Path:
    dest = DATABASE_DIR / name
    legacy = REPO_ROOT / name
    if dest.is_file() or not legacy.is_file():
        return dest
    return legacy


SPOT_CANDLES_CSV = _data_csv("btc_spot_candles.csv")
FUTURES_CANDLES_CSV = _data_csv("btc_futures_candles.csv")
SPOT_TRADES_CSV = _data_csv("btc_spot_trades.csv")
FUTURES_TRADES_CSV = _data_csv("btc_futures_trades.csv")
SPOT_BOOK_TICKER_CSV = _data_csv("btc_spot_book_ticker.csv")
FUTURES_BOOK_TICKER_CSV = _data_csv("btc_futures_book_ticker.csv")
SPOT_DEPTH_CSV = _data_csv("btc_spot_depth.csv")
FUTURES_DEPTH_CSV = _data_csv("btc_futures_depth.csv")
FUTURES_BOOK_DEPTH_CSV = _data_csv("btc_futures_book_depth.csv")
SPOT_DEPTH_UPDATES_JSONL = _data_csv("btc_spot_depth_updates.jsonl")
FUTURES_DEPTH_UPDATES_JSONL = _data_csv("btc_futures_depth_updates.jsonl")


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
    ohlcv = np.stack([open_, high, low, close, vol], axis=1)
    open_time_ms = t0 + np.arange(n, dtype=np.int64) * 1000
    return {
        "open_time_ms": open_time_ms,
        "ohlcv": ohlcv,
        "complete_history": bool(complete),
    }


def _trades(n: int, t0: int, px: float, rng: np.random.Generator) -> dict:
    ts = t0 + rng.integers(0, TRADE_WINDOW_S * 1000, size=n, dtype=np.int64)
    ts.sort()
    price = px * np.exp(rng.normal(0.0, 4e-5, size=n))
    qty = rng.uniform(0.001, 0.8, size=n)
    maker = rng.random(n) < 0.5
    event = ts + rng.integers(0, 3, size=n, dtype=np.int64)
    recv = event + rng.integers(0, 8, size=n, dtype=np.int64)
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
    recv = t0 + np.linspace(0, BOOK_TICKER_WINDOW_S * 1000, n, dtype=np.int64)
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


def _npz_path(src: Path) -> Path:
    return src.with_name(src.name + ".npz")


def _npz_fresh(src: Path, cache: Path) -> bool:
    try:
        return cache.is_file() and cache.stat().st_mtime >= src.stat().st_mtime
    except OSError:
        return False


def _try_npz(src: Path) -> dict[str, np.ndarray] | None:
    cache = _npz_path(src)
    if not _npz_fresh(src, cache):
        return None
    try:
        with np.load(cache) as packed:
            return {name: np.asarray(packed[name]) for name in packed.files}
    except (OSError, ValueError, KeyError):
        return None


def _write_npz(src: Path, arrays: dict[str, np.ndarray]) -> None:
    cache = _npz_path(src)
    tmp = cache.with_name(cache.name + ".building.npz")
    try:
        np.savez(tmp, **arrays)
        tmp.replace(cache)
    except OSError:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)


def _empty_trades() -> dict[str, np.ndarray]:
    return {
        "ts_ms": np.array([], dtype=np.int64),
        "price": np.array([], dtype=np.float64),
        "qty": np.array([], dtype=np.float64),
        "buyer_is_maker": np.array([], dtype=bool),
    }


def _parse_maker_column(path: Path, n: int) -> np.ndarray:
    maker = np.empty(n, dtype=bool)
    with path.open("rb") as handle:
        handle.readline()
        i = 0
        for line in handle:
            if not line.strip():
                continue
            token = line.rsplit(b",", 1)[-1].lstrip()[:1]
            maker[i] = token in b"Tt1"
            i += 1
            if i >= n:
                break
    if i < n:
        return maker[:i]
    return maker


@functools.lru_cache(maxsize=8)
def _load_candles_csv(path_str: str) -> dict[str, np.ndarray] | None:
    path = Path(path_str)
    if not path.is_file():
        return None
    cached = _try_npz(path)
    if cached is not None and "open_time_ms" in cached and "ohlcv" in cached:
        return {
            "open_time_ms": cached["open_time_ms"].astype(np.int64, copy=False),
            "ohlcv": cached["ohlcv"].astype(np.float64, copy=False),
        }
    try:
        arr = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 2, 3, 4, 5, 6))
    except Exception:
        return None
    if arr.size == 0:
        return {
            "open_time_ms": np.array([], dtype=np.int64),
            "ohlcv": np.zeros((0, 5), dtype=np.float64),
        }
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out = {
        "open_time_ms": arr[:, 0].astype(np.int64, copy=False),
        "ohlcv": arr[:, 1:6].astype(np.float64, copy=False),
    }
    _write_npz(path, out)
    return out


@functools.lru_cache(maxsize=8)
def _load_trades_csv(path_str: str) -> dict[str, np.ndarray] | None:
    path = Path(path_str)
    if not path.is_file():
        return None
    cached = _try_npz(path)
    if cached is not None and "ts_ms" in cached:
        return {
            "ts_ms": cached["ts_ms"].astype(np.int64, copy=False),
            "price": cached["price"].astype(np.float64, copy=False),
            "qty": cached["qty"].astype(np.float64, copy=False),
            "buyer_is_maker": cached["buyer_is_maker"].astype(bool, copy=False),
        }
    try:
        numeric = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 3, 4), dtype=np.float64)
    except Exception:
        return _empty_trades()
    if numeric.size == 0:
        return _empty_trades()
    if numeric.ndim == 1:
        numeric = numeric.reshape(1, -1)
    maker = _parse_maker_column(path, numeric.shape[0])
    n = min(numeric.shape[0], maker.size)
    out = {
        "ts_ms": numeric[:n, 0].astype(np.int64, copy=False),
        "price": numeric[:n, 1].astype(np.float64, copy=False),
        "qty": numeric[:n, 2].astype(np.float64, copy=False),
        "buyer_is_maker": maker[:n],
    }
    _write_npz(path, out)
    return out


def _window_slice(times: np.ndarray, lo_exclusive: int, hi_inclusive: int) -> slice:
    i0 = int(np.searchsorted(times, lo_exclusive, side="right"))
    i1 = int(np.searchsorted(times, hi_inclusive, side="right"))
    return slice(i0, i1)


def _candles_at(path: Path, current_time_ms: int, *, compact: bool) -> dict | None:
    table = _load_candles_csv(str(path))
    if table is None:
        return None
    times = table["open_time_ms"]
    sl = _window_slice(times, current_time_ms - CANDLE_WINDOW_S * 1000, current_time_ms)
    open_time_ms = times[sl]
    ohlcv = table["ohlcv"][sl]
    if open_time_ms.size == 0:
        return None
    complete = int(open_time_ms.size) >= CANDLE_WINDOW_S
    if compact and open_time_ms.size > 8:
        open_time_ms = open_time_ms[-8:]
        ohlcv = ohlcv[-8:]
    return {
        "open_time_ms": np.ascontiguousarray(open_time_ms, dtype=np.int64),
        "ohlcv": np.ascontiguousarray(ohlcv, dtype=np.float64),
        "complete_history": bool(complete),
    }


def _trades_at(path: Path, current_time_ms: int, *, compact: bool) -> dict | None:
    table = _load_trades_csv(str(path))
    if table is None:
        return None
    times = table["ts_ms"]
    sl = _window_slice(times, current_time_ms - TRADE_WINDOW_S * 1000, current_time_ms)
    ts = times[sl]
    if ts.size == 0:
        return None
    if compact and ts.size > 6:
        sl_last = slice(ts.size - 6, ts.size)
        ts = ts[sl_last]
        price = table["price"][sl][sl_last]
        qty = table["qty"][sl][sl_last]
        maker = table["buyer_is_maker"][sl][sl_last]
    else:
        price = table["price"][sl]
        qty = table["qty"][sl]
        maker = table["buyer_is_maker"][sl]
    ts = np.ascontiguousarray(ts, dtype=np.int64)
    return {
        "ts_ms": ts,
        "event_ts_ms": ts.copy(),
        "recv_ts_ms": ts.copy(),
        "price": np.ascontiguousarray(price, dtype=np.float64),
        "qty": np.ascontiguousarray(qty, dtype=np.float64),
        "buyer_is_maker": np.ascontiguousarray(maker, dtype=bool),
    }


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    return int(float(text))


@functools.lru_cache(maxsize=8)
def _load_book_ticker_csv(path_str: str) -> dict[str, np.ndarray] | None:
    path = Path(path_str)
    if not path.is_file():
        return None
    cached = _try_npz(path)
    if cached is not None and "recv_ts_ms" in cached:
        return cached
    with path.open(newline="", encoding="utf-8") as handle:
        header = handle.readline()
    if not header.strip():
        return None
    names = [part.strip() for part in header.split(",")]
    idx = {name: i for i, name in enumerate(names)}
    required = ("recv_ts_ms", "bid_price", "bid_qty", "ask_price", "ask_qty")
    if any(name not in idx for name in required):
        return None
    cols = list(required)
    extra = [name for name in ("event_ts_ms", "transaction_ts_ms") if name in idx]
    usecols = [idx[name] for name in cols + extra]
    try:
        arr = np.loadtxt(path, delimiter=",", skiprows=1, usecols=usecols)
    except Exception:
        return None
    if arr.size == 0:
        out: dict[str, np.ndarray] = {
            "recv_ts_ms": np.array([], dtype=np.int64),
            "bid_price": np.array([], dtype=np.float64),
            "bid_qty": np.array([], dtype=np.float64),
            "ask_price": np.array([], dtype=np.float64),
            "ask_qty": np.array([], dtype=np.float64),
        }
        if extra:
            out["event_ts_ms"] = np.array([], dtype=np.int64)
            out["transaction_ts_ms"] = np.array([], dtype=np.int64)
        return out
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out = {
        "recv_ts_ms": arr[:, 0].astype(np.int64, copy=False),
        "bid_price": arr[:, 1].astype(np.float64, copy=False),
        "bid_qty": arr[:, 2].astype(np.float64, copy=False),
        "ask_price": arr[:, 3].astype(np.float64, copy=False),
        "ask_qty": arr[:, 4].astype(np.float64, copy=False),
    }
    for i, name in enumerate(extra, start=5):
        out[name] = arr[:, i].astype(np.int64, copy=False)
    _write_npz(path, out)
    return out


def _book_ticker_at(path: Path, current_time_ms: int, *, compact: bool, futures: bool) -> dict | None:
    table = _load_book_ticker_csv(str(path))
    if table is None:
        return None
    times = table["recv_ts_ms"]
    sl = _window_slice(times, current_time_ms - BOOK_TICKER_WINDOW_S * 1000, current_time_ms)
    recv = times[sl]
    if recv.size == 0:
        return None
    if compact and recv.size > 8:
        sl_last = slice(recv.size - 8, recv.size)
        recv = recv[sl_last]
        bid_p = table["bid_price"][sl][sl_last]
        bid_q = table["bid_qty"][sl][sl_last]
        ask_p = table["ask_price"][sl][sl_last]
        ask_q = table["ask_qty"][sl][sl_last]
        event = table["event_ts_ms"][sl][sl_last] if "event_ts_ms" in table else None
        tx = table["transaction_ts_ms"][sl][sl_last] if "transaction_ts_ms" in table else None
    else:
        bid_p = table["bid_price"][sl]
        bid_q = table["bid_qty"][sl]
        ask_p = table["ask_price"][sl]
        ask_q = table["ask_qty"][sl]
        event = table.get("event_ts_ms")
        if event is not None:
            event = event[sl]
        tx = table.get("transaction_ts_ms")
        if tx is not None:
            tx = tx[sl]
    out: dict[str, Any] = {
        "recv_ts_ms": np.ascontiguousarray(recv, dtype=np.int64),
        "bid_price": np.ascontiguousarray(bid_p, dtype=np.float64),
        "bid_qty": np.ascontiguousarray(bid_q, dtype=np.float64),
        "ask_price": np.ascontiguousarray(ask_p, dtype=np.float64),
        "ask_qty": np.ascontiguousarray(ask_q, dtype=np.float64),
    }
    if futures:
        out["event_ts_ms"] = (
            np.ascontiguousarray(event, dtype=np.int64)
            if event is not None
            else out["recv_ts_ms"].copy()
        )
        out["transaction_ts_ms"] = (
            np.ascontiguousarray(tx, dtype=np.int64) if tx is not None else out["recv_ts_ms"].copy()
        )
    return out


def _percent_levels_to_book(
    pct: np.ndarray, depth: np.ndarray, notional: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Turn Vision bookDepth (% from mid, cumulative qty) into best-first [price, qty]."""
    pct = np.asarray(pct, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    notional = np.asarray(notional, dtype=np.float64)
    bid_i = np.where(pct < 0.0)[0]
    ask_i = np.where(pct > 0.0)[0]
    if bid_i.size == 0 or ask_i.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    inner_bid = bid_i[int(np.argmin(np.abs(pct[bid_i])))]
    inner_ask = ask_i[int(np.argmin(np.abs(pct[ask_i])))]
    vwap_b = float(notional[inner_bid]) / max(float(depth[inner_bid]), 1e-12)
    vwap_a = float(notional[inner_ask]) / max(float(depth[inner_ask]), 1e-12)
    mid = 0.5 * (vwap_b + vwap_a)

    def side(indices: np.ndarray) -> np.ndarray:
        order = indices[np.argsort(np.abs(pct[indices]))]
        levels: list[list[float]] = []
        prev = 0.0
        for i in order:
            qty = max(float(depth[i]) - prev, 0.0)
            prev = float(depth[i])
            price = mid * (1.0 + float(pct[i]) / 100.0)
            levels.append([price, qty])
        if not levels:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray(levels, dtype=np.float64)

    return side(bid_i), side(ask_i)


@functools.lru_cache(maxsize=4)
def _load_book_depth_csv(path_str: str) -> dict[str, np.ndarray] | None:
    path = Path(path_str)
    if not path.is_file():
        return None
    cached = _try_npz(path)
    if cached is not None and "ts_ms" in cached:
        arr_ts = cached["ts_ms"].astype(np.int64, copy=False)
        percentage = cached["percentage"].astype(np.float64, copy=False)
        depth = cached["depth"].astype(np.float64, copy=False)
        notional = cached["notional"].astype(np.float64, copy=False)
    else:
        try:
            arr = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 2, 3, 4))
        except Exception:
            return None
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr_ts = arr[:, 0].astype(np.int64, copy=False)
        percentage = arr[:, 1].astype(np.float64, copy=False)
        depth = arr[:, 2].astype(np.float64, copy=False)
        notional = arr[:, 3].astype(np.float64, copy=False)
        _write_npz(
            path,
            {
                "ts_ms": arr_ts,
                "percentage": percentage,
                "depth": depth,
                "notional": notional,
            },
        )
    uniq, starts = np.unique(arr_ts, return_index=True)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = arr_ts.size
    return {
        "ts_ms": arr_ts,
        "percentage": percentage,
        "depth": depth,
        "notional": notional,
        "uniq_ts": uniq,
        "starts": starts,
        "ends": ends,
    }


def _snapshot_from_book_depth(table: dict[str, np.ndarray], t_ms: int, *, futures: bool) -> dict | None:
    uniq = table["uniq_ts"]
    i = int(np.searchsorted(uniq, t_ms, side="right") - 1)
    if i < 0:
        return None
    sl = slice(int(table["starts"][i]), int(table["ends"][i]))
    bids, asks = _percent_levels_to_book(
        table["percentage"][sl], table["depth"][sl], table["notional"][sl]
    )
    ts = int(uniq[i])
    return {
        "recv_ts_ms": np.int64(ts),
        "update_id": ts,
        "bids": bids,
        "asks": asks,
        "event_ts_ms": ts if futures else None,
        "transaction_ts_ms": ts if futures else None,
    }


def _encode_depth_snaps(snaps: list[dict]) -> dict[str, np.ndarray]:
    n = len(snaps)
    recv = np.empty(n, dtype=np.int64)
    uid = np.empty(n, dtype=np.int64)
    event = np.empty(n, dtype=np.int64)
    tx = np.empty(n, dtype=np.int64)
    bid_off = np.empty(n, dtype=np.int64)
    bid_n = np.empty(n, dtype=np.int32)
    ask_off = np.empty(n, dtype=np.int64)
    ask_n = np.empty(n, dtype=np.int32)
    bid_parts: list[np.ndarray] = []
    ask_parts: list[np.ndarray] = []
    bo = 0
    ao = 0
    for i, snap in enumerate(snaps):
        recv[i] = int(snap["recv_ts_ms"])
        uid[i] = int(snap["update_id"])
        event[i] = -1 if snap["event_ts_ms"] is None else int(snap["event_ts_ms"])
        tx[i] = -1 if snap["transaction_ts_ms"] is None else int(snap["transaction_ts_ms"])
        bids = np.asarray(snap["bids"], dtype=np.float64).reshape(-1, 2)
        asks = np.asarray(snap["asks"], dtype=np.float64).reshape(-1, 2)
        bid_off[i] = bo
        bid_n[i] = bids.shape[0]
        ask_off[i] = ao
        ask_n[i] = asks.shape[0]
        if bids.size:
            bid_parts.append(bids)
        if asks.size:
            ask_parts.append(asks)
        bo += bids.shape[0]
        ao += asks.shape[0]
    return {
        "recv_ts_ms": recv,
        "update_id": uid,
        "event_ts_ms": event,
        "transaction_ts_ms": tx,
        "bid_off": bid_off,
        "bid_n": bid_n,
        "ask_off": ask_off,
        "ask_n": ask_n,
        "bids": np.vstack(bid_parts) if bid_parts else np.zeros((0, 2), dtype=np.float64),
        "asks": np.vstack(ask_parts) if ask_parts else np.zeros((0, 2), dtype=np.float64),
    }


def _decode_depth_snaps(cached: dict[str, np.ndarray]) -> list[dict]:
    recv = cached["recv_ts_ms"]
    bids_all = cached["bids"]
    asks_all = cached["asks"]
    snaps: list[dict] = []
    for i in range(recv.size):
        b0 = int(cached["bid_off"][i])
        bn = int(cached["bid_n"][i])
        a0 = int(cached["ask_off"][i])
        an = int(cached["ask_n"][i])
        bids = bids_all[b0 : b0 + bn] if bn else np.zeros((0, 2), dtype=np.float64)
        asks = asks_all[a0 : a0 + an] if an else np.zeros((0, 2), dtype=np.float64)
        ev = int(cached["event_ts_ms"][i])
        tx = int(cached["transaction_ts_ms"][i])
        snaps.append(
            {
                "recv_ts_ms": np.int64(recv[i]),
                "update_id": int(cached["update_id"][i]),
                "bids": bids,
                "asks": asks,
                "event_ts_ms": None if ev < 0 else ev,
                "transaction_ts_ms": None if tx < 0 else tx,
            }
        )
    return snaps


def _parse_depth_snapshot_csv(path: Path) -> list[dict] | None:
    recv_l: list[int] = []
    uid_l: list[int] = []
    event_l: list[int | None] = []
    tx_l: list[int | None] = []
    is_bid_l: list[bool] = []
    level_l: list[int] = []
    price_l: list[float] = []
    qty_l: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            return None
        for row in reader:
            if len(row) < 9:
                continue
            recv_l.append(int(row[0]))
            uid_l.append(int(float(row[2] or 0)))
            event_l.append(_opt_int(row[3]))
            tx_l.append(_opt_int(row[4]))
            is_bid_l.append(str(row[5]).lower().startswith("b"))
            level_l.append(int(row[6]))
            price_l.append(float(row[7]))
            qty_l.append(float(row[8]))
    if not recv_l:
        return None
    recv = np.asarray(recv_l, dtype=np.int64)
    bounds = np.concatenate(([0], np.flatnonzero(recv[1:] != recv[:-1]) + 1, [recv.size]))
    snaps: list[dict] = []
    for i in range(bounds.size - 1):
        lo = int(bounds[i])
        hi = int(bounds[i + 1])
        bids: list[tuple[int, float, float]] = []
        asks: list[tuple[int, float, float]] = []
        for j in range(lo, hi):
            item = (level_l[j], price_l[j], qty_l[j])
            if is_bid_l[j]:
                bids.append(item)
            else:
                asks.append(item)
        bid_arr = (
            np.array([[p, q] for _, p, q in sorted(bids)], dtype=np.float64)
            if bids
            else np.zeros((0, 2), dtype=np.float64)
        )
        ask_arr = (
            np.array([[p, q] for _, p, q in sorted(asks)], dtype=np.float64)
            if asks
            else np.zeros((0, 2), dtype=np.float64)
        )
        snaps.append(
            {
                "recv_ts_ms": np.int64(recv[lo]),
                "update_id": int(uid_l[lo]),
                "bids": bid_arr,
                "asks": ask_arr,
                "event_ts_ms": event_l[lo],
                "transaction_ts_ms": tx_l[lo],
            }
        )
    return snaps


@functools.lru_cache(maxsize=4)
def _load_depth_snapshot_csv(path_str: str) -> list[dict] | None:
    path = Path(path_str)
    if not path.is_file():
        return None
    cached = _try_npz(path)
    if cached is not None and "recv_ts_ms" in cached and "bid_off" in cached:
        return _decode_depth_snaps(cached)
    snaps = _parse_depth_snapshot_csv(path)
    if snaps:
        _write_npz(path, _encode_depth_snaps(snaps))
    return snaps


def _last_snapshot_at(snaps: list[dict] | None, t_ms: int) -> dict | None:
    if not snaps:
        return None
    chosen: dict | None = None
    for snap in snaps:
        if int(snap["recv_ts_ms"]) <= t_ms:
            chosen = snap
        else:
            break
    return chosen


@functools.lru_cache(maxsize=4)
def _load_depth_updates_jsonl(path_str: str) -> tuple[np.ndarray, tuple] | None:
    path = Path(path_str)
    if not path.is_file():
        return None
    recs: list[dict] = []
    times: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            times.append(int(rec["recv_ts_ms"]))
            recs.append(rec)
    if not times:
        return None
    times_arr = np.asarray(times, dtype=np.int64)
    order = np.argsort(times_arr, kind="mergesort")
    return times_arr[order], tuple(recs[int(i)] for i in order)


def _materialize_depth_update(rec: dict, *, futures: bool) -> dict:
    bids = np.asarray(rec.get("bids") or [], dtype=np.float64)
    asks = np.asarray(rec.get("asks") or [], dtype=np.float64)
    if bids.ndim != 2:
        bids = bids.reshape(-1, 2) if bids.size else np.zeros((0, 2), dtype=np.float64)
    if asks.ndim != 2:
        asks = asks.reshape(-1, 2) if asks.size else np.zeros((0, 2), dtype=np.float64)
    out: dict[str, Any] = {
        "recv_ts_ms": np.int64(rec["recv_ts_ms"]),
        "event_ts_ms": np.int64(rec.get("event_ts_ms", rec["recv_ts_ms"])),
        "first_id": int(rec["first_id"]),
        "final_id": int(rec["final_id"]),
        "prev_final_id": int(rec["prev_final_id"]),
        "bids": bids,
        "asks": asks,
    }
    if futures:
        tx = rec.get("transaction_ts_ms", rec.get("event_ts_ms", rec["recv_ts_ms"]))
        out["transaction_ts_ms"] = np.int64(tx)
    return out


def _depth_updates_at(path: Path, t0: int, current_time_ms: int, *, compact: bool, futures: bool) -> list[dict] | None:
    loaded = _load_depth_updates_jsonl(str(path))
    if loaded is None:
        return None
    times, recs = loaded
    sl = _window_slice(times, t0, current_time_ms)
    chosen = list(recs[sl])
    if not chosen:
        return []
    if compact and len(chosen) > 2:
        chosen = chosen[-2:]
    return [_materialize_depth_update(rec, futures=futures) for rec in chosen]


def _empty_depth_snapshot(t_ms: int, *, futures: bool) -> dict:
    return {
        "recv_ts_ms": np.int64(t_ms),
        "update_id": 0,
        "bids": np.zeros((0, 2), dtype=np.float64),
        "asks": np.zeros((0, 2), dtype=np.float64),
        "event_ts_ms": int(t_ms - 1) if futures else None,
        "transaction_ts_ms": int(t_ms - 2) if futures else None,
    }


def _empty_book_ticker(*, futures: bool) -> dict:
    recv = np.array([], dtype=np.int64)
    out: dict[str, Any] = {
        "recv_ts_ms": recv,
        "bid_price": np.array([], dtype=np.float64),
        "bid_qty": np.array([], dtype=np.float64),
        "ask_price": np.array([], dtype=np.float64),
        "ask_qty": np.array([], dtype=np.float64),
    }
    if futures:
        out["event_ts_ms"] = recv.copy()
        out["transaction_ts_ms"] = recv.copy()
    return out


def _empty_candles() -> dict:
    return {
        "open_time_ms": np.array([], dtype=np.int64),
        "ohlcv": np.zeros((0, 5), dtype=np.float64),
        "complete_history": False,
    }


def _empty_trades_full() -> dict:
    empty_i = np.array([], dtype=np.int64)
    return {
        "ts_ms": empty_i,
        "event_ts_ms": empty_i.copy(),
        "recv_ts_ms": empty_i.copy(),
        "price": np.array([], dtype=np.float64),
        "qty": np.array([], dtype=np.float64),
        "buyer_is_maker": np.array([], dtype=bool),
    }


def _depth_bundle_at(
    *,
    current_time_ms: int,
    t0: int,
    px: float,
    rng: np.random.Generator,
    futures: bool,
    compact: bool,
    n_upd: int,
    synthetic: bool = False,
    fill_missing: bool = True,
) -> tuple[dict, dict, list[dict]]:
    if synthetic:
        start = _depth_snapshot(t0, px, rng, futures=futures)
        latest = _depth_snapshot(current_time_ms, px, rng, futures=futures)
        updates = _depth_updates(n_upd, t0, px, rng, futures=futures, start_id=start["update_id"])
        return start, latest, updates
    snap_path = FUTURES_DEPTH_CSV if futures else SPOT_DEPTH_CSV
    upd_path = FUTURES_DEPTH_UPDATES_JSONL if futures else SPOT_DEPTH_UPDATES_JSONL
    snaps = _load_depth_snapshot_csv(str(snap_path))
    start = _last_snapshot_at(snaps, t0)
    latest = _last_snapshot_at(snaps, current_time_ms)
    if futures:
        book_depth = _load_book_depth_csv(str(FUTURES_BOOK_DEPTH_CSV))
        if book_depth is not None:
            if start is None:
                start = _snapshot_from_book_depth(book_depth, t0, futures=True)
            if latest is None:
                latest = _snapshot_from_book_depth(book_depth, current_time_ms, futures=True)
    updates = _depth_updates_at(upd_path, t0, current_time_ms, compact=compact, futures=futures)
    real_start = start is not None
    real_latest = latest is not None
    if start is None:
        start = (
            _depth_snapshot(t0, px, rng, futures=futures)
            if fill_missing
            else _empty_depth_snapshot(t0, futures=futures)
        )
    if latest is None:
        latest = (
            _depth_snapshot(current_time_ms, px, rng, futures=futures)
            if fill_missing
            else _empty_depth_snapshot(current_time_ms, futures=futures)
        )
    if fill_missing and not updates and not real_start and not real_latest:
        updates = _depth_updates(
            n_upd, t0, px, rng, futures=futures, start_id=start["update_id"]
        )
    elif updates is None:
        updates = []
    return start, latest, updates


def default_current_time_ms() -> int:
    """Latest candle open time present in both venue CSVs, else a dummy timestamp."""
    times: list[int] = []
    for path in (SPOT_CANDLES_CSV, FUTURES_CANDLES_CSV):
        table = _load_candles_csv(str(path))
        if table is not None and table["open_time_ms"].size:
            times.append(int(table["open_time_ms"][-1]))
    if times:
        return min(times)
    return 1_700_000_000_000


def preload_venue_csvs() -> None:
    """Read candle, trade, book, and depth CSVs into the loader cache."""
    import time

    jobs: list[tuple[str, Path, Any]] = [
        ("btc_spot_candles.csv", SPOT_CANDLES_CSV, _load_candles_csv),
        ("btc_futures_candles.csv", FUTURES_CANDLES_CSV, _load_candles_csv),
        ("btc_spot_trades.csv", SPOT_TRADES_CSV, _load_trades_csv),
        ("btc_futures_trades.csv", FUTURES_TRADES_CSV, _load_trades_csv),
        ("btc_spot_book_ticker.csv", SPOT_BOOK_TICKER_CSV, _load_book_ticker_csv),
        ("btc_futures_book_ticker.csv", FUTURES_BOOK_TICKER_CSV, _load_book_ticker_csv),
        ("btc_spot_depth.csv", SPOT_DEPTH_CSV, _load_depth_snapshot_csv),
        ("btc_futures_depth.csv", FUTURES_DEPTH_CSV, _load_depth_snapshot_csv),
        ("btc_futures_book_depth.csv", FUTURES_BOOK_DEPTH_CSV, _load_book_depth_csv),
        ("btc_spot_depth_updates.jsonl", SPOT_DEPTH_UPDATES_JSONL, _load_depth_updates_jsonl),
        ("btc_futures_depth_updates.jsonl", FUTURES_DEPTH_UPDATES_JSONL, _load_depth_updates_jsonl),
    ]
    total = 0.0
    print("database load")
    for name, path, loader in jobs:
        t0 = time.perf_counter()
        loader(str(path))
        dt = time.perf_counter() - t0
        total += dt
        mb = path.stat().st_size / 1e6 if path.is_file() else 0.0
        print(f"  {name:36s} {mb:7.1f} MB  {dt:7.3f} s")
    print(f"  {'TOTAL':36s} {'':7s}     {total:7.3f} s")


def _venue(
    *,
    current_time_ms: int,
    t0: int,
    px: float,
    rng: np.random.Generator,
    futures: bool,
    complete_candles: bool,
    compact: bool = False,
    synthetic: bool = False,
    fill_missing: bool = True,
) -> dict:
    n_trades = 6 if compact else (400 if futures else 350)
    n_bt = 8 if compact else (800 if futures else 700)
    n_upd = 2 if compact else 40
    n_candles = 8 if compact else CANDLE_WINDOW_S
    candles_path = FUTURES_CANDLES_CSV if futures else SPOT_CANDLES_CSV
    trades_path = FUTURES_TRADES_CSV if futures else SPOT_TRADES_CSV
    book_path = FUTURES_BOOK_TICKER_CSV if futures else SPOT_BOOK_TICKER_CSV
    if synthetic:
        candles = None
        trades = None
        book_ticker = None
    else:
        candles = _candles_at(candles_path, current_time_ms, compact=compact)
        trades = _trades_at(trades_path, current_time_ms, compact=compact)
        book_ticker = _book_ticker_at(book_path, current_time_ms, compact=compact, futures=futures)
    if candles is None:
        candles = (
            _candles(
                n_candles,
                current_time_ms - n_candles * 1000,
                px,
                rng,
                complete_candles,
                compact=compact,
            )
            if fill_missing or synthetic
            else _empty_candles()
        )
    if trades is None:
        trades = _trades(n_trades, t0, px, rng) if fill_missing or synthetic else _empty_trades_full()
    if book_ticker is None:
        book_ticker = (
            _book_ticker(n_bt, t0, px, rng, futures=futures)
            if fill_missing or synthetic
            else _empty_book_ticker(futures=futures)
        )
    if candles["ohlcv"].size:
        px = float(candles["ohlcv"][-1, 3])
    start, latest, updates = _depth_bundle_at(
        current_time_ms=current_time_ms,
        t0=t0,
        px=px,
        rng=rng,
        futures=futures,
        compact=compact,
        n_upd=n_upd,
        synthetic=synthetic,
        fill_missing=fill_missing or synthetic,
    )
    last_trade = (
        int(trades["recv_ts_ms"][-1]) if trades["recv_ts_ms"].size else current_time_ms
    )
    last_kline = int(candles["open_time_ms"][-1]) if candles["open_time_ms"].size else current_time_ms
    last_book = (
        int(book_ticker["recv_ts_ms"][-1]) if book_ticker["recv_ts_ms"].size else current_time_ms
    )
    last_event_times = {
        "trade": np.int64(last_trade),
        "depth": np.int64(latest["recv_ts_ms"]),
        "bookTicker": np.int64(last_book),
        "kline_1s": np.int64(last_kline),
    }
    return {
        "symbol": "BTCUSDT",
        "candles_1s": candles,
        "trades": trades,
        "book_ticker": book_ticker,
        "depth_start": start,
        "depth_updates": updates,
        "depth_latest": latest,
        "last_event_times": last_event_times,
    }


def make_sample_payload(
    seed: int = 0,
    *,
    price: float = 97_500.0,
    current_time_ms: int | None = None,
    compact: bool = False,
    synthetic: bool = False,
    fill_missing: bool = True,
) -> dict:
    """Build a schema_version-3 payload with NumPy arrays, oldest-first.

    ``current_time_ms`` is the forecast anchor. Candles are the trailing hour
    and trades / book ticker the trailing 60s from the venue CSVs. Depth uses
    REST snapshots, futures bookDepth, or live diffs when those files exist.
    ``compact=True`` keeps the same keys and dtypes but short arrays, so the
    file is small enough to inspect before a full-size latency test.
    ``synthetic=True`` ignores CSVs and builds a shape-faithful dummy (input.md).
    ``fill_missing=False`` never invents streams — used by the local backtest.
    """
    rng = np.random.default_rng(seed)
    now = int(current_time_ms if current_time_ms is not None else default_current_time_ms())
    t0 = now - TRADE_WINDOW_S * 1000
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt": {
            "asset": "BTC",
            "horizon_seconds": HORIZON_SECONDS,
            "num_percentiles": NUM_PERCENTILES,
            "quantile_grid": "centered-100",
            "current_time_ms": now,
            "trigger": {"kind": "interval", "venue": None},
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
                synthetic=synthetic,
                fill_missing=fill_missing,
            ),
            "futures": _venue(
                current_time_ms=now,
                t0=t0,
                px=price + 8.0,
                rng=rng,
                futures=True,
                complete_candles=False,
                compact=compact,
                synthetic=synthetic,
                fill_missing=fill_missing,
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
    out: dict[str, Any] = {
        "recv_ts_ms": np.int64(upd["recv_ts_ms"]),
        "event_ts_ms": np.int64(upd["event_ts_ms"]),
        "first_id": int(upd["first_id"]),
        "final_id": int(upd["final_id"]),
        "prev_final_id": int(upd["prev_final_id"]),
        "bids": np.ascontiguousarray(bids, dtype=np.float64),
        "asks": np.ascontiguousarray(asks, dtype=np.float64),
    }
    if futures:
        out["transaction_ts_ms"] = np.int64(upd["transaction_ts_ms"])
    return out


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


def assert_payload_schema(payload: dict) -> None:
    """Fail if the payload does not match input.md schema_version 3."""
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
    if trigger.get("kind") not in ("interval", "trade", "book"):
        raise ValueError("prompt.trigger.kind")
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
            if futures:
                if "transaction_ts_ms" not in upd:
                    raise ValueError("futures.depth_updates.transaction_ts_ms")
            elif "transaction_ts_ms" in upd:
                raise ValueError("spot.depth_updates must omit T")
        last = venue.get("last_event_times") or {}
        for stream in ("trade", "depth", "bookTicker", "kline_1s"):
            if stream not in last:
                raise ValueError(f"{name}.last_event_times.{stream}")


def write_sample_payload(path: Path | None = None, **kwargs: Any) -> Path:
    """Write a shape-faithful compact sample matching input.md."""
    kwargs.setdefault("seed", 0)
    kwargs.setdefault("compact", True)
    kwargs.setdefault("synthetic", True)
    kwargs.setdefault("current_time_ms", 1_700_000_000_000)
    payload = make_sample_payload(**kwargs)
    assert_payload_schema(payload)
    target = Path(path) if path is not None else SAMPLE_PAYLOAD_JSON
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload_to_jsonable(payload), indent=2), encoding="utf-8")
    return target
