"""Shape-faithful sample payload for local validation (input.md schema_version 3)."""

from __future__ import annotations

import functools
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
SPOT_CANDLES_CSV = REPO_ROOT / "btc_spot_candles.csv"
FUTURES_CANDLES_CSV = REPO_ROOT / "btc_futures_candles.csv"
SPOT_TRADES_CSV = REPO_ROOT / "btc_spot_trades.csv"
FUTURES_TRADES_CSV = REPO_ROOT / "btc_futures_trades.csv"


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


@functools.lru_cache(maxsize=8)
def _load_candles_csv(path_str: str) -> dict[str, np.ndarray] | None:
    path = Path(path_str)
    if not path.is_file():
        return None
    # open_time_ms, open_time_utc, open, high, low, close, volume
    arr = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 2, 3, 4, 5, 6))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return {
        "open_time_ms": arr[:, 0].astype(np.int64, copy=False),
        "ohlcv": arr[:, 1:6].astype(np.float64, copy=False),
    }


@functools.lru_cache(maxsize=8)
def _load_trades_csv(path_str: str) -> dict[str, np.ndarray] | None:
    path = Path(path_str)
    if not path.is_file():
        return None
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if raw.size == 0:
        return {
            "ts_ms": np.array([], dtype=np.int64),
            "price": np.array([], dtype=np.float64),
            "qty": np.array([], dtype=np.float64),
            "buyer_is_maker": np.array([], dtype=bool),
        }
    maker = raw["buyer_is_maker"]
    if maker.dtype.kind in ("U", "S", "O"):
        maker_bool = np.array(
            [str(x).strip().lower() in ("true", "1") for x in maker], dtype=bool
        )
    else:
        maker_bool = np.asarray(maker, dtype=bool)
    return {
        "ts_ms": np.asarray(raw["ts_ms"], dtype=np.int64),
        "price": np.asarray(raw["price"], dtype=np.float64),
        "qty": np.asarray(raw["qty"], dtype=np.float64),
        "buyer_is_maker": maker_bool,
    }


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
    """Read candle and trade CSVs into the loader cache."""
    for path in (SPOT_CANDLES_CSV, FUTURES_CANDLES_CSV):
        _load_candles_csv(str(path))
    for path in (SPOT_TRADES_CSV, FUTURES_TRADES_CSV):
        _load_trades_csv(str(path))


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
    n_candles = 8 if compact else CANDLE_WINDOW_S
    candles_path = FUTURES_CANDLES_CSV if futures else SPOT_CANDLES_CSV
    trades_path = FUTURES_TRADES_CSV if futures else SPOT_TRADES_CSV
    candles = _candles_at(candles_path, current_time_ms, compact=compact)
    trades = _trades_at(trades_path, current_time_ms, compact=compact)
    if candles is None:
        candles = _candles(
            n_candles,
            current_time_ms - n_candles * 1000,
            px,
            rng,
            complete_candles,
            compact=compact,
        )
    if trades is None:
        trades = _trades(n_trades, t0, px, rng)
    if candles["ohlcv"].size:
        px = float(candles["ohlcv"][-1, 3])
    start = _depth_snapshot(t0, px, rng, futures=futures)
    latest = _depth_snapshot(current_time_ms, px, rng, futures=futures)
    last_trade = int(trades["ts_ms"][-1]) if trades["ts_ms"].size else current_time_ms
    last_kline = int(candles["open_time_ms"][-1]) if candles["open_time_ms"].size else current_time_ms
    last_event_times = {
        "trade": np.int64(last_trade),
        "depth": np.int64(current_time_ms - 1),
        "bookTicker": np.int64(current_time_ms - 2),
        "kline_1s": np.int64(last_kline),
    }
    return {
        "symbol": "BTCUSDT",
        "candles_1s": candles,
        "trades": trades,
        "book_ticker": _book_ticker(n_bt, t0, px, rng, futures=futures),
        "depth_start": start,
        "depth_updates": _depth_updates(
            n_upd, t0, px, rng, futures=futures, start_id=start["update_id"]
        ),
        "depth_latest": latest,
        "last_event_times": last_event_times,
    }


def make_sample_payload(
    seed: int = 0,
    *,
    price: float = 97_500.0,
    current_time_ms: int | None = None,
    compact: bool = False,
) -> dict:
    """Build a schema_version-3 payload with NumPy arrays, oldest-first.

    ``current_time_ms`` is the forecast anchor. Candles are the trailing hour
    and trades the trailing 60s from the venue CSVs.
    ``compact=True`` keeps the same keys and dtypes but short arrays, so the
    file is small enough to inspect before a full-size latency test.
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
