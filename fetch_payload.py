"""Capture backtest pairs the way the evaluator calls the model.

    python fetch_payload.py
    python fetch_payload.py --asset ETH --interval 20
    python fetch_payload.py --asset SOL --hours 48
    python fetch_payload.py --interval 20

The default clock is 200 seconds: every tenth live cycle. The evaluator clock
is 20 seconds. BTC is on ``:00``, ``:20``, and ``:40``; ETH, SOL, and XRP are
5, 10, and 15 seconds later. ``--interval 20`` keeps that schedule. A wider
multiple (200, 400, ...) drops the cycles in between, so a test run writes
that many times fewer pairs.

``current_time_ms`` for a time prompt is the asset's slot. A qualifying price
move is taken with a 50% chance inside the 4-second window that opens 10.25
seconds after that slot; a miss waits for the next move in the window. The
window is not stretched, so the mix of clock prompts and event prompts stays
the live mix. Of the moves that are taken, half are ``event`` (no delay) and
half are ``event_delayed`` by ``uniform(0, 500 ms)``. The anchor stays the
move's receive time either way (FAQ.md).

Each prompt is written under ``database/{asset}/{current_time_ms}/``:

    payload_{current_time_ms}.json
    spot_book_ticker.json

The book file is the last spot bookTicker in the following 10 seconds.
At the start of a run, capture folders older than 7 days are deleted.
"""

from __future__ import annotations

import argparse
import json
import random
import select
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fetch_live import (
    HORIZON_SECONDS,
    RAW_STREAMS,
    TRADE_WINDOW_S,
    CANDLE_WINDOW_S,
    LiveCapture,
    VenueBuf,
    _drain_socket,
    _http_get_json,
    _now_ms,
    _ws_connect,
    assemble_payload,
    assert_env_payload_schema,
    rest_depth_snapshot,
    rest_futures_candles_1s,
    rest_spot_candles_1s,
    rest_spot_depth_bands,
    save_payload,
    use_asset,
)
from harness.constants import (
    ASSETS,
    CLOCK_INTERVAL_MS,
    EVENT_DELAY_MAX_MS,
    EVENT_WINDOW_LEN_MS,
    EVENT_WINDOW_OPEN_MS,
    asset_spec,
)
from harness.payload import empty_depth_bands, history_is_complete
from harness.saved_pairs import book_path, payload_path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = REPO_ROOT / "database"
KEEP_CAPTURE_MS = 7 * 24 * 60 * 60 * 1000
DEPTH_SAMPLE_S = 1.0
BOOK_KEEP_MS = 70_000
LIQUIDATION_KEEP_MS = (CANDLE_WINDOW_S + 100) * 1000


def interval_to_ms(seconds: float) -> int:
    """``57.171`` seconds -> ``57171`` milliseconds."""
    ms = int(round(float(seconds) * 1000.0))
    if ms <= 0:
        raise ValueError("interval must be > 0")
    return ms


def _asset_name(symbol: str) -> str:
    for name, spec in ASSETS.items():
        if spec.symbol == symbol:
            return name
    return "BTC"


def clock_interval_ms(interval_s: float) -> int:
    """Clock period for this capture. A whole multiple of the live 20s clock.

    ``20`` is the evaluator schedule. ``200`` keeps every tenth live cycle:
    one clock prompt and that cycle's 4-second event window, then nine cycles
    skipped. The window stays the live slice (opens 10.25s after the asset
    slot, lasts 4s).
    """
    clock_ms = interval_to_ms(interval_s)
    live = CLOCK_INTERVAL_MS
    if clock_ms % live != 0:
        raise ValueError(
            "interval must be a whole multiple of the live "
            f"{live // 1000}s clock (20 is live, 200 keeps every tenth cycle)"
        )
    return clock_ms


def delete_captures_older_than(
    database: Path,
    now_ms: int,
    *,
    max_age_ms: int = KEEP_CAPTURE_MS,
) -> int:
    """Delete ``database/{current_time_ms}`` folders older than ``max_age_ms``.

    The cutoff is ``now_ms`` at the start of the run. A folder exactly one
    week old is kept. Anything that is not a timestamp directory is left alone.
    """
    if not database.is_dir():
        return 0
    cutoff = int(now_ms) - int(max_age_ms)
    removed = 0
    for path in list(database.iterdir()):
        if not path.is_dir() or not path.name.isdigit():
            continue
        if int(path.name) >= cutoff:
            continue
        shutil.rmtree(path)
        removed += 1
    return removed


def ms_to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat(
        sep=" ", timespec="milliseconds"
    )


def moved_more_than_one_tick(prev: float, price: float, tick: float) -> bool:
    """FAQ.md: a trade group that shifts price by more than one tick."""
    steps = abs(float(price) - float(prev)) / float(tick)
    return steps > 1.0 + 1e-6


def event_window_boundary(
    recv_ms: int,
    *,
    clock_ms: int = CLOCK_INTERVAL_MS,
    offset_ms: int = 0,
) -> int | None:
    """Asset slot whose event window contains ``recv_ms``.

    The window opens 10.25s after the asset's slot and closes 4s later.
    ``offset_ms`` is that slot's stagger (ETH 5s, SOL 10s, XRP 15s). A move
    before the window, or after it has closed, is not a prompt. With a 200s
    clock that is every tenth live slot, and the nine slots in between are ignored.
    """
    shifted = int(recv_ms) - int(offset_ms)
    clock_ms = int(clock_ms)
    boundary = shifted - (shifted % clock_ms)
    open_ms = boundary + EVENT_WINDOW_OPEN_MS
    if open_ms <= shifted < open_ms + EVENT_WINDOW_LEN_MS:
        return boundary + int(offset_ms)
    return None


def next_time_boundary(
    now_ms: int,
    *,
    clock_ms: int = CLOCK_INTERVAL_MS,
    offset_ms: int = 0,
) -> int:
    """Next asset slot at or after ``now_ms``.

    BTC slots are wall-clock ``:00`` / ``:20`` / ``:40``. The other assets use
    the same 20s grid plus ``offset_ms``. A longer clock is every Nth slot.
    """
    shifted = int(now_ms) - int(offset_ms)
    clock_ms = int(clock_ms)
    boundary = shifted - (shifted % clock_ms)
    if boundary != shifted:
        boundary += clock_ms
    return boundary + int(offset_ms)


def take_qualifying_move(rng: random.Random, boundary: int, taken: set[int]) -> bool:
    """Take this move with a 50% chance. One taken move closes the window."""
    if boundary in taken:
        return False
    if rng.random() >= 0.5:
        return False
    taken.add(boundary)
    return True


def event_delay_ms(rng: random.Random) -> int:
    """``uniform(0, 500 ms)`` for the delayed half of move prompts."""
    return int(rng.random() * EVENT_DELAY_MAX_MS)


class TradeGroupDetector:
    """Group trades that share an exchange timestamp, then compare group prices."""

    def __init__(self, tick: float) -> None:
        self.tick = float(tick)
        self._prev: float | None = None
        self._group_ts: int | None = None
        self._group_px: float | None = None
        self._group_recv: int | None = None

    def push(self, ts_ms: int, price: float, recv_ms: int) -> int | None:
        """Return the closed group's receive time when that group moved the price."""
        ts_ms = int(ts_ms)
        price = float(price)
        recv_ms = int(recv_ms)
        if self._group_ts is None:
            self._group_ts = ts_ms
            self._group_px = price
            self._group_recv = recv_ms
            return None
        if ts_ms == self._group_ts:
            self._group_px = price
            self._group_recv = recv_ms
            return None
        prev = self._prev
        group_px = float(self._group_px if self._group_px is not None else price)
        group_recv = int(self._group_recv if self._group_recv is not None else recv_ms)
        self._prev = group_px
        self._group_ts = ts_ms
        self._group_px = price
        self._group_recv = recv_ms
        if prev is None:
            return None
        if moved_more_than_one_tick(prev, group_px, self.tick):
            return group_recv
        return None


# 1s sampler. A start more than one missed sample off the 60s mark is not usable.
DEPTH_START_TOLERANCE_MS = 1_500


def select_depth_pair(snaps: list[dict], anchor_ms: int) -> tuple[dict, dict]:
    """``depth_latest`` at or before the anchor; ``depth_start`` closest to 60s earlier."""
    latest = None
    start = None
    start_dist: int | None = None
    start_cut = int(anchor_ms) - TRADE_WINDOW_S * 1000
    for snap in snaps:
        recv = int(snap["recv_ms"])
        if recv > int(anchor_ms):
            continue
        if latest is None or recv >= int(latest["recv_ms"]):
            latest = snap
        dist = abs(recv - start_cut)
        if (
            start is None
            or start_dist is None
            or dist < start_dist
            or (dist == start_dist and recv < int(start["recv_ms"]))
        ):
            start = snap
            start_dist = dist
    if latest is None or start is None or start_dist is None:
        raise RuntimeError("no depth snapshot at or before current_time_ms")
    if start_dist > DEPTH_START_TOLERANCE_MS:
        raise RuntimeError(f"depth_start is {start_dist} ms from the 60s mark")
    return start, latest


def fetch_tick_sizes(symbol: str) -> dict[str, float]:
    """Price tick for the captured symbol. Falls back to the asset table."""
    spec = asset_spec(_asset_name(symbol))
    sizes = {"spot": spec.tick, "futures": spec.futures_tick}
    endpoints = (
        ("spot", "https://api.binance.com/api/v3/exchangeInfo"),
        ("futures", "https://fapi.binance.com/fapi/v1/exchangeInfo"),
    )
    for venue, url in endpoints:
        try:
            payload = _http_get_json(url, {"symbol": symbol})
        except Exception as exc:
            print(f"  tick size {venue} fallback {sizes[venue]} ({exc})", flush=True)
            continue
        symbols = payload.get("symbols") or []
        info = symbols[0] if symbols else {}
        for filt in info.get("filters") or []:
            if filt.get("filterType") == "PRICE_FILTER" and filt.get("tickSize") is not None:
                sizes[venue] = float(filt["tickSize"])
    return sizes


def last_spot_book_in_window(rows: list[dict], start_ms: int, end_ms: int) -> dict | None:
    """Last spot book row with ``start_ms < recv_ts_ms <= end_ms``."""
    found: dict | None = None
    found_recv = start_ms
    for row in rows:
        recv = int(row["recv_ts_ms"])
        if recv <= start_ms or recv > end_ms:
            continue
        if found is None or recv >= found_recv:
            found = row
            found_recv = recv
    return found


class CandleBook:
    """Rolling 1-second OHLCV used to fill each payload's trailing hour."""

    def __init__(self) -> None:
        self.rows: dict[int, list[float]] = {}

    def upsert(self, open_ms: int, o: float, h: float, l: float, c: float, v: float) -> None:
        self.rows[int(open_ms)] = [float(o), float(h), float(l), float(c), float(v)]

    def seed(self, candles: dict) -> None:
        times = candles.get("open_time_ms")
        ohlcv = candles.get("ohlcv")
        if times is None or ohlcv is None:
            return
        for i, open_ms in enumerate(times):
            bar = ohlcv[i]
            self.upsert(int(open_ms), bar[0], bar[1], bar[2], bar[3], bar[4])

    def add_trade(self, ts_ms: int, price: float, qty: float, min_ts: int) -> None:
        if int(ts_ms) < int(min_ts):
            return
        open_ms = (int(ts_ms) // 1000) * 1000
        px = float(price)
        q = float(qty)
        row = self.rows.get(open_ms)
        if row is None:
            self.rows[open_ms] = [px, px, px, px, q]
            return
        row[1] = max(row[1], px)
        row[2] = min(row[2], px)
        row[3] = px
        row[4] += q

    def window(self, end_ms: int) -> dict:
        last_open = ((int(end_ms) - 1) // 1000) * 1000
        first_open = last_open - (CANDLE_WINDOW_S - 1) * 1000
        stale = [key for key in self.rows if key < first_open - 60_000]
        for key in stale:
            del self.rows[key]
        earlier = [key for key in self.rows if key < first_open]
        prev = self.rows[max(earlier)][3] if earlier else None
        opens: list[int] = []
        bars: list[list[float]] = []
        for i in range(CANDLE_WINDOW_S):
            open_ms = first_open + i * 1000
            row = self.rows.get(open_ms)
            if row is None:
                if prev is None:
                    continue
                row = [prev, prev, prev, prev, 0.0]
            else:
                prev = row[3]
            opens.append(open_ms)
            bars.append(row)
        open_arr = np.asarray(opens, dtype=np.int64)
        ohlcv = (
            np.asarray(bars, dtype=np.float64)
            if bars
            else np.zeros((0, 5), dtype=np.float64)
        )
        return {
            "open_time_ms": open_arr,
            "ohlcv": ohlcv if ohlcv.ndim == 2 else ohlcv.reshape(-1, 5),
            "complete_history": bool(history_is_complete(open_arr)),
        }


def _prune_prefix(rows: list[dict], cutoff_ms: int, key: str = "recv_ts_ms") -> int:
    index = 0
    while index < len(rows) and int(rows[index][key]) < cutoff_ms:
        index += 1
    if index:
        del rows[:index]
    return index


@dataclass
class _Pending:
    anchor_ms: int
    folder: Path
    deadline_ms: int


@dataclass
class _ScheduledEvent:
    anchor_ms: int
    kind: str
    venue: str
    fire_at_ms: int


@dataclass
class _Writer:
    thread: threading.Thread | None = None
    errors: list[BaseException] = field(default_factory=list)

    def submit(self, payload: dict, path: Path) -> None:
        if self.thread is not None and self.thread.is_alive():
            self.thread.join()
        self.thread = threading.Thread(
            target=self._write,
            args=(payload, path),
            name="payload-writer",
            daemon=True,
        )
        self.thread.start()

    def _write(self, payload: dict, path: Path) -> None:
        try:
            save_payload(payload, path, pretty=False)
            print(
                f"  wrote {path}  ({path.stat().st_size} bytes)",
                flush=True,
            )
        except Exception as exc:
            self.errors.append(exc)
            print(f"  payload write failed: {exc}", file=sys.stderr, flush=True)

    def close(self) -> None:
        if self.thread is not None:
            self.thread.join()


def _take_depth() -> dict:
    """Spot and futures snapshots started together, so neither is a second later."""
    out: dict[str, dict] = {}
    errors: list[BaseException] = []

    def grab(futures: bool) -> None:
        try:
            out["futures" if futures else "spot"] = rest_depth_snapshot(futures=futures)
        except Exception as exc:
            errors.append(exc)

    threads = (
        threading.Thread(target=grab, args=(False,)),
        threading.Thread(target=grab, args=(True,)),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors or "spot" not in out or "futures" not in out:
        raise RuntimeError(errors[0] if errors else "depth snapshot missing a venue")
    fut = out["futures"]
    if fut.get("event_ts_ms") is None:
        fut["event_ts_ms"] = int(fut["recv_ts_ms"])
    if fut.get("transaction_ts_ms") is None:
        fut["transaction_ts_ms"] = int(fut["recv_ts_ms"])
    recv = max(int(out["spot"]["recv_ts_ms"]), int(out["futures"]["recv_ts_ms"]))
    return {"recv_ms": recv, "spot": out["spot"], "futures": out["futures"]}


def _write_book(folder: Path, anchor_ms: int, row: dict | None) -> None:
    target_ms = anchor_ms + HORIZON_SECONDS * 1000
    doc: dict = {
        "current_time_ms": anchor_ms,
        "target_ms": target_ms,
        "recv_ts_ms": None,
        "recv_ts_utc": None,
        "bid_price": None,
        "bid_qty": None,
        "ask_price": None,
        "ask_qty": None,
    }
    if row is not None:
        recv = int(row["recv_ts_ms"])
        doc.update(
            {
                "recv_ts_ms": recv,
                "recv_ts_utc": ms_to_utc(recv),
                "bid_price": float(row["bid_price"]),
                "bid_qty": float(row["bid_qty"]),
                "ask_price": float(row["ask_price"]),
                "ask_qty": float(row["ask_qty"]),
            }
        )
    path = book_path(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    if row is None:
        print(f"  no spot book tick in ({ms_to_utc(anchor_ms)}, {ms_to_utc(target_ms)}]", flush=True)
    else:
        print(
            f"  spot book  recv={doc['recv_ts_utc']}  "
            f"bid={doc['bid_price']} ask={doc['ask_price']}  -> {path}",
            flush=True,
        )


def _close_socks(socks: dict) -> None:
    for sock in list(socks):
        try:
            sock.close()
        except OSError:
            pass


def _connect() -> dict:
    socks: dict = {}
    _ensure_streams(socks)
    if not socks:
        raise RuntimeError("no Binance websockets connected")
    return socks


def _ensure_streams(socks: dict) -> None:
    have = {(venue, stream) for venue, _host, _port, _path, stream in socks.values()}
    for venue, host, port, path, stream_name in RAW_STREAMS:
        if (venue, stream_name) in have:
            continue
        try:
            sock = _ws_connect(host, path, port)
        except Exception as exc:
            print(f"  ws {venue} {stream_name} connect failed: {exc}", file=sys.stderr, flush=True)
            continue
        socks[sock] = (venue, host, port, path, stream_name)
        print(f"  ws {venue} {path} connected", flush=True)


def _drain(socks: dict, bufs: dict[str, VenueBuf], timeout: float) -> None:
    if not socks:
        time.sleep(timeout)
        return
    readable, _, _ = select.select(list(socks.keys()), [], [], timeout)
    for sock in readable:
        venue, host, port, path, stream_name = socks[sock]
        try:
            _drain_socket(sock, venue, stream_name, bufs)
        except (ConnectionError, OSError) as exc:
            print(f"  ws {venue} {stream_name} dropped: {exc}", flush=True)
            try:
                sock.close()
            except OSError:
                pass
            socks.pop(sock, None)
            try:
                fresh = _ws_connect(host, path, port)
                socks[fresh] = (venue, host, port, path, stream_name)
                print(f"  ws {venue} {stream_name} reconnected", flush=True)
            except Exception as rec_exc:
                print(f"  ws {venue} {stream_name} reconnect failed: {rec_exc}", file=sys.stderr)


def _apply_live_candles(
    bufs: dict[str, VenueBuf],
    spot_candles: CandleBook,
    fut_candles: CandleBook,
    fut_from_ms: int,
    fut_index: int,
) -> int:
    klines = bufs["spot"].klines
    bufs["spot"].klines = []
    for row in klines:
        spot_candles.upsert(
            int(row["open_time_ms"]),
            row["open"],
            row["high"],
            row["low"],
            row["close"],
            row["volume"],
        )
    trades = bufs["futures"].trades
    while fut_index < len(trades):
        row = trades[fut_index]
        fut_candles.add_trade(int(row["ts_ms"]), float(row["price"]), float(row["qty"]), fut_from_ms)
        fut_index += 1
    return fut_index


def _close_due(
    pending: list[_Pending],
    book_rows: list[dict],
    now_ms: int,
) -> list[_Pending]:
    still: list[_Pending] = []
    for job in pending:
        if now_ms < job.deadline_ms:
            still.append(job)
            continue
        row = last_spot_book_in_window(book_rows, job.anchor_ms, job.deadline_ms)
        _write_book(job.folder, job.anchor_ms, row)
    return still


def _fire_payload(
    *,
    anchor_ms: int,
    trigger: dict,
    bufs: dict[str, VenueBuf],
    snaps: list[dict],
    snap_lock: threading.Lock,
    spot_candles: CandleBook,
    fut_candles: CandleBook,
    database: Path,
    writer: _Writer,
    depth_bands: dict,
) -> _Pending | None:
    folder = database / str(anchor_ms)
    if folder.exists():
        print(f"  skip {anchor_ms} (folder already exists)", flush=True)
        return None
    with snap_lock:
        start, latest = select_depth_pair(list(snaps), anchor_ms)
    capture = LiveCapture(
        t0_ms=anchor_ms - TRADE_WINDOW_S * 1000,
        depth_start={"spot": start["spot"], "futures": start["futures"]},
        depth_latest={"spot": latest["spot"], "futures": latest["futures"]},
        venues=bufs,
        depth_bands=depth_bands,
    )
    payload = assemble_payload(
        capture,
        spot_candles=spot_candles.window(anchor_ms),
        futures_candles=fut_candles.window(anchor_ms),
        current_time_ms=anchor_ms,
        trigger=trigger,
    )
    try:
        assert_env_payload_schema(payload)
    except ValueError as exc:
        print(f"  payload schema warning: {exc}", file=sys.stderr, flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    path = payload_path(folder)
    spot_n = int(payload["venues"]["spot"]["trades"]["ts_ms"].size)
    fut_n = int(payload["venues"]["futures"]["trades"]["ts_ms"].size)
    age_s = (anchor_ms - int(start["recv_ms"])) / 1000.0
    latest_skew = int(latest["recv_ms"]) - anchor_ms
    kind = trigger["kind"]
    venue = trigger.get("venue")
    print(
        f"payload {kind} venue={venue}  {ms_to_utc(anchor_ms)}  "
        f"trades spot={spot_n} fut={fut_n}  "
        f"depth_start_age={age_s:.1f}s  depth_latest_skew_ms={latest_skew}",
        flush=True,
    )
    writer.submit(payload, path)
    return _Pending(
        anchor_ms=anchor_ms,
        folder=folder,
        deadline_ms=anchor_ms + HORIZON_SECONDS * 1000,
    )


def _seed_candles() -> tuple[CandleBook, CandleBook, int]:
    end_ms = _now_ms()
    start_ms = end_ms - CANDLE_WINDOW_S * 1000
    print(f"Seeding 1h spot candles  {ms_to_utc(start_ms)} -> {ms_to_utc(end_ms)}", flush=True)
    spot = CandleBook()
    spot.seed(rest_spot_candles_1s(start_ms, end_ms))
    print(f"Seeding 1h futures candles  {ms_to_utc(start_ms)} -> {ms_to_utc(end_ms)}", flush=True)
    fut = CandleBook()
    fut.seed(rest_futures_candles_1s(start_ms, end_ms))
    return spot, fut, end_ms


class _Pump:
    """Drain sockets on a side thread while the main thread seeds candles."""

    def __init__(self, socks: dict, bufs: dict[str, VenueBuf]) -> None:
        self.socks = socks
        self.bufs = bufs
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ws-pump", daemon=True)

    def __enter__(self) -> "_Pump":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                _drain(self.socks, self.bufs, 0.25)
            except Exception as exc:
                print(f"  live drain failed: {exc}", file=sys.stderr, flush=True)
                time.sleep(0.5)
                continue
            now_ms = _now_ms()
            keep = now_ms - BOOK_KEEP_MS
            _prune_prefix(self.bufs["spot"].book, keep)
            _prune_prefix(self.bufs["futures"].book, keep)
            _prune_prefix(self.bufs["spot"].trades, keep)
            _prune_prefix(self.bufs["spot"].depth_updates, keep)
            _prune_prefix(self.bufs["futures"].depth_updates, keep)
            _prune_prefix(self.bufs["futures"].liquidations, now_ms - LIQUIDATION_KEEP_MS)


class _BandBook:
    """Latest spot depth-band snapshot. The evaluator refreshes this about every 30s."""

    def __init__(self, bucket_usd: float) -> None:
        self._lock = threading.Lock()
        self._bands = empty_depth_bands(bucket_usd)

    def get(self) -> dict:
        with self._lock:
            return self._bands

    def set(self, bands: dict) -> None:
        with self._lock:
            self._bands = bands


class _DepthSampler:
    """REST depth once a second so depth_start is the book from ~60s earlier."""

    def __init__(self, snaps: list[dict], lock: threading.Lock, bands: _BandBook | None = None) -> None:
        self.snaps = snaps
        self.lock = lock
        self.bands = bands
        self._n = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="depth-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.time()
            try:
                snap = _take_depth()
            except Exception as exc:
                print(f"  depth snapshot failed: {exc}", file=sys.stderr, flush=True)
            else:
                with self.lock:
                    self.snaps.append(snap)
                    cutoff = int(snap["recv_ms"]) - 180_000
                    self.snaps[:] = [item for item in self.snaps if int(item["recv_ms"]) >= cutoff]
            self._n += 1
            if self.bands is not None and (self._n == 1 or self._n % 30 == 0):
                try:
                    self.bands.set(rest_spot_depth_bands())
                except Exception as exc:
                    print(f"  depth bands failed: {exc}", file=sys.stderr, flush=True)
            remain = DEPTH_SAMPLE_S - (time.time() - started)
            if remain > 0 and self._stop.wait(remain):
                return


def run_capture(
    *,
    interval_s: float,
    database: Path,
    count: int | None = None,
    hours: float | None = None,
    asset: str = "BTC",
) -> int:
    spec = use_asset(asset)
    name = str(asset).strip().upper()
    clock_ms = clock_interval_ms(interval_s)
    offset_ms = spec.slot_offset_ms
    database.mkdir(parents=True, exist_ok=True)
    removed = delete_captures_older_than(database, _now_ms())
    if removed:
        print(f"Removed {removed} capture folder(s) older than 7 days.", flush=True)
    print("Connecting live trades / bookTicker / depth / spot klines / liquidations...", flush=True)
    bufs = {"spot": VenueBuf(), "futures": VenueBuf()}
    socks = _connect()
    snaps: list[dict] = []
    snap_lock = threading.Lock()
    bands = _BandBook(spec.bucket_usd)
    sampler = _DepthSampler(snaps, snap_lock, bands)
    writer = _Writer()
    sampler.start()
    fired = 0
    try:
        connected_at = time.time()
        try:
            with _Pump(socks, bufs):
                spot_candles, fut_candles, fut_from_ms = _seed_candles()
        except KeyboardInterrupt:
            print("Stopped during candle seed. No pairs written.", flush=True)
            return 1
        fut_index = _apply_live_candles(bufs, spot_candles, fut_candles, fut_from_ms, 0)
        _prune_prefix(bufs["futures"].trades, _now_ms() - BOOK_KEEP_MS)
        fut_index = len(bufs["futures"].trades)
        last_reconnect = time.time()
        remain = TRADE_WINDOW_S - (time.time() - connected_at)
        if remain > 0:
            print(
                f"Warming up {remain:.0f}s so the first payload has a full trade/book window...",
                flush=True,
            )
            warmup_deadline = time.time() + remain
            try:
                while time.time() < warmup_deadline:
                    _drain(socks, bufs, 0.25)
                    fut_index = _apply_live_candles(
                        bufs, spot_candles, fut_candles, fut_from_ms, fut_index
                    )
            except KeyboardInterrupt:
                print("Stopped during warmup. No pairs written.", flush=True)
                return 1

        ticks = fetch_tick_sizes(spec.symbol)
        print(
            f"Tick size  spot={ticks['spot']}  futures={ticks['futures']}",
            flush=True,
        )
        detectors = {
            "spot": TradeGroupDetector(ticks["spot"]),
            "futures": TradeGroupDetector(ticks["futures"]),
        }
        spot_i = len(bufs["spot"].trades)
        fut_move_i = len(bufs["futures"].trades)
        rng = random.Random()
        events: list[_ScheduledEvent] = []
        taken_windows: set[int] = set()

        origin = time.time()
        stop_at = None if hours is None else origin + float(hours) * 3600.0
        next_boundary = next_time_boundary(_now_ms() + 1, clock_ms=clock_ms, offset_ms=offset_ms)
        pending: list[_Pending] = []
        limit = "until Ctrl+C" if count is None and hours is None else (
            f"count={count}" if count is not None else f"hours={hours:g}"
        )
        scale = clock_ms // CLOCK_INTERVAL_MS
        print(
            f"{name} {spec.symbol}  slot offset {offset_ms} ms  bucket ${spec.bucket_usd}. "
            f"Time triggers every {clock_ms} ms ({scale}x the live {CLOCK_INTERVAL_MS} ms clock). "
            f"Event window opens {EVENT_WINDOW_OPEN_MS} ms later for {EVENT_WINDOW_LEN_MS} ms "
            f"(half event, half event_delayed < {EVENT_DELAY_MAX_MS} ms). {limit}. "
            f"dir={database.resolve()}",
            flush=True,
        )
        while socks or pending or events:
            soon = any(job.deadline_ms - _now_ms() <= 1000 for job in pending) or any(
                ev.fire_at_ms - _now_ms() <= 1000 for ev in events
            )
            _drain(socks, bufs, 0.05 if soon else 0.25)
            fut_index = _apply_live_candles(bufs, spot_candles, fut_candles, fut_from_ms, fut_index)
            now_ms = _now_ms()
            now = time.time()
            want_more = (count is None or fired < count) and (stop_at is None or now < stop_at)

            def _watch(venue: str, rows: list[dict], index: int) -> tuple[int, list[tuple[int, str]]]:
                found: list[tuple[int, str]] = []
                while index < len(rows):
                    row = rows[index]
                    index += 1
                    recv = detectors[venue].push(
                        int(row["ts_ms"]), float(row["price"]), int(row["recv_ts_ms"])
                    )
                    if recv is not None:
                        found.append((recv, venue))
                return index, found

            spot_i, spot_hits = _watch("spot", bufs["spot"].trades, spot_i)
            fut_move_i, fut_hits = _watch("futures", bufs["futures"].trades, fut_move_i)
            if want_more:
                for recv, venue in sorted(spot_hits + fut_hits):
                    boundary = event_window_boundary(
                        recv, clock_ms=clock_ms, offset_ms=offset_ms
                    )
                    if boundary is None or not take_qualifying_move(rng, boundary, taken_windows):
                        continue
                    if rng.random() < 0.5:
                        kind = "event"
                        fire_at = recv
                    else:
                        kind = "event_delayed"
                        fire_at = recv + event_delay_ms(rng)
                    events.append(_ScheduledEvent(recv, kind, venue, fire_at))
            pending = _close_due(pending, bufs["spot"].book, now_ms)
            if now - last_reconnect >= 5.0 and len(socks) < len(RAW_STREAMS):
                _ensure_streams(socks)
                last_reconnect = now
            still_events: list[_ScheduledEvent] = []
            for ev in events:
                if now_ms < ev.fire_at_ms:
                    still_events.append(ev)
                    continue
                if not want_more:
                    continue
                try:
                    job = _fire_payload(
                        anchor_ms=ev.anchor_ms,
                        trigger={"kind": ev.kind, "venue": ev.venue},
                        bufs=bufs,
                        snaps=snaps,
                        snap_lock=snap_lock,
                        spot_candles=spot_candles,
                        fut_candles=fut_candles,
                        database=database,
                        writer=writer,
                        depth_bands=bands.get(),
                    )
                except Exception as exc:
                    print(f"  payload failed: {exc}", file=sys.stderr, flush=True)
                    job = None
                if job is not None:
                    pending.append(job)
                    fired += 1
            events[:] = still_events
            if want_more and now_ms >= next_boundary:
                anchor = next_boundary
                while anchor + clock_ms <= now_ms:
                    anchor += clock_ms
                try:
                    job = _fire_payload(
                        anchor_ms=anchor,
                        trigger={"kind": "time", "venue": None},
                        bufs=bufs,
                        snaps=snaps,
                        snap_lock=snap_lock,
                        spot_candles=spot_candles,
                        fut_candles=fut_candles,
                        database=database,
                        writer=writer,
                        depth_bands=bands.get(),
                    )
                except Exception as exc:
                    print(f"  payload failed: {exc}", file=sys.stderr, flush=True)
                    job = None
                if job is not None:
                    pending.append(job)
                    fired += 1
                next_boundary = anchor + clock_ms
            book_cut = now_ms - BOOK_KEEP_MS
            _prune_prefix(bufs["spot"].book, book_cut)
            _prune_prefix(bufs["futures"].book, book_cut)
            spot_cut = _prune_prefix(bufs["spot"].trades, book_cut)
            spot_i = max(0, spot_i - spot_cut)
            trade_cut = _prune_prefix(bufs["futures"].trades, book_cut)
            fut_index = max(0, fut_index - trade_cut)
            fut_move_i = max(0, fut_move_i - trade_cut)
            _prune_prefix(bufs["spot"].depth_updates, book_cut)
            _prune_prefix(bufs["futures"].depth_updates, book_cut)
            _prune_prefix(bufs["futures"].liquidations, now_ms - LIQUIDATION_KEEP_MS)
            stale_windows = [b for b in taken_windows if b + 2 * clock_ms < now_ms]
            for boundary in stale_windows:
                taken_windows.discard(boundary)
            if not want_more and not pending and not events:
                break
            if not socks and not pending and not events:
                break
    except KeyboardInterrupt:
        print("Stopped. Unfinished 10s windows were not saved.", flush=True)
    finally:
        sampler.stop()
        writer.close()
        _close_socks(socks)
    if writer.errors:
        return 1
    print(f"Captured {fired} payload(s) under {database.resolve()}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write payload + 10s spot book pairs under database/{current_time_ms}/"
    )
    parser.add_argument(
        "--asset",
        default="BTC",
        help="asset to capture: BTC, ETH, SOL, or XRP (default BTC)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=200.0,
        help=(
            "clock period in seconds. A whole multiple of the live 20s clock. "
            "200 (default) keeps every tenth cycle; 20 matches the evaluator"
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="stop after this many payloads (default: run until Ctrl+C or --hours)",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="stop firing new payloads after this many hours",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help=f"output root (default {DEFAULT_DATABASE}/<asset>)",
    )
    args = parser.parse_args(argv)
    if args.count is not None and args.count <= 0:
        parser.error("--count must be > 0")
    if args.hours is not None and args.hours <= 0:
        parser.error("--hours must be > 0")
    asset = args.asset.strip().upper()
    try:
        asset_spec(asset)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        clock_interval_ms(args.interval)
    except ValueError as exc:
        parser.error(str(exc))
    database = args.database
    if database == DEFAULT_DATABASE:
        database = DEFAULT_DATABASE / asset
    try:
        return run_capture(
            interval_s=args.interval,
            database=database,
            count=args.count,
            hours=args.hours,
            asset=asset,
        )
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
