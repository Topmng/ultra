"""Capture one Synth Ultra evaluation payload from live Binance feeds.

This is separate from fetch_history.py (historical CSV dumps). It records the
same windows the evaluator sends: 1h of 1s candles, 60s of trades / bookTicker /
depth diffs, and 20-level depth_start + depth_latest snapshots.

    python fetch_live.py
    python fetch_live.py --window-seconds 60 --out examples/env_payload.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import select
import socket
import ssl
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from synth_ultra.constants import (
    CANDLE_WINDOW_S,
    DEPTH_LEVELS,
    HORIZON_SECONDS,
    NUM_PERCENTILES,
    SCHEMA_VERSION,
    TRADE_WINDOW_S,
)
from synth_ultra.payload import ENV_PAYLOAD_JSON, payload_to_jsonable

SYMBOL = "BTCUSDT"
STREAM_SYMBOL = "btcusdt"
LIMIT = 1000
SPOT_KLINES_URL = "https://api.binance.com/api/v3/klines"
SPOT_DEPTH_URL = "https://api.binance.com/api/v3/depth"
SPOT_AGGTRADES_URL = "https://api.binance.com/api/v3/aggTrades"
FUTURES_DEPTH_URL = "https://fapi.binance.com/fapi/v1/depth"
FUTURES_AGGTRADES_URL = "https://fapi.binance.com/fapi/v1/aggTrades"
HEADERS = {"Accept": "*/*", "User-Agent": "synth-ultra-payload-fetch/1.0"}
TIMEOUT_S = 30
MAX_RETRIES = 5


def _now_ms() -> int:
    return int(time.time() * 1000)


def _to_ms(ts: int) -> int:
    if ts >= 10**15:
        return ts // 1000
    return ts


def _http_open(url: str, timeout: int = TIMEOUT_S) -> bytes:
    request = urllib.request.Request(url, headers=HEADERS, method="GET")
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 404:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if exc.code in (418, 429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                wait_s = float(retry_after) if retry_after else min(2**attempt, 16)
                time.sleep(wait_s)
                continue
            raise RuntimeError(f"Binance HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(min(2**attempt, 16))
                continue
            raise RuntimeError(f"Binance request failed: {exc}") from exc
    raise RuntimeError(f"Binance request failed: {last_error}")


def _http_get_json(url: str, params: dict[str, str | int]) -> Any:
    query = urllib.parse.urlencode(params)
    raw = _http_open(f"{url}?{query}")
    return json.loads(raw.decode("utf-8"))


def rest_depth_snapshot(*, futures: bool) -> dict:
    """20-level REST book, stamped with local recv_ts_ms (evaluator clock)."""
    recv = _now_ms()
    url = FUTURES_DEPTH_URL if futures else SPOT_DEPTH_URL
    data = _http_get_json(url, {"symbol": SYMBOL, "limit": DEPTH_LEVELS})
    bids = np.asarray(
        [[float(p), float(q)] for p, q in (data.get("bids") or [])[:DEPTH_LEVELS]],
        dtype=np.float64,
    )
    asks = np.asarray(
        [[float(p), float(q)] for p, q in (data.get("asks") or [])[:DEPTH_LEVELS]],
        dtype=np.float64,
    )
    if bids.size == 0:
        bids = np.zeros((0, 2), dtype=np.float64)
    if asks.size == 0:
        asks = np.zeros((0, 2), dtype=np.float64)
    return {
        "recv_ts_ms": np.int64(recv),
        "update_id": int(data["lastUpdateId"]),
        "bids": np.ascontiguousarray(bids, dtype=np.float64),
        "asks": np.ascontiguousarray(asks, dtype=np.float64),
        "event_ts_ms": int(data["E"]) if futures and data.get("E") is not None else None,
        "transaction_ts_ms": int(data["T"]) if futures and data.get("T") is not None else None,
    }


def rest_spot_candles_1s(start_ms: int, end_ms: int) -> dict:
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _http_get_json(
            SPOT_KLINES_URL,
            {
                "symbol": SYMBOL,
                "interval": "1s",
                "startTime": cursor,
                "endTime": end_ms - 1,
                "limit": LIMIT,
            },
        )
        if not batch:
            break
        if rows and _to_ms(int(batch[0][0])) == int(rows[-1][0]):
            batch = batch[1:]
        if not batch:
            break
        stop = False
        for item in batch:
            ts = _to_ms(int(item[0]))
            if ts < start_ms:
                continue
            if ts >= end_ms:
                stop = True
                break
            rows.append([ts, float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5])])
        if stop or len(batch) < LIMIT:
            break
        cursor = int(rows[-1][0]) + 1000
        print(f"  spot klines  {len(rows)}", flush=True)
    open_time = np.asarray([r[0] for r in rows], dtype=np.int64)
    ohlcv = np.asarray([r[1:] for r in rows], dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
    return {
        "open_time_ms": open_time,
        "ohlcv": ohlcv if ohlcv.ndim == 2 else ohlcv.reshape(-1, 5),
        "complete_history": bool(open_time.size >= CANDLE_WINDOW_S),
    }


def rest_futures_candles_1s(start_ms: int, end_ms: int) -> dict:
    """Build 1s OHLCV from USDT-M aggTrades (fapi has no 1s kline feed)."""
    trades: list[tuple[int, float, float]] = []
    cursor = start_ms
    from_id: int | None = None
    print("  futures aggTrades (1s candles)...", flush=True)
    while cursor < end_ms:
        params: dict[str, str | int] = {"symbol": SYMBOL, "limit": LIMIT}
        if from_id is None:
            params["startTime"] = cursor
            params["endTime"] = end_ms - 1
        else:
            params["fromId"] = from_id
        batch = _http_get_json(FUTURES_AGGTRADES_URL, params)
        if not batch:
            if from_id is None:
                break
            from_id = None
            cursor = min(cursor + 60_000, end_ms)
            continue
        stop = False
        for item in batch:
            ts = int(item["T"])
            if ts < start_ms:
                continue
            if ts >= end_ms:
                stop = True
                break
            trades.append((ts, float(item["p"]), float(item["q"])))
        if stop:
            break
        from_id = int(batch[-1]["a"]) + 1
        last_ts = int(batch[-1]["T"])
        if last_ts > cursor:
            cursor = last_ts
        if len(batch) < LIMIT:
            break
        if len(trades) % 20000 < LIMIT:
            print(f"  futures aggTrades  {len(trades)}", flush=True)
        time.sleep(0.05)

    start_ms = (start_ms // 1000) * 1000
    n = max((end_ms - start_ms) // 1000, 1)
    buckets: dict[int, list[float]] = {}
    for ts, price, qty in trades:
        open_time = (ts // 1000) * 1000
        candle = buckets.get(open_time)
        if candle is None:
            buckets[open_time] = [price, price, price, price, qty]
        else:
            candle[1] = max(candle[1], price)
            candle[2] = min(candle[2], price)
            candle[3] = price
            candle[4] += qty
    if not trades:
        return {
            "open_time_ms": np.array([], dtype=np.int64),
            "ohlcv": np.zeros((0, 5), dtype=np.float64),
            "complete_history": False,
        }
    prev = trades[0][1]
    open_times: list[int] = []
    ohlcv_rows: list[list[float]] = []
    for i in range(n):
        open_time = start_ms + i * 1000
        candle = buckets.get(open_time)
        if candle is None:
            open_times.append(open_time)
            ohlcv_rows.append([prev, prev, prev, prev, 0.0])
        else:
            open_times.append(open_time)
            ohlcv_rows.append(candle)
            prev = candle[3]
    return {
        "open_time_ms": np.asarray(open_times, dtype=np.int64),
        "ohlcv": np.asarray(ohlcv_rows, dtype=np.float64),
        "complete_history": False,
    }


def _ws_connect(host: str, path: str, port: int) -> ssl.SSLSocket:
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    raw = socket.create_connection((host, port), timeout=30)
    sock = ssl.create_default_context().wrap_socket(raw, server_hostname=host)
    sock.sendall(
        (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        ).encode("ascii")
    )
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise RuntimeError(f"websocket handshake closed: {host}")
        buf += chunk
    header, _, rest = buf.partition(b"\r\n\r\n")
    if b"101" not in header.split(b"\r\n", 1)[0]:
        raise RuntimeError(f"websocket handshake failed: {header[:200]!r}")
    sock._ws_buf = rest  # type: ignore[attr-defined]
    sock.settimeout(1.0)
    return sock


def _ws_send_frame(sock: ssl.SSLSocket, opcode: int, payload: bytes) -> None:
    mask = os.urandom(4)
    header = bytearray([0x80 | opcode])
    n = len(payload)
    if n < 126:
        header.append(0x80 | n)
    elif n < 65536:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", n))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", n))
    header.extend(mask)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    sock.sendall(bytes(header) + masked)


def _ws_recv_exact(sock: ssl.SSLSocket, n: int) -> bytes:
    buf: bytes = getattr(sock, "_ws_buf", b"")
    while len(buf) < n:
        chunk = sock.recv(max(n - len(buf), 4096))
        if not chunk:
            raise ConnectionError("websocket closed")
        buf += chunk
    out, rest = buf[:n], buf[n:]
    sock._ws_buf = rest  # type: ignore[attr-defined]
    return out


def _ws_recv_message(sock: ssl.SSLSocket) -> str | None:
    payload = bytearray()
    while True:
        hdr = _ws_recv_exact(sock, 2)
        fin = hdr[0] >> 7
        opcode = hdr[0] & 0x0F
        length = hdr[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", _ws_recv_exact(sock, 2))[0]
        elif length == 127:
            length = struct.unpack("!Q", _ws_recv_exact(sock, 8))[0]
        if hdr[1] & 0x80:
            mask = _ws_recv_exact(sock, 4)
            data = bytes(b ^ mask[i % 4] for i, b in enumerate(_ws_recv_exact(sock, length)))
        else:
            data = _ws_recv_exact(sock, length)
        if opcode == 0x8:
            return None
        if opcode == 0x9:
            _ws_send_frame(sock, 0xA, data)
            continue
        if opcode == 0xA:
            continue
        payload.extend(data)
        if fin:
            return payload.decode("utf-8")


@dataclass
class VenueBuf:
    trades: list[dict] = field(default_factory=list)
    book: list[dict] = field(default_factory=list)
    depth_updates: list[dict] = field(default_factory=list)
    last_recv: dict[str, int] = field(default_factory=dict)


@dataclass
class LiveCapture:
    t0_ms: int
    depth_start: dict[str, dict]
    depth_latest: dict[str, dict]
    venues: dict[str, VenueBuf]


def _parse_ws(stream: str, data: dict, recv_ms: int, *, futures: bool, buf: VenueBuf) -> None:
    name = stream.split("@", 1)[-1] if stream else str(data.get("e") or "")
    if name == "aggTrade" or data.get("e") == "aggTrade":
        buf.trades.append(
            {
                "ts_ms": int(data["T"]),
                "event_ts_ms": int(data["E"]),
                "recv_ts_ms": recv_ms,
                "price": float(data["p"]),
                "qty": float(data["q"]),
                "buyer_is_maker": bool(data["m"]),
            }
        )
        buf.last_recv["trade"] = recv_ms
        return
    if name == "bookTicker" or (
        "b" in data and "B" in data and "a" in data and "A" in data and "U" not in data
    ):
        row: dict[str, Any] = {
            "recv_ts_ms": recv_ms,
            "bid_price": float(data["b"]),
            "bid_qty": float(data["B"]),
            "ask_price": float(data["a"]),
            "ask_qty": float(data["A"]),
        }
        if futures:
            row["event_ts_ms"] = int(data["E"])
            row["transaction_ts_ms"] = int(data["T"])
        buf.book.append(row)
        buf.last_recv["bookTicker"] = recv_ms
        return
    if "depth" in name or data.get("e") == "depthUpdate" or "U" in data:
        upd: dict[str, Any] = {
            "recv_ts_ms": np.int64(recv_ms),
            "event_ts_ms": np.int64(data["E"]),
            "first_id": int(data["U"]),
            "final_id": int(data["u"]),
            "prev_final_id": int(data["pu"]) if data.get("pu") is not None else int(data["U"]) - 1,
            "bids": np.asarray(
                [[float(p), float(q)] for p, q in data.get("b") or []], dtype=np.float64
            ).reshape(-1, 2),
            "asks": np.asarray(
                [[float(p), float(q)] for p, q in data.get("a") or []], dtype=np.float64
            ).reshape(-1, 2),
        }
        if futures:
            upd["transaction_ts_ms"] = np.int64(data["T"])
        buf.depth_updates.append(upd)
        buf.last_recv["depth"] = recv_ms
        return
    if name.startswith("kline") or data.get("e") == "kline":
        buf.last_recv["kline_1s"] = recv_ms


RAW_STREAMS: list[tuple[str, str, int, str]] = [
    ("spot", "stream.binance.com", 9443, f"{STREAM_SYMBOL}@aggTrade"),
    ("spot", "stream.binance.com", 9443, f"{STREAM_SYMBOL}@bookTicker"),
    ("spot", "stream.binance.com", 9443, f"{STREAM_SYMBOL}@depth@100ms"),
    ("spot", "stream.binance.com", 9443, f"{STREAM_SYMBOL}@kline_1s"),
    ("futures", "fstream.binance.com", 443, f"{STREAM_SYMBOL}@aggTrade"),
    ("futures", "fstream.binance.com", 443, f"{STREAM_SYMBOL}@bookTicker"),
    ("futures", "fstream.binance.com", 443, f"{STREAM_SYMBOL}@depth@100ms"),
]


def _ingest_text(
    text: str, venue: str, stream_name: str, recv_ms: int, bufs: dict[str, VenueBuf]
) -> None:
    msg = json.loads(text)
    if not isinstance(msg, dict):
        return
    if msg.get("id") is not None and "result" in msg:
        return
    data = msg.get("data") if isinstance(msg.get("data"), dict) else msg
    if not isinstance(data, dict):
        return
    stream = str(msg.get("stream") or data.get("e") or stream_name)
    _parse_ws(stream, data, recv_ms, futures=(venue == "futures"), buf=bufs[venue])


def _drain_socket(
    sock: ssl.SSLSocket, venue: str, stream_name: str, bufs: dict[str, VenueBuf]
) -> None:
    sock.settimeout(0.0)
    try:
        while True:
            text = _ws_recv_message(sock)
            if text is None:
                break
            _ingest_text(text, venue, stream_name, _now_ms(), bufs)
    except (TimeoutError, BlockingIOError, ssl.SSLWantReadError, ConnectionError):
        pass
    finally:
        sock.settimeout(1.0)


def record_live_window(window_s: float) -> LiveCapture:
    """Subscribe first, snapshot depth_start, record `window_s`, snapshot depth_latest."""
    socks: dict[ssl.SSLSocket, tuple[str, str, int, str]] = {}
    bufs = {"spot": VenueBuf(), "futures": VenueBuf()}
    for venue, host, port, stream_name in RAW_STREAMS:
        sock = _ws_connect(host, f"/ws/{stream_name}", port)
        socks[sock] = (venue, host, port, stream_name)
        print(f"  ws {venue} {stream_name} connected", flush=True)

    depth_start = {
        "spot": rest_depth_snapshot(futures=False),
        "futures": rest_depth_snapshot(futures=True),
    }
    t0_ms = int(min(int(depth_start["spot"]["recv_ts_ms"]), int(depth_start["futures"]["recv_ts_ms"])))
    print(
        f"  depth_start  spot={depth_start['spot']['update_id']}  "
        f"futures={depth_start['futures']['update_id']}",
        flush=True,
    )

    deadline = time.time() + window_s
    last_print = time.time()
    try:
        while time.time() < deadline and socks:
            readable, _, _ = select.select(list(socks.keys()), [], [], 1.0)
            for sock in readable:
                venue, host, port, stream_name = socks[sock]
                try:
                    _drain_socket(sock, venue, stream_name, bufs)
                except (ConnectionError, OSError) as exc:
                    print(f"  ws {venue} {stream_name} dropped: {exc}", flush=True)
                    try:
                        sock.close()
                    except OSError:
                        pass
                    socks.pop(sock, None)
                    fresh = _ws_connect(host, f"/ws/{stream_name}", port)
                    socks[fresh] = (venue, host, port, stream_name)
                    print(f"  ws {venue} {stream_name} reconnected", flush=True)
            if time.time() - last_print >= 5.0:
                remain = max(deadline - time.time(), 0.0)
                print(
                    f"  live  t-{remain:.0f}s  "
                    f"spot trades={len(bufs['spot'].trades)} book={len(bufs['spot'].book)} "
                    f"depth={len(bufs['spot'].depth_updates)}  "
                    f"fut trades={len(bufs['futures'].trades)} book={len(bufs['futures'].book)} "
                    f"depth={len(bufs['futures'].depth_updates)}",
                    flush=True,
                )
                last_print = time.time()
        for sock, (venue, _host, _port, stream_name) in list(socks.items()):
            _drain_socket(sock, venue, stream_name, bufs)
    finally:
        for sock in list(socks.keys()):
            try:
                sock.close()
            except OSError:
                pass

    depth_latest = {
        "spot": rest_depth_snapshot(futures=False),
        "futures": rest_depth_snapshot(futures=True),
    }
    return LiveCapture(
        t0_ms=t0_ms,
        depth_start=depth_start,
        depth_latest=depth_latest,
        venues=bufs,
    )


def _window_rows(rows: list[dict], lo_ms: int, hi_ms: int, key: str = "recv_ts_ms") -> list[dict]:
    return [r for r in rows if lo_ms < int(r[key]) <= hi_ms]


def _stack_trades(rows: list[dict]) -> dict:
    if not rows:
        return {
            "ts_ms": np.array([], dtype=np.int64),
            "event_ts_ms": np.array([], dtype=np.int64),
            "recv_ts_ms": np.array([], dtype=np.int64),
            "price": np.array([], dtype=np.float64),
            "qty": np.array([], dtype=np.float64),
            "buyer_is_maker": np.array([], dtype=bool),
        }
    return {
        "ts_ms": np.asarray([r["ts_ms"] for r in rows], dtype=np.int64),
        "event_ts_ms": np.asarray([r["event_ts_ms"] for r in rows], dtype=np.int64),
        "recv_ts_ms": np.asarray([r["recv_ts_ms"] for r in rows], dtype=np.int64),
        "price": np.asarray([r["price"] for r in rows], dtype=np.float64),
        "qty": np.asarray([r["qty"] for r in rows], dtype=np.float64),
        "buyer_is_maker": np.asarray([r["buyer_is_maker"] for r in rows], dtype=bool),
    }


def _stack_book(rows: list[dict], *, futures: bool) -> dict:
    if not rows:
        out: dict[str, Any] = {
            "recv_ts_ms": np.array([], dtype=np.int64),
            "bid_price": np.array([], dtype=np.float64),
            "bid_qty": np.array([], dtype=np.float64),
            "ask_price": np.array([], dtype=np.float64),
            "ask_qty": np.array([], dtype=np.float64),
        }
        if futures:
            out["event_ts_ms"] = np.array([], dtype=np.int64)
            out["transaction_ts_ms"] = np.array([], dtype=np.int64)
        return out
    out = {
        "recv_ts_ms": np.asarray([r["recv_ts_ms"] for r in rows], dtype=np.int64),
        "bid_price": np.asarray([r["bid_price"] for r in rows], dtype=np.float64),
        "bid_qty": np.asarray([r["bid_qty"] for r in rows], dtype=np.float64),
        "ask_price": np.asarray([r["ask_price"] for r in rows], dtype=np.float64),
        "ask_qty": np.asarray([r["ask_qty"] for r in rows], dtype=np.float64),
    }
    if futures:
        out["event_ts_ms"] = np.asarray([r["event_ts_ms"] for r in rows], dtype=np.int64)
        out["transaction_ts_ms"] = np.asarray([r["transaction_ts_ms"] for r in rows], dtype=np.int64)
    return out


def _filter_depth_updates(
    rows: list[dict], start: dict, lo_ms: int, hi_ms: int
) -> list[dict]:
    start_id = int(start["update_id"])
    out: list[dict] = []
    for row in rows:
        recv = int(row["recv_ts_ms"])
        if recv <= lo_ms or recv > hi_ms:
            continue
        if int(row["final_id"]) < start_id:
            continue
        out.append(row)
    return out


def assemble_venue(
    *,
    futures: bool,
    buf: VenueBuf,
    depth_start: dict,
    depth_latest: dict,
    candles: dict,
    lo_ms: int,
    hi_ms: int,
) -> dict:
    trades = _stack_trades(_window_rows(buf.trades, lo_ms, hi_ms))
    book = _stack_book(_window_rows(buf.book, lo_ms, hi_ms), futures=futures)
    updates = _filter_depth_updates(buf.depth_updates, depth_start, lo_ms, hi_ms)
    last = dict(buf.last_recv)
    last.setdefault("trade", int(trades["recv_ts_ms"][-1]) if trades["recv_ts_ms"].size else hi_ms)
    last.setdefault("bookTicker", int(book["recv_ts_ms"][-1]) if book["recv_ts_ms"].size else hi_ms)
    last.setdefault("depth", int(depth_latest["recv_ts_ms"]))
    last.setdefault(
        "kline_1s",
        int(candles["open_time_ms"][-1]) if candles["open_time_ms"].size else hi_ms,
    )
    return {
        "symbol": SYMBOL,
        "candles_1s": candles,
        "trades": trades,
        "book_ticker": book,
        "depth_start": depth_start,
        "depth_updates": updates,
        "depth_latest": depth_latest,
        "last_event_times": {k: np.int64(v) for k, v in last.items()},
    }


def assemble_payload(
    capture: LiveCapture,
    *,
    spot_candles: dict,
    futures_candles: dict,
    current_time_ms: int,
    trigger: dict | None = None,
) -> dict:
    lo_ms = current_time_ms - TRADE_WINDOW_S * 1000
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt": {
            "asset": "BTC",
            "horizon_seconds": HORIZON_SECONDS,
            "num_percentiles": NUM_PERCENTILES,
            "quantile_grid": "centered-100",
            "current_time_ms": int(current_time_ms),
            "trigger": trigger or {"kind": "interval", "venue": None},
        },
        "venues": {
            "spot": assemble_venue(
                futures=False,
                buf=capture.venues["spot"],
                depth_start=capture.depth_start["spot"],
                depth_latest=capture.depth_latest["spot"],
                candles=spot_candles,
                lo_ms=lo_ms,
                hi_ms=current_time_ms,
            ),
            "futures": assemble_venue(
                futures=True,
                buf=capture.venues["futures"],
                depth_start=capture.depth_start["futures"],
                depth_latest=capture.depth_latest["futures"],
                candles=futures_candles,
                lo_ms=lo_ms,
                hi_ms=current_time_ms,
            ),
        },
    }


def assert_env_payload_schema(payload: dict) -> None:
    """Fail if the payload does not match input.md schema_version 3."""
    if int(payload["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("schema_version")
    prompt = payload["prompt"]
    for key in ("asset", "horizon_seconds", "num_percentiles", "quantile_grid", "current_time_ms", "trigger"):
        if key not in prompt:
            raise ValueError(f"prompt.{key}")
    for name, futures in (("spot", False), ("futures", True)):
        venue = payload["venues"][name]
        if venue["symbol"] != SYMBOL:
            raise ValueError(f"{name}.symbol")
        candles = venue["candles_1s"]
        if candles["ohlcv"].shape[-1] != 5:
            raise ValueError(f"{name}.candles_1s.ohlcv")
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
            if snap["bids"].ndim != 2 or snap["bids"].shape[1] != 2:
                raise ValueError(f"{name}.{snap_name}.bids")
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
        if not venue["last_event_times"]:
            raise ValueError(f"{name}.last_event_times")


def save_payload(payload: dict, path: Path, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload_to_jsonable(payload),
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )
    path.write_text(text, encoding="utf-8")


def capture_env_payload(
    *,
    window_s: float = TRADE_WINDOW_S,
    trigger: dict | None = None,
) -> dict:
    print(f"Recording live window {window_s:.0f}s (trades / bookTicker / depth@100ms)...", flush=True)
    capture = record_live_window(window_s)
    current_time_ms = _now_ms()
    candle_start = current_time_ms - CANDLE_WINDOW_S * 1000
    print("Fetching trailing 1h candles...", flush=True)
    spot_candles = rest_spot_candles_1s(candle_start, current_time_ms)
    futures_candles = rest_futures_candles_1s(candle_start, current_time_ms)
    payload = assemble_payload(
        capture,
        spot_candles=spot_candles,
        futures_candles=futures_candles,
        current_time_ms=current_time_ms,
        trigger=trigger,
    )
    assert_env_payload_schema(payload)
    return payload


def _summarize(payload: dict) -> None:
    prompt = payload["prompt"]
    print(f"schema_version={payload['schema_version']}  current_time_ms={prompt['current_time_ms']}")
    for name in ("spot", "futures"):
        v = payload["venues"][name]
        c = v["candles_1s"]
        t = v["trades"]
        b = v["book_ticker"]
        print(
            f"  {name:7} candles={c['open_time_ms'].size} complete={c['complete_history']}  "
            f"trades={t['ts_ms'].size}  book={b['recv_ts_ms'].size}  "
            f"depth_updates={len(v['depth_updates'])}  "
            f"depth_levels={v['depth_latest']['bids'].shape[0]}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture one evaluation-shaped Synth Ultra payload from live Binance data"
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=float(TRADE_WINDOW_S),
        help="live trades/book/depth window (default 60, matching input.md)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ENV_PAYLOAD_JSON,
        help=f"JSON output path (default {ENV_PAYLOAD_JSON})",
    )
    parser.add_argument("--pretty", action="store_true", help="indent JSON")
    parser.add_argument(
        "--trigger",
        choices=("interval", "trade", "book"),
        default="interval",
        help="prompt.trigger.kind",
    )
    args = parser.parse_args(argv)
    try:
        payload = capture_env_payload(
            window_s=args.window_seconds,
            trigger={"kind": args.trigger, "venue": None},
        )
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    save_payload(payload, args.out, pretty=args.pretty)
    _summarize(payload)
    print(f"wrote {args.out.resolve()}  ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
