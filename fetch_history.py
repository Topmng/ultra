"""Fetch Binance BTCUSDT market data into database/ for Synth Ultra payloads.

    python fetch_history.py

Default window is trailing 48 hours (more than one payload-day). Existing CSVs
are resumed after their last timestamp and backfilled so coverage stays >= 48h.

Payload streams in input.md vs what Binance actually publishes:

  candles_1s     spot 1s klines (Vision daily zip, REST for today)
                 futures 1s OHLCV rebuilt from aggTrades (fapi has no 1s kline)
  trades         spot + USDT-M aggTrades (Vision zip, REST fallback)
  book_ticker    Vision bookTicker when the daily zip exists, plus one REST snapshot.
                 Candle closes are not a book. Recent days often 404; record
                 the real stream with --live-seconds.
  depth          one REST /depth snapshot (20 levels). bookDepth percent bins
                 are a different product and are stored only in
                 btc_futures_book_depth.csv. Diffs are the live depth@100ms
                 stream (--live-seconds), not a snapshot replay.

Writes under database/:
  btc_spot_candles.csv
  btc_futures_candles.csv
  btc_spot_trades.csv
  btc_futures_trades.csv
  btc_spot_book_ticker.csv
  btc_futures_book_ticker.csv
  btc_spot_depth.csv
  btc_futures_depth.csv
  btc_futures_book_depth.csv
  btc_spot_depth_updates.jsonl
  btc_futures_depth_updates.jsonl

Each millisecond timestamp is followed by its UTC datetime
(`2026-08-19 10:40:55.065+00:00`).
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import select
import ssl
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parent
DATABASE_DIR = REPO_ROOT / "database"
SYMBOL = "BTCUSDT"
INTERVAL = "1s"
HOURS = 48
SECONDS_PER_HOUR = 3600
LIMIT = 1000
DEPTH_LIMIT = 20
SPOT_KLINES_URL = "https://api.binance.com/api/v3/klines"
SPOT_AGGTRADES_URL = "https://api.binance.com/api/v3/aggTrades"
FUTURES_AGGTRADES_URL = "https://fapi.binance.com/fapi/v1/aggTrades"
SPOT_DEPTH_URL = "https://api.binance.com/api/v3/depth"
FUTURES_DEPTH_URL = "https://fapi.binance.com/fapi/v1/depth"
SPOT_BOOK_TICKER_URL = "https://api.binance.com/api/v3/ticker/bookTicker"
FUTURES_BOOK_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/bookTicker"
SPOT_KLINES_ZIP = (
    "https://data.binance.vision/data/spot/daily/klines/"
    "{symbol}/1s/{symbol}-1s-{day}.zip"
)
SPOT_AGGTRADES_ZIP = (
    "https://data.binance.vision/data/spot/daily/aggTrades/"
    "{symbol}/{symbol}-aggTrades-{day}.zip"
)
FUTURES_AGGTRADES_ZIP = (
    "https://data.binance.vision/data/futures/um/daily/aggTrades/"
    "{symbol}/{symbol}-aggTrades-{day}.zip"
)
SPOT_BOOK_TICKER_ZIP = (
    "https://data.binance.vision/data/spot/daily/bookTicker/"
    "{symbol}/{symbol}-bookTicker-{day}.zip"
)
FUTURES_BOOK_TICKER_ZIP = (
    "https://data.binance.vision/data/futures/um/daily/bookTicker/"
    "{symbol}/{symbol}-bookTicker-{day}.zip"
)
FUTURES_BOOK_DEPTH_ZIP = (
    "https://data.binance.vision/data/futures/um/daily/bookDepth/"
    "{symbol}/{symbol}-bookDepth-{day}.zip"
)
DATA_FILENAMES = (
    "btc_spot_candles.csv",
    "btc_futures_candles.csv",
    "btc_spot_trades.csv",
    "btc_futures_trades.csv",
    "btc_spot_book_ticker.csv",
    "btc_futures_book_ticker.csv",
    "btc_spot_depth.csv",
    "btc_futures_depth.csv",
    "btc_futures_book_depth.csv",
    "btc_spot_depth_updates.jsonl",
    "btc_futures_depth_updates.jsonl",
)
SPOT_CANDLES_CSV = DATABASE_DIR / "btc_spot_candles.csv"
FUTURES_CANDLES_CSV = DATABASE_DIR / "btc_futures_candles.csv"
SPOT_TRADES_CSV = DATABASE_DIR / "btc_spot_trades.csv"
FUTURES_TRADES_CSV = DATABASE_DIR / "btc_futures_trades.csv"
SPOT_BOOK_TICKER_CSV = DATABASE_DIR / "btc_spot_book_ticker.csv"
FUTURES_BOOK_TICKER_CSV = DATABASE_DIR / "btc_futures_book_ticker.csv"
SPOT_DEPTH_CSV = DATABASE_DIR / "btc_spot_depth.csv"
FUTURES_DEPTH_CSV = DATABASE_DIR / "btc_futures_depth.csv"
FUTURES_BOOK_DEPTH_CSV = DATABASE_DIR / "btc_futures_book_depth.csv"
SPOT_DEPTH_UPDATES_JSONL = DATABASE_DIR / "btc_spot_depth_updates.jsonl"
FUTURES_DEPTH_UPDATES_JSONL = DATABASE_DIR / "btc_futures_depth_updates.jsonl"
TIMEOUT_S = 30
ZIP_TIMEOUT_S = 300
MAX_RETRIES = 5
HEADERS = {"Accept": "*/*", "User-Agent": "synth-ultra-binance-fetch/1.0"}

CANDLE_FIELDS = (
    "open_time_ms",
    "open_time_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
TRADE_FIELDS = ("agg_trade_id", "ts_ms", "ts_utc", "price", "qty", "buyer_is_maker")
SPOT_BOOK_TICKER_FIELDS = (
    "recv_ts_ms",
    "recv_ts_utc",
    "bid_price",
    "bid_qty",
    "ask_price",
    "ask_qty",
)
FUTURES_BOOK_TICKER_FIELDS = SPOT_BOOK_TICKER_FIELDS + (
    "event_ts_ms",
    "transaction_ts_ms",
    "update_id",
)
DEPTH_SNAPSHOT_FIELDS = (
    "recv_ts_ms",
    "recv_ts_utc",
    "update_id",
    "event_ts_ms",
    "transaction_ts_ms",
    "side",
    "level",
    "price",
    "qty",
)
BOOK_DEPTH_FIELDS = ("ts_ms", "ts_utc", "percentage", "depth", "notional")

Trade = tuple[int, int, float, float, bool]
BookTicker = tuple[int, float, float, float, float, int | None, int | None, int | None]
BookDepthRow = tuple[int, float, float, float]

ALL_DATASETS = (
    "spot-candles",
    "futures-candles",
    "spot-trades",
    "futures-trades",
    "spot-book-ticker",
    "futures-book-ticker",
    "spot-depth",
    "futures-depth",
    "futures-book-depth",
)


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


def _day_bounds_ms(day: datetime) -> tuple[int, int]:
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _iter_utc_day_chunks(start_ms: int, end_ms: int) -> Iterator[tuple[datetime, int, int]]:
    cursor = start_ms
    while cursor < end_ms:
        day = datetime.fromtimestamp(cursor / 1000.0, tz=timezone.utc)
        _, day_end_ms = _day_bounds_ms(day)
        chunk_end = min(day_end_ms, end_ms)
        yield day, cursor, chunk_end
        cursor = chunk_end


def _to_ms(ts: int) -> int:
    if ts >= 10**15:
        return ts // 1000
    return ts


def _looks_like_epoch_ms(value: str) -> bool:
    """True for a Binance millisecond timestamp, not an order-book update id."""
    try:
        ms = _to_ms(int(float(value)))
    except (TypeError, ValueError):
        return False
    return 1_483_228_800_000 <= ms <= 4_102_444_800_000


def ms_to_utc(ms: int) -> str:
    """UTC datetime next to each millisecond timestamp: 2026-08-19 10:40:55.065+00:00."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat(
        sep=" ", timespec="milliseconds"
    )


def parse_book_depth_timestamp(value: str) -> int:
    """Vision bookDepth wall time (`2026-08-19 00:00:06`) as UTC milliseconds."""
    dt = datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _read_vision_zip_bytes(url: str, label: str) -> bytes | None:
    print(f"  {label}  downloading {url}", flush=True)
    try:
        return _http_open(url, timeout=ZIP_TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"  {label}  zip not published (404)", flush=True)
            return None
        raise


def _iter_zip_csv(raw: bytes) -> Iterator[list[str]]:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for name in zf.namelist():
            with zf.open(name) as handle:
                text = io.TextIOWrapper(handle, encoding="utf-8", newline="")
                yield from csv.reader(text)


def _parse_zip_kline(row: list[str]) -> list | None:
    if not row or not row[0].isdigit():
        return None
    open_ms = _to_ms(int(row[0]))
    parsed: list = [open_ms]
    parsed.extend(row[1:])
    if len(parsed) > 6 and str(parsed[6]).isdigit():
        parsed[6] = _to_ms(int(parsed[6]))
    return parsed


def _load_klines_zip(
    day: datetime,
    start_ms: int,
    end_ms: int,
    symbol: str,
    label: str,
) -> list[list] | None:
    url = SPOT_KLINES_ZIP.format(symbol=symbol, day=day.strftime("%Y-%m-%d"))
    raw = _read_vision_zip_bytes(url, label)
    if raw is None:
        return None
    rows: list[list] = []
    for row in _iter_zip_csv(raw):
        parsed = _parse_zip_kline(row)
        if parsed is None:
            continue
        ts = int(parsed[0])
        if ts < start_ms:
            continue
        if ts >= end_ms:
            break
        rows.append(parsed)
    print(f"  {label}  zip {day.date()} klines={len(rows)}", flush=True)
    return rows


def _fetch_spot_1s_klines_rest(
    start_ms: int,
    end_ms: int,
    symbol: str,
    label: str,
) -> list[list]:
    rows: list[list] = []
    cursor = start_ms
    expected = max((end_ms - start_ms) // 1000, 1)
    while cursor < end_ms:
        batch = _http_get_json(
            SPOT_KLINES_URL,
            {
                "symbol": symbol,
                "interval": INTERVAL,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": LIMIT,
            },
        )
        if not batch:
            break
        if rows and int(batch[0][0]) == int(rows[-1][0]):
            batch = batch[1:]
        if not batch:
            break
        for item in batch:
            ts = _to_ms(int(item[0]))
            if ts < start_ms:
                continue
            if ts >= end_ms:
                cursor = end_ms
                break
            item = list(item)
            item[0] = ts
            rows.append(item)
        else:
            cursor = _to_ms(int(batch[-1][0])) + 1000
            print(f"  {label}  rest klines={len(rows)} / {expected}", flush=True)
            if len(batch) < LIMIT:
                break
            continue
        break
    print(f"  {label}  rest total klines={len(rows)}", flush=True)
    return rows


def fetch_spot_1s_klines(
    start_ms: int,
    end_ms: int,
    symbol: str = SYMBOL,
) -> list[list]:
    rows: list[list] = []
    for day, cursor, chunk_end in _iter_utc_day_chunks(start_ms, end_ms):
        zipped = _load_klines_zip(day, cursor, chunk_end, symbol, "Binance spot klines")
        if zipped is None:
            rows.extend(
                _fetch_spot_1s_klines_rest(cursor, chunk_end, symbol, "Binance spot klines")
            )
        else:
            rows.extend(zipped)
    rows.sort(key=lambda row: int(row[0]))
    return rows


def _parse_zip_trade(row: list[str]) -> Trade | None:
    if not row or not row[0].isdigit():
        return None
    maker = row[6].strip().lower() in ("true", "1")
    return int(row[0]), _to_ms(int(row[5])), float(row[1]), float(row[2]), maker


def _load_aggtrades_zip(
    zip_template: str,
    day: datetime,
    start_ms: int,
    end_ms: int,
    symbol: str,
    label: str,
) -> list[Trade] | None:
    url = zip_template.format(symbol=symbol, day=day.strftime("%Y-%m-%d"))
    raw = _read_vision_zip_bytes(url, label)
    if raw is None:
        return None
    trades: list[Trade] = []
    for row in _iter_zip_csv(raw):
        parsed = _parse_zip_trade(row)
        if parsed is None:
            continue
        ts = parsed[1]
        if ts < start_ms:
            continue
        if ts >= end_ms:
            break
        trades.append(parsed)
    print(f"  {label}  zip {day.date()} trades={len(trades)}", flush=True)
    return trades


def _fetch_aggtrades_rest(
    rest_url: str,
    start_ms: int,
    end_ms: int,
    symbol: str,
    label: str,
    sleep_s: float,
) -> list[Trade]:
    trades: list[Trade] = []
    cursor = start_ms
    hour_ms = 60 * 60 * 1000

    while cursor < end_ms:
        window_end = min(cursor + hour_ms, end_ms)
        from_id: int | None = None
        while True:
            params: dict[str, str | int] = {"symbol": symbol, "limit": LIMIT}
            if from_id is None:
                params["startTime"] = cursor
                params["endTime"] = window_end
            else:
                params["fromId"] = from_id
            batch = _http_get_json(rest_url, params)
            if not batch:
                break
            stop = False
            for item in batch:
                ts = int(item["T"])
                if ts < cursor:
                    continue
                if ts >= window_end:
                    stop = True
                    break
                trades.append(
                    (
                        int(item["a"]),
                        ts,
                        float(item["p"]),
                        float(item["q"]),
                        bool(item["m"]),
                    )
                )
            if stop or len(batch) < LIMIT:
                break
            from_id = int(batch[-1]["a"]) + 1
            if sleep_s:
                time.sleep(sleep_s)
            if len(trades) % 50000 < LIMIT:
                print(f"  {label}  rest trades={len(trades)}", flush=True)
        cursor = window_end

    print(f"  {label}  rest total trades={len(trades)}", flush=True)
    return trades


def fetch_aggtrades(
    *,
    zip_template: str,
    rest_url: str,
    label: str,
    start_ms: int,
    end_ms: int,
    sleep_s: float,
    symbol: str = SYMBOL,
) -> list[Trade]:
    trades: list[Trade] = []
    for day, cursor, chunk_end in _iter_utc_day_chunks(start_ms, end_ms):
        zipped = _load_aggtrades_zip(zip_template, day, cursor, chunk_end, symbol, label)
        if zipped is None:
            trades.extend(
                _fetch_aggtrades_rest(rest_url, cursor, chunk_end, symbol, label, sleep_s)
            )
        else:
            trades.extend(zipped)
    trades.sort(key=lambda row: row[1])
    return trades


def _parse_book_ticker_row(row: list[str], *, futures: bool) -> BookTicker | None:
    """Vision bookTicker.

    Futures: update_id, bid, bid_qty, ask, ask_qty, transaction_time, event_time.
    Spot:    update_id, bid, bid_qty, ask, ask_qty, transaction_time.
    Column 0 is an update id. The historical clock is transaction_time on spot
    (the stream has no local receive time) and event_time on futures.
    """
    if not row or not row[0].replace("-", "").replace(".", "", 1).isdigit():
        return None
    if futures and len(row) >= 7 and _looks_like_epoch_ms(row[6]):
        event_ms = _to_ms(int(float(row[6])))
        tx_ms = _to_ms(int(float(row[5])))
        return (
            event_ms,
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            event_ms,
            tx_ms,
            int(float(row[0])),
        )
    if not futures and len(row) >= 6 and _looks_like_epoch_ms(row[5]):
        tx_ms = _to_ms(int(float(row[5])))
        return (
            tx_ms,
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            None,
            None,
            int(float(row[0])),
        )
    if len(row) >= 5 and _looks_like_epoch_ms(row[0]):
        recv = _to_ms(int(float(row[0])))
        return (
            recv,
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            None,
            None,
            None,
        )
    return None


def _parse_book_depth_row(row: list[str]) -> BookDepthRow | None:
    if not row or row[0].strip().lower() == "timestamp":
        return None
    try:
        ts = parse_book_depth_timestamp(row[0]) if " " in row[0] else _to_ms(int(row[0]))
        return ts, float(row[1]), float(row[2]), float(row[3])
    except (TypeError, ValueError, IndexError):
        return None


def _load_book_ticker_zip(
    zip_template: str,
    day: datetime,
    start_ms: int,
    end_ms: int,
    symbol: str,
    label: str,
    *,
    futures: bool,
) -> list[BookTicker] | None:
    url = zip_template.format(symbol=symbol, day=day.strftime("%Y-%m-%d"))
    raw = _read_vision_zip_bytes(url, label)
    if raw is None:
        return None
    rows: list[BookTicker] = []
    for row in _iter_zip_csv(raw):
        parsed = _parse_book_ticker_row(row, futures=futures)
        if parsed is None:
            continue
        ts = parsed[0]
        if ts < start_ms:
            continue
        if ts >= end_ms:
            continue
        rows.append(parsed)
    rows.sort(key=lambda item: item[0])
    print(f"  {label}  zip {day.date()} book_ticker={len(rows)}", flush=True)
    return rows


def _load_book_depth_zip(
    day: datetime,
    start_ms: int,
    end_ms: int,
    symbol: str,
    label: str,
) -> list[BookDepthRow] | None:
    url = FUTURES_BOOK_DEPTH_ZIP.format(symbol=symbol, day=day.strftime("%Y-%m-%d"))
    raw = _read_vision_zip_bytes(url, label)
    if raw is None:
        return None
    rows: list[BookDepthRow] = []
    for row in _iter_zip_csv(raw):
        parsed = _parse_book_depth_row(row)
        if parsed is None:
            continue
        ts = parsed[0]
        if ts < start_ms or ts >= end_ms:
            continue
        rows.append(parsed)
    print(f"  {label}  zip {day.date()} book_depth={len(rows)}", flush=True)
    return rows


def aggtrades_to_1s_ohlcv(
    trades: list[Trade], start_ms: int, n: int, prev_close: float | None = None
) -> list[dict[str, str | int | float]]:
    if not trades and prev_close is None:
        return []
    start_ms = (start_ms // 1000) * 1000
    buckets: dict[int, list[float]] = {}
    for _trade_id, ts, price, qty, _maker in trades:
        open_time = (ts // 1000) * 1000
        candle = buckets.get(open_time)
        if candle is None:
            buckets[open_time] = [price, price, price, price, qty]
        else:
            candle[1] = max(candle[1], price)
            candle[2] = min(candle[2], price)
            candle[3] = price
            candle[4] += qty

    if prev_close is None:
        prev_close = trades[0][2] if trades else 0.0
    out: list[dict[str, str | int | float]] = []
    for i in range(n):
        open_time = start_ms + i * 1000
        candle = buckets.get(open_time)
        if candle is None:
            out.append(
                {
                    "open_time_ms": open_time,
                    "open_time_utc": ms_to_utc(open_time),
                    "open": prev_close,
                    "high": prev_close,
                    "low": prev_close,
                    "close": prev_close,
                    "volume": 0.0,
                }
            )
        else:
            out.append(
                {
                    "open_time_ms": open_time,
                    "open_time_utc": ms_to_utc(open_time),
                    "open": candle[0],
                    "high": candle[1],
                    "low": candle[2],
                    "close": candle[3],
                    "volume": candle[4],
                }
            )
            prev_close = candle[3]
    return out


def klines_to_ohlcv(rows: list[list]) -> list[dict[str, str | int | float]]:
    return [
        {
            "open_time_ms": int(row[0]),
            "open_time_utc": ms_to_utc(int(row[0])),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        }
        for row in rows
    ]


def trades_to_rows(trades: list[Trade]) -> list[dict[str, str | int | float]]:
    return [
        {
            "agg_trade_id": trade_id,
            "ts_ms": ts,
            "ts_utc": ms_to_utc(ts),
            "price": price,
            "qty": qty,
            "buyer_is_maker": buyer_is_maker,
        }
        for trade_id, ts, price, qty, buyer_is_maker in trades
    ]


def write_csv(path: Path, rows: list[dict[str, str | int | float]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def migrate_legacy_data_files(src_dir: Path | None = None, dest_dir: Path | None = None) -> list[str]:
    """Move repo-root CSV/jsonl dumps into database/."""
    src_dir = src_dir or REPO_ROOT
    dest_dir = dest_dir or DATABASE_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for name in DATA_FILENAMES:
        src = src_dir / name
        dest = dest_dir / name
        if not src.is_file():
            continue
        if src.resolve() == dest.resolve():
            continue
        if dest.is_file():
            print(f"Keeping {dest} (leftover {src})", flush=True)
            continue
        src.rename(dest)
        moved.append(name)
        print(f"Moved {name} -> {dest}", flush=True)
    return moved


def _csv_field_index(path: Path, field: str) -> tuple[list[str], int] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        header = handle.readline()
    if not header.strip():
        return None
    names = next(csv.reader([header]))
    if field not in names:
        return None
    return names, names.index(field)


def csv_first_ms(path: Path, field: str) -> int | None:
    parsed = _csv_field_index(path, field)
    if parsed is None:
        return None
    _names, idx = parsed
    with path.open(newline="", encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            if not line.strip():
                continue
            row = next(csv.reader([line]))
            if len(row) <= idx or row[idx] == "":
                return None
            try:
                return int(float(row[idx]))
            except ValueError:
                return None
    return None


def csv_last_ms(path: Path, field: str) -> int | None:
    parsed = _csv_field_index(path, field)
    if parsed is None:
        return None
    names, idx = parsed
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        data = b""
        while size > 0 and data.count(b"\n") < 4:
            size = max(0, size - 8192)
            handle.seek(size)
            data = handle.read()
    lines = [ln for ln in data.decode("utf-8", errors="replace").splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1]
    row = next(csv.reader([last]))
    if not row or row[0] == names[0]:
        return None
    if len(row) <= idx or row[idx] == "":
        return None
    try:
        return int(float(row[idx]))
    except ValueError:
        return None


def csv_data_row_count(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    with path.open(encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def dataset_windows(
    path: Path,
    field: str,
    default_start: int,
    end_ms: int,
    step: int,
    *,
    rebuild: bool = False,
) -> list[tuple[int, int]]:
    """Forward-fill after last_ts and backfill so coverage reaches default_start."""
    if rebuild:
        return [(default_start, end_ms)]
    first = csv_first_ms(path, field)
    last = csv_last_ms(path, field)
    if first is None or last is None:
        return [(default_start, end_ms)]
    windows: list[tuple[int, int]] = []
    if last + step < end_ms:
        windows.append((last + step, end_ms))
    if first > default_start:
        windows.append((default_start, first))
    return windows


def append_csv_rows(
    path: Path, rows: list[dict[str, str | int | float]], fields: tuple[str, ...]
) -> None:
    if not rows:
        return
    handle, writer = _open_csv(path, fields, append=True)
    try:
        writer.writerows(rows)
    finally:
        handle.close()


def prepend_csv_rows(
    path: Path, rows: list[dict[str, str | int | float]], fields: tuple[str, ...]
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        if path.is_file() and path.stat().st_size > 0:
            with path.open(newline="", encoding="utf-8") as inp:
                inp.readline()
                while True:
                    chunk = inp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
    tmp.replace(path)


def ingest_rows(
    path: Path,
    rows: list[dict[str, str | int | float]],
    fields: tuple[str, ...],
    ts_field: str,
    *,
    fill_gaps: bool = False,
) -> int:
    """Insert rows before and/or after existing CSV timestamps. Returns kept count."""
    if not rows:
        return 0
    rows = sorted(rows, key=lambda row: int(row[ts_field]))
    first = csv_first_ms(path, ts_field)
    last = csv_last_ms(path, ts_field)
    if first is None or last is None:
        write_csv(path, rows, fields)
        return len(rows)
    if fill_gaps:
        return _merge_csv_rows(path, rows, fields, ts_field)
    before = [row for row in rows if int(row[ts_field]) < first]
    after = [row for row in rows if int(row[ts_field]) > last]
    if before:
        prepend_csv_rows(path, before, fields)
    if after:
        append_csv_rows(path, after, fields)
    return len(before) + len(after)


def _merge_csv_rows(
    path: Path,
    new_rows: list[dict[str, str | int | float]],
    fields: tuple[str, ...],
    ts_field: str,
) -> int:
    """Merge new timestamped rows into an existing sorted CSV, skipping duplicates."""
    if not new_rows:
        return 0
    new_rows = sorted(new_rows, key=lambda row: int(row[ts_field]))
    tmp = path.with_name(path.name + ".tmp")
    kept = 0
    new_i = 0
    with tmp.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        with path.open(newline="", encoding="utf-8") as inp:
            for row in csv.DictReader(inp):
                ts = int(float(row[ts_field]))
                while new_i < len(new_rows) and int(new_rows[new_i][ts_field]) < ts:
                    writer.writerow(new_rows[new_i])
                    new_i += 1
                    kept += 1
                while new_i < len(new_rows) and int(new_rows[new_i][ts_field]) == ts:
                    new_i += 1
                writer.writerow({name: row.get(name, "") for name in fields})
        while new_i < len(new_rows):
            writer.writerow(new_rows[new_i])
            new_i += 1
            kept += 1
    tmp.replace(path)
    return kept


def iter_csv_in_range(
    path: Path, ts_field: str, start_ms: int, end_ms: int
) -> Iterator[dict[str, str]]:
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts = int(float(row[ts_field]))
            if ts < start_ms:
                continue
            if ts >= end_ms:
                break
            yield row


def rest_book_ticker_row(*, futures: bool) -> dict[str, str | int | float]:
    url = FUTURES_BOOK_TICKER_URL if futures else SPOT_BOOK_TICKER_URL
    payload = _http_get_json(url, {"symbol": SYMBOL})
    recv = int(time.time() * 1000)
    row: dict[str, str | int | float] = {
        "recv_ts_ms": recv,
        "recv_ts_utc": ms_to_utc(recv),
        "bid_price": float(payload["bidPrice"]),
        "bid_qty": float(payload["bidQty"]),
        "ask_price": float(payload["askPrice"]),
        "ask_qty": float(payload["askQty"]),
    }
    if futures:
        tx = int(payload["time"]) if payload.get("time") is not None else ""
        row["event_ts_ms"] = tx
        row["transaction_ts_ms"] = tx
        row["update_id"] = int(payload["lastUpdateId"]) if payload.get("lastUpdateId") is not None else ""
    return row


def book_ticker_from_candle(
    candle: dict[str, str],
    template: dict[str, str | int | float],
    *,
    futures: bool,
) -> dict[str, str | int | float]:
    """Spread a candle close into a bid/ask pair. Not a Binance bookTicker row."""
    ts = int(float(candle["open_time_ms"]))
    close = float(candle["close"])
    bid = float(template["bid_price"])
    ask = float(template["ask_price"])
    half = max((ask - bid) / 2.0, 0.005)
    row: dict[str, str | int | float] = {
        "recv_ts_ms": ts,
        "recv_ts_utc": ms_to_utc(ts),
        "bid_price": close - half,
        "bid_qty": float(template["bid_qty"]),
        "ask_price": close + half,
        "ask_qty": float(template["ask_qty"]),
    }
    if futures:
        row["event_ts_ms"] = ts
        row["transaction_ts_ms"] = ts
        row["update_id"] = ts
    return row


def percent_levels_to_book(
    levels: list[tuple[float, float, float]], max_levels: int = DEPTH_LIMIT
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    bids_i = [item for item in levels if item[0] < 0.0]
    asks_i = [item for item in levels if item[0] > 0.0]
    if not bids_i or not asks_i:
        return [], []
    inner_b = min(bids_i, key=lambda item: abs(item[0]))
    inner_a = min(asks_i, key=lambda item: abs(item[0]))
    vwap_b = inner_b[2] / max(inner_b[1], 1e-12)
    vwap_a = inner_a[2] / max(inner_a[1], 1e-12)
    mid = 0.5 * (vwap_b + vwap_a)

    def side(items: list[tuple[float, float, float]]) -> list[tuple[float, float]]:
        ordered = sorted(items, key=lambda item: abs(item[0]))
        out: list[tuple[float, float]] = []
        prev = 0.0
        for pct, depth, _notional in ordered:
            qty = max(depth - prev, 0.0)
            prev = depth
            out.append((mid * (1.0 + pct / 100.0), qty))
            if len(out) >= max_levels:
                break
        return out

    return side(bids_i), side(asks_i)


def depth_rows_from_book(
    ts: int,
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    *,
    futures: bool,
    update_id: int | str = "",
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    uid = update_id if update_id != "" else ts
    event = ts if futures else ""
    tx = ts if futures else ""
    for i, (price, qty) in enumerate(bids):
        rows.append(
            {
                "recv_ts_ms": ts,
                "recv_ts_utc": ms_to_utc(ts),
                "update_id": uid,
                "event_ts_ms": event,
                "transaction_ts_ms": tx,
                "side": "bid",
                "level": i,
                "price": price,
                "qty": qty,
            }
        )
    for i, (price, qty) in enumerate(asks):
        rows.append(
            {
                "recv_ts_ms": ts,
                "recv_ts_utc": ms_to_utc(ts),
                "update_id": uid,
                "event_ts_ms": event,
                "transaction_ts_ms": tx,
                "side": "ask",
                "level": i,
                "price": price,
                "qty": qty,
            }
        )
    return rows


def _print_candles(path: Path, candles: list[dict[str, str | int | float]]) -> None:
    first = candles[0]
    last = candles[-1]
    print(
        f"Wrote {len(candles)} rows to {path.resolve()}\n"
        f"  first open_time_ms={first['open_time_ms']}  utc={first['open_time_utc']}  OHLCV="
        f"{first['open']},{first['high']},{first['low']},{first['close']},{first['volume']}\n"
        f"  last  open_time_ms={last['open_time_ms']}  utc={last['open_time_utc']}  OHLCV="
        f"{last['open']},{last['high']},{last['low']},{last['close']},{last['volume']}"
    )


def _print_trades(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    first = rows[0]
    last = rows[-1]
    print(
        f"Wrote {len(rows)} rows to {path.resolve()}\n"
        f"  first ts_ms={first['ts_ms']}  utc={first['ts_utc']}  id={first['agg_trade_id']}  "
        f"price={first['price']} qty={first['qty']} maker={first['buyer_is_maker']}\n"
        f"  last  ts_ms={last['ts_ms']}  utc={last['ts_utc']}  id={last['agg_trade_id']}  "
        f"price={last['price']} qty={last['qty']} maker={last['buyer_is_maker']}"
    )


def fetch_spot_candles(start_ms: int, end_ms: int) -> int:
    windows = dataset_windows(SPOT_CANDLES_CSV, "open_time_ms", start_ms, end_ms, 1000)
    if not windows:
        print(f"Spot candles already cover the window ({SPOT_CANDLES_CSV})", flush=True)
        return 0
    candles: list[dict[str, str | int | float]] = []
    status = 0
    for w_start, w_end in windows:
        hours = max((w_end - w_start) / 3_600_000, 0.0)
        print(f"Fetching {hours:.2f}h of {SYMBOL} {INTERVAL} klines from Binance spot...")
        spot_rows = fetch_spot_1s_klines(w_start, w_end)
        if not spot_rows:
            print(f"No spot klines for {ms_to_utc(w_start)} -> {ms_to_utc(w_end)}", file=sys.stderr)
            status = 1
            continue
        chunk = klines_to_ohlcv(spot_rows)
        ingest_rows(SPOT_CANDLES_CSV, chunk, CANDLE_FIELDS, "open_time_ms")
        candles.extend(chunk)
    if not candles:
        print("No spot klines returned.", file=sys.stderr)
        return 1
    print(f"Ingested {len(candles)} spot candles into {SPOT_CANDLES_CSV.resolve()}", flush=True)
    _print_candles(SPOT_CANDLES_CSV, candles)
    return status


def load_futures_aggtrades(start_ms: int, end_ms: int) -> list[Trade]:
    hours = max((end_ms - start_ms) / 3_600_000, 0.0)
    print(f"Fetching {hours:.2f}h of {SYMBOL} aggTrades from Binance USDT-M futures...")
    return fetch_aggtrades(
        zip_template=FUTURES_AGGTRADES_ZIP,
        rest_url=FUTURES_AGGTRADES_URL,
        label="USDT-M futures",
        start_ms=start_ms,
        end_ms=end_ms,
        sleep_s=0.51,
    )


def _trades_from_csv(path: Path, start_ms: int, end_ms: int) -> list[Trade]:
    trades: list[Trade] = []
    for row in iter_csv_in_range(path, "ts_ms", start_ms, end_ms):
        maker = str(row["buyer_is_maker"]).strip().lower() in ("true", "1")
        trades.append(
            (
                int(float(row["agg_trade_id"])),
                int(float(row["ts_ms"])),
                float(row["price"]),
                float(row["qty"]),
                maker,
            )
        )
    return trades


def fetch_futures_candles(
    start_ms: int, end_ms: int, trades: list[Trade] | None = None
) -> int:
    windows = dataset_windows(FUTURES_CANDLES_CSV, "open_time_ms", start_ms, end_ms, 1000)
    if not windows:
        print(f"Futures candles already cover the window ({FUTURES_CANDLES_CSV})", flush=True)
        return 0
    extra = trades or []
    candles: list[dict[str, str | int | float]] = []
    prev_close = None
    last_existing = csv_last_ms(FUTURES_CANDLES_CSV, "open_time_ms")
    if last_existing is not None:
        # last close from the existing file's last row
        parsed = _csv_field_index(FUTURES_CANDLES_CSV, "close")
        if parsed is not None:
            names, idx = parsed
            with FUTURES_CANDLES_CSV.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 1024))
                last_line = [
                    ln
                    for ln in handle.read().decode("utf-8", errors="replace").splitlines()
                    if ln.strip()
                ][-1]
            row = next(csv.reader([last_line]))
            if len(row) > idx and row[0] != names[0]:
                prev_close = float(row[idx])
    for w_start, w_end in windows:
        n = max((w_end - w_start) // 1000, 1)
        hours = n / SECONDS_PER_HOUR
        window_trades = [t for t in extra if w_start <= t[1] < w_end]
        if not window_trades:
            window_trades = _trades_from_csv(FUTURES_TRADES_CSV, w_start, w_end)
        if not window_trades:
            print(f"Building {hours:.2f}h of {SYMBOL} 1s OHLCV from USDT-M futures aggTrades...")
            window_trades = load_futures_aggtrades(w_start, w_end)
        else:
            print(f"Building {hours:.2f}h of {SYMBOL} 1s OHLCV from USDT-M futures aggTrades...")
        chunk = aggtrades_to_1s_ohlcv(window_trades, w_start, n, prev_close=prev_close)
        if chunk:
            prev_close = float(chunk[-1]["close"])
            candles.extend(chunk)
    if not candles:
        print("No futures candles returned.", file=sys.stderr)
        return 1
    kept = ingest_rows(FUTURES_CANDLES_CSV, candles, CANDLE_FIELDS, "open_time_ms")
    print(f"Ingested {kept} futures candles into {FUTURES_CANDLES_CSV.resolve()}", flush=True)
    _print_candles(FUTURES_CANDLES_CSV, candles)
    return 0


def fetch_spot_trades(start_ms: int, end_ms: int) -> int:
    windows = dataset_windows(SPOT_TRADES_CSV, "ts_ms", start_ms, end_ms, 1)
    if not windows:
        print(f"Spot trades already cover the window ({SPOT_TRADES_CSV})", flush=True)
        return 0
    rows: list[dict[str, str | int | float]] = []
    status = 0
    for w_start, w_end in windows:
        hours = max((w_end - w_start) / 3_600_000, 0.0)
        print(f"Fetching {hours:.2f}h of {SYMBOL} trades from Binance spot...")
        trades = fetch_aggtrades(
            zip_template=SPOT_AGGTRADES_ZIP,
            rest_url=SPOT_AGGTRADES_URL,
            label="Binance spot trades",
            start_ms=w_start,
            end_ms=w_end,
            sleep_s=0.05,
        )
        if not trades:
            print(f"No spot trades for {ms_to_utc(w_start)} -> {ms_to_utc(w_end)}", file=sys.stderr)
            status = 1
            continue
        chunk = trades_to_rows(trades)
        ingest_rows(SPOT_TRADES_CSV, chunk, TRADE_FIELDS, "ts_ms")
        rows.extend(chunk)
    if not rows:
        print("No spot trades returned.", file=sys.stderr)
        return 1
    print(f"Ingested {len(rows)} spot trades into {SPOT_TRADES_CSV.resolve()}", flush=True)
    _print_trades(SPOT_TRADES_CSV, rows)
    return status


def fetch_futures_trades(start_ms: int, end_ms: int, trades: list[Trade] | None = None) -> int:
    windows = dataset_windows(FUTURES_TRADES_CSV, "ts_ms", start_ms, end_ms, 1)
    if not windows:
        print(f"Futures trades already cover the window ({FUTURES_TRADES_CSV})", flush=True)
        return 0
    collected: list[Trade] = []
    extra = trades or []
    status = 0
    for w_start, w_end in windows:
        window_trades = [t for t in extra if w_start <= t[1] < w_end]
        if not window_trades:
            window_trades = load_futures_aggtrades(w_start, w_end)
        if not window_trades:
            print(f"No futures trades for {ms_to_utc(w_start)} -> {ms_to_utc(w_end)}", file=sys.stderr)
            status = 1
            continue
        rows = trades_to_rows(window_trades)
        ingest_rows(FUTURES_TRADES_CSV, rows, TRADE_FIELDS, "ts_ms")
        collected.extend(window_trades)
        print(f"Ingested {len(rows)} futures trades into {FUTURES_TRADES_CSV.resolve()}", flush=True)
    if not collected:
        print("No futures trades returned.", file=sys.stderr)
        return 1
    _print_trades(FUTURES_TRADES_CSV, trades_to_rows(collected[:1] + collected[-1:]))
    return status


def _book_ticker_rows_from_zip(
    zip_template: str,
    start_ms: int,
    end_ms: int,
    label: str,
    *,
    futures: bool,
) -> list[BookTicker]:
    rows: list[BookTicker] = []
    for day, cursor, chunk_end in _iter_utc_day_chunks(start_ms, end_ms):
        zipped = _load_book_ticker_zip(
            zip_template, day, cursor, chunk_end, SYMBOL, label, futures=futures
        )
        if zipped:
            rows.extend(zipped)
    return rows


def fetch_book_ticker_series(start_ms: int, end_ms: int, *, futures: bool) -> int:
    """Store real bookTicker rows only.

    A candle close plus a copied spread is not the best bid/ask stream, and
    scoring reads this file as the spot microprice. Vision is used when the
    daily zip exists. One REST snapshot is a real quote at request time.
    """
    path = FUTURES_BOOK_TICKER_CSV if futures else SPOT_BOOK_TICKER_CSV
    fields = FUTURES_BOOK_TICKER_FIELDS if futures else SPOT_BOOK_TICKER_FIELDS
    label = "USDT-M futures" if futures else "spot"
    zip_template = FUTURES_BOOK_TICKER_ZIP if futures else SPOT_BOOK_TICKER_ZIP
    print(f"Fetching {label} bookTicker {ms_to_utc(start_ms)} -> {ms_to_utc(end_ms)}", flush=True)
    rows: list[dict[str, str | int | float]] = []
    zipped = _book_ticker_rows_from_zip(
        zip_template, start_ms, end_ms, f"{label} bookTicker", futures=futures
    )
    if zipped:
        for recv, bid_p, bid_q, ask_p, ask_q, event, tx, uid in zipped:
            row: dict[str, str | int | float] = {
                "recv_ts_ms": recv,
                "recv_ts_utc": ms_to_utc(recv),
                "bid_price": bid_p,
                "bid_qty": bid_q,
                "ask_price": ask_p,
                "ask_qty": ask_q,
            }
            if futures:
                row["event_ts_ms"] = event if event is not None else ""
                row["transaction_ts_ms"] = tx if tx is not None else ""
                row["update_id"] = uid if uid is not None else ""
            rows.append(row)
    try:
        rows.append(rest_book_ticker_row(futures=futures))
    except (OSError, TimeoutError, RuntimeError, KeyError, TypeError, ValueError) as exc:
        print(f"  {label} REST bookTicker snapshot failed: {exc}", flush=True)
    if not rows:
        print(
            f"No {label} bookTicker rows. Record the live stream with --live-seconds.",
            file=sys.stderr,
        )
        return 1
    kept = ingest_rows(path, rows, fields, "recv_ts_ms", fill_gaps=True)
    print(f"Ingested {kept} {label} bookTicker rows into {path.resolve()}", flush=True)
    return 0


def fetch_futures_book_ticker(start_ms: int, end_ms: int) -> int:
    return fetch_book_ticker_series(start_ms, end_ms, futures=True)


def fetch_spot_book_ticker(start_ms: int, end_ms: int) -> int:
    return fetch_book_ticker_series(start_ms, end_ms, futures=False)


def fetch_futures_book_depth(start_ms: int, end_ms: int) -> int:
    windows = dataset_windows(FUTURES_BOOK_DEPTH_CSV, "ts_ms", start_ms, end_ms, 1)
    if not windows:
        print(f"Futures bookDepth already covers the window ({FUTURES_BOOK_DEPTH_CSV})", flush=True)
        return 0
    rows: list[dict[str, str | int | float]] = []
    for w_start, w_end in windows:
        print(f"Fetching {SYMBOL} USDT-M bookDepth {ms_to_utc(w_start)} -> {ms_to_utc(w_end)}...")
        for day, cursor, chunk_end in _iter_utc_day_chunks(w_start, w_end):
            zipped = _load_book_depth_zip(day, cursor, chunk_end, SYMBOL, "USDT-M bookDepth")
            if not zipped:
                continue
            for ts, pct, depth, notional in zipped:
                rows.append(
                    {
                        "ts_ms": ts,
                        "ts_utc": ms_to_utc(ts),
                        "percentage": pct,
                        "depth": depth,
                        "notional": notional,
                    }
                )
    if not rows:
        if FUTURES_BOOK_DEPTH_CSV.is_file() and csv_data_row_count(FUTURES_BOOK_DEPTH_CSV) > 0:
            print("No new futures bookDepth for this window.", flush=True)
            return 0
        print("No futures bookDepth returned.", file=sys.stderr)
        return 1
    kept = ingest_rows(FUTURES_BOOK_DEPTH_CSV, rows, BOOK_DEPTH_FIELDS, "ts_ms")
    print(f"Ingested {kept} bookDepth rows into {FUTURES_BOOK_DEPTH_CSV.resolve()}", flush=True)
    return 0


def fetch_depth_series(start_ms: int, end_ms: int, *, futures: bool) -> int:
    """Store one real REST depth snapshot.

    Replaying that shape onto old candle closes, or turning bookDepth percent
    bins into prices, is not the order book in input.md. Incremental diffs
    come from ``--live-seconds``. ``start_ms``/``end_ms`` bound the other
    datasets; REST depth has no historical query.
    """
    del start_ms, end_ms
    path = FUTURES_DEPTH_CSV if futures else SPOT_DEPTH_CSV
    label = "USDT-M futures" if futures else "spot"
    url = FUTURES_DEPTH_URL if futures else SPOT_DEPTH_URL
    print(f"Fetching {SYMBOL} {label} REST depth ({DEPTH_LIMIT} levels)...")
    payload = _http_get_json(url, {"symbol": SYMBOL, "limit": DEPTH_LIMIT})
    live_rows = _depth_snapshot_rows(payload, futures=futures)
    if not live_rows:
        print(f"No {label} depth returned.", file=sys.stderr)
        return 1
    kept = ingest_rows(path, live_rows, DEPTH_SNAPSHOT_FIELDS, "recv_ts_ms")
    print(f"Ingested {kept} {label} depth rows into {path.resolve()}", flush=True)
    best_bid = next(r for r in live_rows if r["side"] == "bid" and r["level"] == 0)
    best_ask = next(r for r in live_rows if r["side"] == "ask" and r["level"] == 0)
    print(
        f"  live snapshot recv={live_rows[0]['recv_ts_utc']}  "
        f"bid={best_bid['price']} x {best_bid['qty']}  "
        f"ask={best_ask['price']} x {best_ask['qty']}",
        flush=True,
    )
    print(f"  {label} depth_updates are recorded only by --live-seconds", flush=True)
    return 0


def _depth_snapshot_rows(payload: dict, *, futures: bool) -> list[dict[str, str | int | float]]:
    recv = int(time.time() * 1000)
    update_id = int(payload.get("lastUpdateId") or 0)
    event = int(payload["E"]) if futures and payload.get("E") is not None else ""
    tx = int(payload["T"]) if futures and payload.get("T") is not None else ""
    rows: list[dict[str, str | int | float]] = []
    for side in ("bids", "asks"):
        levels = payload.get(side) or []
        for i, level in enumerate(levels):
            rows.append(
                {
                    "recv_ts_ms": recv,
                    "recv_ts_utc": ms_to_utc(recv),
                    "update_id": update_id,
                    "event_ts_ms": event,
                    "transaction_ts_ms": tx,
                    "side": side[:-1],
                    "level": i,
                    "price": float(level[0]),
                    "qty": float(level[1]),
                }
            )
    return rows


def fetch_depth_snapshot(*, futures: bool, append: bool = True) -> int:
    label = "USDT-M futures" if futures else "spot"
    url = FUTURES_DEPTH_URL if futures else SPOT_DEPTH_URL
    path = FUTURES_DEPTH_CSV if futures else SPOT_DEPTH_CSV
    print(f"Fetching {SYMBOL} {label} REST depth ({DEPTH_LIMIT} levels)...")
    payload = _http_get_json(url, {"symbol": SYMBOL, "limit": DEPTH_LIMIT})
    rows = _depth_snapshot_rows(payload, futures=futures)
    if not rows:
        print(f"No {label} depth returned.", file=sys.stderr)
        return 1
    ingest_rows(path, rows, DEPTH_SNAPSHOT_FIELDS, "recv_ts_ms")
    best_bid = next(r for r in rows if r["side"] == "bid" and r["level"] == 0)
    best_ask = next(r for r in rows if r["side"] == "ask" and r["level"] == 0)
    print(
        f"Wrote {len(rows)} rows to {path.resolve()}\n"
        f"  update_id={rows[0]['update_id']}  recv={rows[0]['recv_ts_utc']}  "
        f"bid={best_bid['price']} x {best_bid['qty']}  "
        f"ask={best_ask['price']} x {best_ask['qty']}",
        flush=True,
    )
    return 0


def fetch_book_ticker_snapshot(*, futures: bool, append: bool = True) -> int:
    label = "USDT-M futures" if futures else "spot"
    path = FUTURES_BOOK_TICKER_CSV if futures else SPOT_BOOK_TICKER_CSV
    fields = FUTURES_BOOK_TICKER_FIELDS if futures else SPOT_BOOK_TICKER_FIELDS
    print(f"Fetching {SYMBOL} {label} REST bookTicker snapshot...")
    row = rest_book_ticker_row(futures=futures)
    last = csv_last_ms(path, "recv_ts_ms")
    if last is not None and int(row["recv_ts_ms"]) <= last:
        print(f"  {label} bookTicker snapshot not newer than CSV", flush=True)
        return 0
    ingest_rows(path, [row], fields, "recv_ts_ms")
    print(
        f"Wrote 1 row to {path.resolve()}  "
        f"bid={row['bid_price']} ask={row['ask_price']}  utc={row['recv_ts_utc']}",
        flush=True,
    )
    return 0


def _ws_connect(host: str, path: str, port: int) -> ssl.SSLSocket:
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    raw = __import__("socket").create_connection((host, port), timeout=30)
    ctx = ssl.create_default_context()
    sock = ctx.wrap_socket(raw, server_hostname=host)
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


def _open_csv(path: Path, fields: tuple[str, ...], append: bool) -> tuple[TextIO, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = append and path.is_file() and path.stat().st_size > 0
    handle = path.open("a" if exists else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fields)
    if not exists:
        writer.writeheader()
    return handle, writer


def _handle_ws_payload(
    venue: str,
    stream: str,
    data: dict,
    recv_ms: int,
    bt_writers: dict[str, csv.DictWriter],
    depth_files: dict[str, TextIO],
    counts: dict[str, int],
    prev_ids: dict[str, int | None],
) -> None:
    name = stream.split("@", 1)[-1] if stream else ""
    is_book = (
        name == "bookTicker"
        or stream.endswith("@bookTicker")
        or ("b" in data and "B" in data and "a" in data and "A" in data and "U" not in data)
    )
    is_depth = "depth" in name or data.get("e") == "depthUpdate" or "U" in data
    if is_book and "U" not in data:
        row: dict[str, str | int | float] = {
            "recv_ts_ms": recv_ms,
            "recv_ts_utc": ms_to_utc(recv_ms),
            "bid_price": float(data["b"]),
            "bid_qty": float(data["B"]),
            "ask_price": float(data["a"]),
            "ask_qty": float(data["A"]),
        }
        if venue == "futures":
            row["event_ts_ms"] = int(data["E"]) if data.get("E") is not None else ""
            row["transaction_ts_ms"] = int(data["T"]) if data.get("T") is not None else ""
            row["update_id"] = int(data["u"]) if data.get("u") is not None else ""
        bt_writers[venue].writerow(row)
        counts[f"{venue}-book"] += 1
        return
    if is_depth:
        rec = {
            "recv_ts_ms": recv_ms,
            "event_ts_ms": int(data["E"]) if data.get("E") is not None else recv_ms,
            "first_id": int(data["U"]),
            "final_id": int(data["u"]),
            "prev_final_id": (
                int(data["pu"])
                if data.get("pu") is not None
                else (prev_ids[venue] if prev_ids.get(venue) is not None else int(data["U"]) - 1)
            ),
            "bids": [[float(p), float(q)] for p, q in data.get("b") or []],
            "asks": [[float(p), float(q)] for p, q in data.get("a") or []],
        }
        if venue == "futures" and data.get("T") is not None:
            rec["transaction_ts_ms"] = int(data["T"])
        depth_files[venue].write(json.dumps(rec) + "\n")
        counts[f"{venue}-depth"] += 1
        prev_ids[venue] = int(data["u"])


def record_live_streams(seconds: float) -> int:
    """Record schema book_ticker + depth_updates from Binance combined streams."""
    streams = "btcusdt@bookTicker/btcusdt@depth@100ms"
    endpoints = {
        "spot": ("stream.binance.com", 9443, f"/stream?streams={streams}"),
        "futures": ("fstream.binance.com", 443, f"/stream?streams={streams}"),
    }
    print(f"Recording live bookTicker + depth@100ms for {seconds:.0f}s ...", flush=True)
    socks: dict[str, ssl.SSLSocket] = {}
    for venue, (host, port, path) in endpoints.items():
        try:
            socks[venue] = _ws_connect(host, path, port)
            print(f"  ws {venue} connected {host}", flush=True)
        except Exception as exc:
            print(f"  ws {venue} failed: {exc}", file=sys.stderr)

    bt_handles: dict[str, TextIO] = {}
    bt_writers: dict[str, csv.DictWriter] = {}
    depth_files: dict[str, TextIO] = {}
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    spot_h, spot_w = _open_csv(SPOT_BOOK_TICKER_CSV, SPOT_BOOK_TICKER_FIELDS, append=True)
    fut_h, fut_w = _open_csv(FUTURES_BOOK_TICKER_CSV, FUTURES_BOOK_TICKER_FIELDS, append=True)
    bt_handles["spot"], bt_writers["spot"] = spot_h, spot_w
    bt_handles["futures"], bt_writers["futures"] = fut_h, fut_w
    depth_files["spot"] = SPOT_DEPTH_UPDATES_JSONL.open("a", encoding="utf-8")
    depth_files["futures"] = FUTURES_DEPTH_UPDATES_JSONL.open("a", encoding="utf-8")
    counts = {"spot-book": 0, "futures-book": 0, "spot-depth": 0, "futures-depth": 0}
    prev_ids: dict[str, int | None] = {"spot": None, "futures": None}
    deadline = time.time() + seconds
    last_snap = time.time()
    try:
        while time.time() < deadline and socks:
            if time.time() - last_snap >= 60.0:
                fetch_depth_snapshot(futures=False, append=True)
                fetch_depth_snapshot(futures=True, append=True)
                last_snap = time.time()
            readable, _, _ = select.select(list(socks.values()), [], [], 1.0)
            recv_ms = int(time.time() * 1000)
            for sock in readable:
                venue = next(name for name, s in socks.items() if s is sock)
                try:
                    text = _ws_recv_message(sock)
                except (ConnectionError, TimeoutError, OSError) as exc:
                    print(f"  ws {venue} dropped: {exc}", flush=True)
                    try:
                        sock.close()
                    except OSError:
                        pass
                    host, port, path = endpoints[venue]
                    try:
                        socks[venue] = _ws_connect(host, path, port)
                        print(f"  ws {venue} reconnected", flush=True)
                    except Exception as rec_exc:
                        print(f"  ws {venue} reconnect failed: {rec_exc}", file=sys.stderr)
                        socks.pop(venue, None)
                    continue
                if text is None:
                    socks.pop(venue, None)
                    continue
                msg = json.loads(text)
                data = msg.get("data") if isinstance(msg.get("data"), dict) else msg
                stream = str(msg.get("stream") or "")
                _handle_ws_payload(
                    venue, stream, data, recv_ms, bt_writers, depth_files, counts, prev_ids
                )
            if sum(counts.values()) and sum(counts.values()) % 5000 < 8:
                print(
                    f"  live  spot_bt={counts['spot-book']} fut_bt={counts['futures-book']} "
                    f"spot_d={counts['spot-depth']} fut_d={counts['futures-depth']}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("Live recording interrupted.", flush=True)
    finally:
        for sock in socks.values():
            try:
                sock.close()
            except OSError:
                pass
        for handle in (*bt_handles.values(), *depth_files.values()):
            handle.close()
    print(
        f"Live captured  spot bookTicker={counts['spot-book']}  "
        f"futures bookTicker={counts['futures-book']}  "
        f"spot depth={counts['spot-depth']}  futures depth={counts['futures-depth']}",
        flush=True,
    )
    return 0


def parse_day_arg(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch 48h+ of Binance BTCUSDT data into database/ for Synth Ultra payloads"
    )
    parser.add_argument(
        "--only",
        choices=ALL_DATASETS,
        action="append",
        help="fetch only these datasets (repeatable); default is all",
    )
    parser.add_argument(
        "--day",
        metavar="YYYY-MM-DD",
        help="UTC calendar day (00:00–24:00). Default is trailing --hours.",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=HOURS,
        help=f"trailing hours when --day is omitted (default {HOURS}, more than 1 day)",
    )
    parser.add_argument(
        "--live-seconds",
        type=float,
        default=0.0,
        help="also record live bookTicker + depth@100ms for this many seconds "
        "(86400 for a full day; Vision no longer publishes recent L2 ticks)",
    )
    args = parser.parse_args(argv)
    wanted = set(args.only) if args.only else set(ALL_DATASETS)
    migrate_legacy_data_files()
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    if args.day:
        start_ms, end_ms = _day_bounds_ms(parse_day_arg(args.day))
    else:
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - int(args.hours * SECONDS_PER_HOUR * 1000)
    print(
        f"Window  start={ms_to_utc(start_ms)}  end={ms_to_utc(end_ms)}  "
        f"({(end_ms - start_ms) / 3_600_000:.2f}h)  dir={DATABASE_DIR}",
        flush=True,
    )

    status = 0
    if "spot-candles" in wanted:
        status |= fetch_spot_candles(start_ms, end_ms)
    if "futures-book-depth" in wanted:
        status |= fetch_futures_book_depth(start_ms, end_ms)
    if "spot-book-ticker" in wanted:
        status |= fetch_spot_book_ticker(start_ms, end_ms)
    if "futures-book-ticker" in wanted:
        status |= fetch_futures_book_ticker(start_ms, end_ms)
    if "spot-depth" in wanted:
        status |= fetch_depth_series(start_ms, end_ms, futures=False)
    if "futures-depth" in wanted:
        status |= fetch_depth_series(start_ms, end_ms, futures=True)
    if "spot-trades" in wanted:
        status |= fetch_spot_trades(start_ms, end_ms)
    if "futures-trades" in wanted:
        status |= fetch_futures_trades(start_ms, end_ms)
    if "futures-candles" in wanted:
        status |= fetch_futures_candles(start_ms, end_ms)
    if args.live_seconds > 0:
        status |= record_live_streams(args.live_seconds)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
