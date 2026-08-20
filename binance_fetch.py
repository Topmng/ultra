"""Fetch Binance BTCUSDT market data for the trailing 24 hours.

Writes:
  btc_spot_candles.csv      spot 1s OHLCV klines
  btc_futures_candles.csv   USDT-M futures 1s OHLCV (from aggTrades; fapi has no 1s kline)
  btc_spot_trades.csv       spot aggregate trade history
  btc_futures_trades.csv    USDT-M futures aggregate trade history

Each millisecond timestamp is followed by its UTC datetime
(`2026-08-19 10:40:55.065+00:00`).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

SYMBOL = "BTCUSDT"
INTERVAL = "1s"
HOURS = 24
SECONDS_PER_HOUR = 3600
EXPECTED_ROWS = HOURS * SECONDS_PER_HOUR
LIMIT = 1000
SPOT_KLINES_URL = "https://api.binance.com/api/v3/klines"
SPOT_AGGTRADES_URL = "https://api.binance.com/api/v3/aggTrades"
FUTURES_AGGTRADES_URL = "https://fapi.binance.com/fapi/v1/aggTrades"
SPOT_AGGTRADES_ZIP = (
    "https://data.binance.vision/data/spot/daily/aggTrades/"
    "{symbol}/{symbol}-aggTrades-{day}.zip"
)
FUTURES_AGGTRADES_ZIP = (
    "https://data.binance.vision/data/futures/um/daily/aggTrades/"
    "{symbol}/{symbol}-aggTrades-{day}.zip"
)
SPOT_CANDLES_CSV = Path("btc_spot_candles.csv")
FUTURES_CANDLES_CSV = Path("btc_futures_candles.csv")
SPOT_TRADES_CSV = Path("btc_spot_trades.csv")
FUTURES_TRADES_CSV = Path("btc_futures_trades.csv")
TIMEOUT_S = 30
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

Trade = tuple[int, int, float, float, bool]


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


def _http_get_json(url: str, params: dict[str, str | int]) -> list:
    query = urllib.parse.urlencode(params)
    raw = _http_open(f"{url}?{query}")
    return json.loads(raw.decode("utf-8"))


def fetch_spot_1s_klines(symbol: str = SYMBOL) -> list[list]:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - EXPECTED_ROWS * 1000
    rows: list[list] = []
    cursor = start_ms

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
        if rows and batch[0][0] == rows[-1][0]:
            batch = batch[1:]
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 1000
        print(f"  Binance spot klines  {len(rows)} / {EXPECTED_ROWS}", flush=True)
        if len(batch) < LIMIT:
            break

    if len(rows) > EXPECTED_ROWS:
        rows = rows[-EXPECTED_ROWS:]
    return rows


def _day_bounds_ms(day: datetime) -> tuple[int, int]:
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _to_ms(ts: int) -> int:
    if ts >= 10**15:
        return ts // 1000
    return ts


def ms_to_utc(ms: int) -> str:
    """UTC datetime next to each millisecond timestamp: 2026-08-19 10:40:55.065+00:00."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat(
        sep=" ", timespec="milliseconds"
    )


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
    print(f"  {label}  downloading {url}", flush=True)
    try:
        raw = _http_open(url, timeout=180)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise

    trades: list[Trade] = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for name in zf.namelist():
            with zf.open(name) as handle:
                text = io.TextIOWrapper(handle, encoding="utf-8", newline="")
                reader = csv.reader(text)
                for row in reader:
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
    cursor = start_ms
    while cursor < end_ms:
        day = datetime.fromtimestamp(cursor / 1000.0, tz=timezone.utc)
        _, day_end_ms = _day_bounds_ms(day)
        chunk_end = min(day_end_ms, end_ms)
        zipped = _load_aggtrades_zip(zip_template, day, cursor, chunk_end, symbol, label)
        if zipped is None:
            trades.extend(
                _fetch_aggtrades_rest(rest_url, cursor, chunk_end, symbol, label, sleep_s)
            )
        else:
            trades.extend(zipped)
        cursor = chunk_end
    trades.sort(key=lambda row: row[1])
    return trades


def aggtrades_to_1s_ohlcv(trades: list[Trade], start_ms: int, n: int) -> list[dict[str, str | int | float]]:
    if not trades:
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

    first_price = trades[0][2]
    prev_close = first_price
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


def fetch_spot_candles() -> int:
    print(f"Fetching last {HOURS}h of {SYMBOL} {INTERVAL} klines from Binance spot...")
    spot_rows = fetch_spot_1s_klines()
    if not spot_rows:
        print("No spot klines returned.", file=sys.stderr)
        return 1
    candles = klines_to_ohlcv(spot_rows)
    write_csv(SPOT_CANDLES_CSV, candles, CANDLE_FIELDS)
    _print_candles(SPOT_CANDLES_CSV, candles)
    return 0


def load_futures_aggtrades(start_ms: int, end_ms: int) -> list[Trade]:
    print(f"Fetching last {HOURS}h of {SYMBOL} aggTrades from Binance USDT-M futures...")
    return fetch_aggtrades(
        zip_template=FUTURES_AGGTRADES_ZIP,
        rest_url=FUTURES_AGGTRADES_URL,
        label="USDT-M futures",
        start_ms=start_ms,
        end_ms=end_ms,
        sleep_s=0.51,
    )


def fetch_futures_candles(start_ms: int, end_ms: int, trades: list[Trade] | None = None) -> int:
    if trades is None:
        trades = load_futures_aggtrades(start_ms, end_ms)
    else:
        print(f"Building last {HOURS}h of {SYMBOL} 1s OHLCV from USDT-M futures aggTrades...")
    candles = aggtrades_to_1s_ohlcv(trades, start_ms, EXPECTED_ROWS)
    if not candles:
        print("No futures trades returned.", file=sys.stderr)
        return 1
    write_csv(FUTURES_CANDLES_CSV, candles, CANDLE_FIELDS)
    _print_candles(FUTURES_CANDLES_CSV, candles)
    return 0


def fetch_spot_trades(start_ms: int, end_ms: int) -> int:
    print(f"Fetching last {HOURS}h of {SYMBOL} trades from Binance spot...")
    trades = fetch_aggtrades(
        zip_template=SPOT_AGGTRADES_ZIP,
        rest_url=SPOT_AGGTRADES_URL,
        label="Binance spot trades",
        start_ms=start_ms,
        end_ms=end_ms,
        sleep_s=0.05,
    )
    if not trades:
        print("No spot trades returned.", file=sys.stderr)
        return 1
    rows = trades_to_rows(trades)
    write_csv(SPOT_TRADES_CSV, rows, TRADE_FIELDS)
    _print_trades(SPOT_TRADES_CSV, rows)
    return 0


def fetch_futures_trades(start_ms: int, end_ms: int, trades: list[Trade] | None = None) -> int:
    if trades is None:
        trades = load_futures_aggtrades(start_ms, end_ms)
    if not trades:
        print("No futures trades returned.", file=sys.stderr)
        return 1
    rows = trades_to_rows(trades)
    write_csv(FUTURES_TRADES_CSV, rows, TRADE_FIELDS)
    _print_trades(FUTURES_TRADES_CSV, rows)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Binance BTCUSDT 24h market data")
    parser.add_argument(
        "--only",
        choices=("spot-candles", "futures-candles", "spot-trades", "futures-trades"),
        action="append",
        help="fetch only these datasets (repeatable); default is all",
    )
    args = parser.parse_args(argv)
    wanted = set(args.only) if args.only else {
        "spot-candles",
        "futures-candles",
        "spot-trades",
        "futures-trades",
    }

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - EXPECTED_ROWS * 1000
    status = 0
    if "spot-candles" in wanted:
        status |= fetch_spot_candles()

    futures_trades: list[Trade] | None = None
    if "futures-candles" in wanted and "futures-trades" in wanted:
        futures_trades = load_futures_aggtrades(start_ms, end_ms)
    if "futures-candles" in wanted:
        status |= fetch_futures_candles(start_ms, end_ms, trades=futures_trades)
    if "spot-trades" in wanted:
        status |= fetch_spot_trades(start_ms, end_ms)
    if "futures-trades" in wanted:
        status |= fetch_futures_trades(start_ms, end_ms, trades=futures_trades)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
