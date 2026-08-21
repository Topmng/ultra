from __future__ import annotations

from datetime import datetime, timezone

from binance_fetch import (
    CANDLE_FIELDS,
    _parse_book_depth_row,
    _parse_book_ticker_row,
    _parse_zip_kline,
    _to_ms,
    book_ticker_from_candle,
    csv_first_ms,
    csv_last_ms,
    dataset_windows,
    ingest_rows,
    migrate_legacy_data_files,
    parse_book_depth_timestamp,
    parse_day_arg,
    percent_levels_to_book,
    write_csv,
)
from synth_ultra.payload import _percent_levels_to_book
import numpy as np


def test_to_ms_converts_microseconds():
    assert _to_ms(1_787_097_600_000_000) == 1_787_097_600_000
    assert _to_ms(1_787_097_600_000) == 1_787_097_600_000


def test_parse_zip_kline_microseconds():
    row = _parse_zip_kline(
        [
            "1787097600000000",
            "64725.42",
            "64725.42",
            "64725.42",
            "64725.42",
            "0.027",
            "1787097600999999",
            "1752.11",
            "2",
            "0.027",
            "1752.11",
            "0",
        ]
    )
    assert row is not None
    assert row[0] == 1_787_097_600_000
    assert row[6] == 1_787_097_600_999
    assert _parse_zip_kline(["open_time", "open"]) is None


def test_parse_book_depth_row():
    parsed = _parse_book_depth_row(
        ["2026-08-19 00:00:06", "-0.20", "535.593", "34613214.08"]
    )
    assert parsed is not None
    ts, pct, depth, notional = parsed
    assert ts == parse_book_depth_timestamp("2026-08-19 00:00:06")
    assert pct == -0.20
    assert depth == 535.593
    assert _parse_book_depth_row(["timestamp", "percentage", "depth", "notional"]) is None


def test_parse_futures_book_ticker_row():
    parsed = _parse_book_ticker_row(
        ["3831603310013", "2474.30", "6.009", "2474.31", "44.757", "1705276800000", "1705276800005"],
        futures=True,
    )
    assert parsed is not None
    recv, bid_p, bid_q, ask_p, ask_q, event, tx, uid = parsed
    assert recv == 1_705_276_800_005
    assert bid_p == 2474.30
    assert ask_p == 2474.31
    assert event == 1_705_276_800_005
    assert tx == 1_705_276_800_000
    assert uid == 3_831_603_310_013


def test_parse_day_arg_is_utc_midnight():
    day = parse_day_arg("2026-08-19")
    assert day == datetime(2026, 8, 19, tzinfo=timezone.utc)


def test_percent_levels_to_book_best_first():
    pct = np.array([-5.0, -1.0, -0.2, 0.2, 1.0, 5.0])
    depth = np.array([100.0, 40.0, 10.0, 12.0, 50.0, 120.0])
    notional = depth * np.array([95.0, 99.0, 99.8, 100.2, 101.0, 105.0])
    bids, asks = _percent_levels_to_book(pct, depth, notional)
    assert bids.shape[0] == 3
    assert asks.shape[0] == 3
    assert bids[0, 0] > bids[-1, 0]
    assert asks[0, 0] < asks[-1, 0]
    np.testing.assert_allclose(bids[0, 1], 10.0)
    np.testing.assert_allclose(asks[0, 1], 12.0)


def test_csv_first_last_and_resume_windows(tmp_path):
    path = tmp_path / "candles.csv"
    write_csv(
        path,
        [
            {
                "open_time_ms": 1_000,
                "open_time_utc": "t0",
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 0,
            },
            {
                "open_time_ms": 3_000,
                "open_time_utc": "t1",
                "open": 2,
                "high": 2,
                "low": 2,
                "close": 2,
                "volume": 0,
            },
        ],
        CANDLE_FIELDS,
    )
    assert csv_first_ms(path, "open_time_ms") == 1_000
    assert csv_last_ms(path, "open_time_ms") == 3_000
    windows = dataset_windows(path, "open_time_ms", 0, 5_000, 1_000)
    assert (3_000 + 1_000, 5_000) in windows
    assert (0, 1_000) in windows
    assert dataset_windows(path, "open_time_ms", 1_000, 3_001, 1_000) == []


def test_ingest_rows_appends_after_last(tmp_path):
    path = tmp_path / "candles.csv"
    write_csv(
        path,
        [
            {
                "open_time_ms": 1_000,
                "open_time_utc": "t0",
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 0,
            }
        ],
        CANDLE_FIELDS,
    )
    kept = ingest_rows(
        path,
        [
            {
                "open_time_ms": 500,
                "open_time_utc": "before",
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 0,
            },
            {
                "open_time_ms": 2_000,
                "open_time_utc": "after",
                "open": 3,
                "high": 3,
                "low": 3,
                "close": 3,
                "volume": 0,
            },
            {
                "open_time_ms": 1_000,
                "open_time_utc": "dup",
                "open": 9,
                "high": 9,
                "low": 9,
                "close": 9,
                "volume": 0,
            },
        ],
        CANDLE_FIELDS,
        "open_time_ms",
    )
    assert kept == 2
    assert csv_first_ms(path, "open_time_ms") == 500
    assert csv_last_ms(path, "open_time_ms") == 2_000


def test_ingest_rows_fills_interior_gaps(tmp_path):
    path = tmp_path / "candles.csv"
    write_csv(
        path,
        [
            {
                "open_time_ms": 1_000,
                "open_time_utc": "t0",
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 0,
            },
            {
                "open_time_ms": 3_000,
                "open_time_utc": "t1",
                "open": 3,
                "high": 3,
                "low": 3,
                "close": 3,
                "volume": 0,
            },
        ],
        CANDLE_FIELDS,
    )
    kept = ingest_rows(
        path,
        [
            {
                "open_time_ms": 2_000,
                "open_time_utc": "mid",
                "open": 2,
                "high": 2,
                "low": 2,
                "close": 2,
                "volume": 0,
            }
        ],
        CANDLE_FIELDS,
        "open_time_ms",
        fill_gaps=True,
    )
    assert kept == 1
    times = []
    with path.open(encoding="utf-8") as handle:
        for row in __import__("csv").DictReader(handle):
            times.append(int(row["open_time_ms"]))
    assert times == [1_000, 2_000, 3_000]


def test_migrate_legacy_data_files(tmp_path):
    src = tmp_path / "root"
    dest = tmp_path / "database"
    src.mkdir()
    (src / "btc_spot_candles.csv").write_text("open_time_ms\n1\n", encoding="utf-8")
    moved = migrate_legacy_data_files(src, dest)
    assert moved == ["btc_spot_candles.csv"]
    assert (dest / "btc_spot_candles.csv").is_file()
    assert not (src / "btc_spot_candles.csv").exists()


def test_book_ticker_from_candle_uses_close_and_spread():
    template = {
        "bid_price": 100.0,
        "ask_price": 100.2,
        "bid_qty": 1.5,
        "ask_qty": 2.5,
    }
    row = book_ticker_from_candle(
        {"open_time_ms": "50", "close": "200"},
        template,
        futures=False,
    )
    assert row["recv_ts_ms"] == 50
    assert row["bid_price"] == 199.9
    assert row["ask_price"] == 200.1
    assert "event_ts_ms" not in row


def test_percent_levels_to_book_python():
    bids, asks = percent_levels_to_book(
        [(-5.0, 100.0, 9500.0), (-0.2, 10.0, 998.0), (0.2, 12.0, 1202.4), (5.0, 120.0, 12600.0)]
    )
    assert len(bids) == 2
    assert len(asks) == 2
    assert bids[0][1] == 10.0
    assert asks[0][1] == 12.0
    assert bids[0][0] > bids[-1][0]
    assert asks[0][0] < asks[-1][0]

