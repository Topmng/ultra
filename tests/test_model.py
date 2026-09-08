from __future__ import annotations

import numpy as np
import pytest

from synth_ultra.constants import CANDLE_WINDOW_S, NUM_PERCENTILES, QUANTILE_GRID, TRADE_WINDOW_S
from synth_ultra.model import predict_percentiles
from synth_ultra.payload import (
    FUTURES_CANDLES_CSV,
    FUTURES_TRADES_CSV,
    SAMPLE_PAYLOAD_JSON,
    SPOT_CANDLES_CSV,
    SPOT_TRADES_CSV,
    assert_payload_schema,
    default_current_time_ms,
    load_payload,
    make_sample_payload,
    payload_to_jsonable,
)
from synth_ultra.scoring import pinball_crps, realized_spot_microprice, spot_microprice
from synth_ultra.validate import backtest_anchors, check_output, run_once, validate


def test_quantile_grid_matches_spec():
    expected = (2 * np.arange(1, 101) - 1) / 200.0
    np.testing.assert_allclose(QUANTILE_GRID, expected)
    assert QUANTILE_GRID[0] == pytest.approx(0.005)
    assert QUANTILE_GRID[-1] == pytest.approx(0.995)


def test_output_contract():
    payload = make_sample_payload(1)
    out, _ = run_once(payload)
    check_output(out)
    assert out.shape == (NUM_PERCENTILES,)
    assert out.dtype == np.float64


def test_deterministic():
    payload = make_sample_payload(2)
    a = predict_percentiles(payload)
    b = predict_percentiles(payload)
    np.testing.assert_array_equal(a, b)


def test_incomplete_futures_candles():
    if not SPOT_CANDLES_CSV.is_file():
        payload = make_sample_payload(3)
        assert payload["venues"]["spot"]["candles_1s"]["complete_history"] is True
        assert payload["venues"]["futures"]["candles_1s"]["complete_history"] is False
        n_spot = payload["venues"]["spot"]["candles_1s"]["ohlcv"].shape[0]
        n_fut = payload["venues"]["futures"]["candles_1s"]["ohlcv"].shape[0]
        assert n_fut < n_spot
        return
    first = int(np.loadtxt(SPOT_CANDLES_CSV, delimiter=",", skiprows=1, max_rows=1, usecols=0))
    payload = make_sample_payload(current_time_ms=first + 120_000, compact=False)
    assert payload["venues"]["spot"]["candles_1s"]["complete_history"] is False
    assert payload["venues"]["spot"]["candles_1s"]["ohlcv"].shape[0] < CANDLE_WINDOW_S
    out = predict_percentiles(payload)
    check_output(out)


def test_csv_candles_and_trades_windows():
    if not all(
        p.is_file()
        for p in (SPOT_CANDLES_CSV, FUTURES_CANDLES_CSV, SPOT_TRADES_CSV, FUTURES_TRADES_CSV)
    ):
        pytest.skip("venue CSVs are not present")
    selected = default_current_time_ms()
    payload = make_sample_payload(current_time_ms=selected, compact=False)
    assert payload["prompt"]["current_time_ms"] == selected
    for name in ("spot", "futures"):
        candles = payload["venues"][name]["candles_1s"]
        trades = payload["venues"][name]["trades"]
        assert candles["complete_history"] is True
        assert candles["ohlcv"].shape[0] >= CANDLE_WINDOW_S
        assert candles["open_time_ms"][-1] <= selected
        assert candles["open_time_ms"][0] > selected - CANDLE_WINDOW_S * 1000
        assert trades["ts_ms"].shape[0] > 0
        assert trades["ts_ms"][-1] <= selected
        assert trades["ts_ms"][0] > selected - TRADE_WINDOW_S * 1000
    out = predict_percentiles(payload)
    check_output(out)


def test_spot_book_ticker_has_no_exchange_times():
    payload = make_sample_payload(0)
    spot_bt = payload["venues"]["spot"]["book_ticker"]
    fut_bt = payload["venues"]["futures"]["book_ticker"]
    assert "event_ts_ms" not in spot_bt
    assert "transaction_ts_ms" not in spot_bt
    assert "event_ts_ms" in fut_bt
    assert "transaction_ts_ms" in fut_bt
    assert payload["venues"]["spot"]["depth_start"]["event_ts_ms"] is None
    assert payload["venues"]["futures"]["depth_start"]["event_ts_ms"] is not None


def test_empty_optional_streams_still_predict():
    payload = make_sample_payload(4)
    payload["venues"]["spot"]["trades"] = {
        "ts_ms": np.array([], dtype=np.int64),
        "event_ts_ms": np.array([], dtype=np.int64),
        "recv_ts_ms": np.array([], dtype=np.int64),
        "price": np.array([], dtype=np.float64),
        "qty": np.array([], dtype=np.float64),
        "buyer_is_maker": np.array([], dtype=bool),
    }
    payload["venues"]["futures"]["depth_latest"] = {
        "recv_ts_ms": np.int64(0),
        "update_id": 0,
        "bids": np.zeros((0, 2), dtype=np.float64),
        "asks": np.zeros((0, 2), dtype=np.float64),
        "event_ts_ms": None,
        "transaction_ts_ms": None,
    }
    out = predict_percentiles(payload)
    check_output(out)


def test_root_import_contract():
    from model import predict_percentiles as imported

    payload = make_sample_payload(5)
    check_output(imported(payload))


def test_crps_perfect_forecast_is_low():
    y = 100_000.0
    x = np.full(100, y, dtype=np.float64)
    x[-1] = y + 1e-8
    assert pinball_crps(x, y) < 1e-6


def test_microprice_formula():
    assert spot_microprice(100.0, 1.0, 102.0, 3.0) == pytest.approx(100.5)


def test_backtest_anchors_match_example_window():
    start = 1_787_208_024_000
    times = backtest_anchors(start, interval_s=1, length=60)
    assert len(times) == 60
    assert times[0] == start
    assert times[1] == start + 1000
    assert times[-1] == start + 59_000
    assert times[-1] + 1000 == start + 60_000


def test_validate_report_keys():
    t = default_current_time_ms() - 10_000
    report = validate(warmup=2, rounds=8, seed=0, strict=False, current_time_ms=t)
    assert report["ok"] is True
    assert "median_ms" in report
    assert report["budget_ms"] == 5.0
    assert report["crps"] is not None
    assert report["realized_price"] > 0
    assert report["n_scored"] >= 1


def test_compact_json_payload_is_predictable():
    payload = make_sample_payload(0, compact=True, synthetic=True, current_time_ms=1_700_000_000_000)
    as_json = payload_to_jsonable(payload)
    assert as_json["venues"]["spot"]["book_ticker"]["bid_price"]
    assert "event_ts_ms" not in as_json["venues"]["spot"]["book_ticker"]
    out = predict_percentiles(as_json)
    check_output(out)
    assert len(as_json["venues"]["spot"]["candles_1s"]["ohlcv"]) == 8


def test_synthetic_sample_matches_input_md():
    payload = make_sample_payload(0, compact=True, synthetic=True, current_time_ms=1_700_000_000_000)
    assert_payload_schema(payload)
    spot = payload["venues"]["spot"]
    fut = payload["venues"]["futures"]
    assert spot["candles_1s"]["complete_history"] is True
    assert fut["candles_1s"]["complete_history"] is False
    assert spot["depth_latest"]["bids"].shape == (20, 2)
    assert fut["depth_latest"]["asks"].shape == (20, 2)
    assert "transaction_ts_ms" not in spot["depth_updates"][0]
    assert "transaction_ts_ms" in fut["depth_updates"][0]
    assert int(spot["last_event_times"]["trade"]) == int(spot["trades"]["recv_ts_ms"][-1])
    assert int(fut["last_event_times"]["bookTicker"]) == int(fut["book_ticker"]["recv_ts_ms"][-1])


def test_sample_payload_json_roundtrip():
    if not SAMPLE_PAYLOAD_JSON.is_file():
        pytest.skip("examples/sample_payload.json is not present")
    payload = load_payload(SAMPLE_PAYLOAD_JSON)
    assert_payload_schema(payload)
    check_output(predict_percentiles(payload))


def test_realized_microprice_from_book_ticker():
    from synth_ultra.payload import SPOT_BOOK_TICKER_CSV, _load_book_ticker_csv

    if not SPOT_BOOK_TICKER_CSV.is_file():
        pytest.skip("spot book-ticker CSV is not present")
    table = _load_book_ticker_csv(str(SPOT_BOOK_TICKER_CSV))
    assert table is not None and table["recv_ts_ms"].size > 10
    i = table["recv_ts_ms"].size // 2
    recv = int(table["recv_ts_ms"][i])
    expected = spot_microprice(
        float(table["bid_price"][i]),
        float(table["bid_qty"][i]),
        float(table["ask_price"][i]),
        float(table["ask_qty"][i]),
    )
    got = realized_spot_microprice(recv - 10_000)
    assert got == pytest.approx(expected)


def test_stale_spot_feed_is_dropped():
    from synth_ultra.payload import SPOT_BOOK_TICKER_CSV, _load_book_ticker_csv

    if not SPOT_BOOK_TICKER_CSV.is_file():
        pytest.skip("spot book-ticker CSV is not present")
    table = _load_book_ticker_csv(str(SPOT_BOOK_TICKER_CSV))
    assert table is not None and table["recv_ts_ms"].size
    last = int(table["recv_ts_ms"][-1])
    assert realized_spot_microprice(last + 60_000) is None
