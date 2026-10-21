from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harness.constants import CANDLE_WINDOW_S, NUM_PERCENTILES, QUANTILE_GRID, SCHEMA_VERSION
from harness.payload import (
    SAMPLE_PAYLOAD_JSON,
    assert_payload_schema,
    depth_bands_from_book,
    empty_depth_bands,
    empty_liquidations,
    load_payload,
    make_sample_payload,
    payload_from_jsonable,
    payload_to_jsonable,
)
from harness.scoring import pinball_crps, prompt_field_scores, spot_microprice, spot_microprice_at
from harness.validate import backtest_anchors, check_output, run_once, validate
from synth_ultra.model import predict_percentiles


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


def test_season_table_matches_slot_index():
    from synth_ultra.model import _slot_index
    from synth_ultra.season_table import SEAS_ABS, SEAS_AC, SEAS_JUMP, SEAS_REL

    for table in (SEAS_ABS, SEAS_AC, SEAS_JUMP, SEAS_REL):
        assert table.shape == (1008,)
        assert np.all(np.isfinite(table))
    assert np.all(SEAS_ABS > 0.0)
    assert np.all(SEAS_REL > 0.0)
    # 1970-01-05 00:00:00 UTC was a Monday, so the first slot is 0.
    assert _slot_index(4 * 86_400_000) == 0
    assert _slot_index((4 + 5) * 86_400_000) == 5 * 144


def test_incomplete_futures_candles():
    payload = make_sample_payload(3)
    assert payload["venues"]["spot"]["candles_1s"]["complete_history"] is True
    assert payload["venues"]["futures"]["candles_1s"]["complete_history"] is False
    n_spot = payload["venues"]["spot"]["candles_1s"]["ohlcv"].shape[0]
    n_fut = payload["venues"]["futures"]["candles_1s"]["ohlcv"].shape[0]
    assert n_fut < n_spot


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


def test_crps_matches_spec_formula():
    y = 10.0
    x = np.full(100, 11.0, dtype=np.float64)
    # u = -1, rho = 1 - tau, sum(tau) = 50, CRPS = (2/100) * 50 = 1
    assert pinball_crps(x, y) == pytest.approx(1.0)


def test_prompt_field_scores_match_faq():
    raw = np.array([1.0, 2.0, 3.0, 4.0, 0.0])
    answered = np.array([True, True, True, True, False])
    adjusted = prompt_field_scores(raw, answered)
    assert adjusted is not None
    penalty = float(np.quantile(raw[:4], 0.95))
    assert adjusted[-1] == pytest.approx(penalty - 1.0)
    assert adjusted[0] == pytest.approx(0.0)
    assert prompt_field_scores(raw, np.zeros(5, dtype=bool)) is None


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
    report = validate(warmup=1, rounds=2, seed=0, strict=False, time_length=1)
    assert report["ok"] is True
    assert "median_ms" in report
    assert report["budget_ms"] == 5.0
    assert report["n"] == 1
    if report["crps"] is None:
        assert report["n_dropped"] == 1
        assert report["n_scored"] == 0
    else:
        assert report["realized_price"] > 0
        assert report["n_scored"] == 1


def test_compact_json_payload_is_predictable():
    payload = make_sample_payload(0, compact=True, current_time_ms=1_700_000_000_000)
    as_json = payload_to_jsonable(payload)
    assert as_json["venues"]["spot"]["book_ticker"]["bid_price"]
    assert "event_ts_ms" not in as_json["venues"]["spot"]["book_ticker"]
    out = predict_percentiles(as_json)
    check_output(out)
    assert len(as_json["venues"]["spot"]["candles_1s"]["ohlcv"]) == 8


def test_synthetic_sample_matches_input_md():
    payload = make_sample_payload(0, compact=True, current_time_ms=1_700_000_000_000)
    assert_payload_schema(payload)
    spot = payload["venues"]["spot"]
    fut = payload["venues"]["futures"]
    assert spot["candles_1s"]["complete_history"] is False
    assert fut["candles_1s"]["complete_history"] is False
    assert payload["prompt"]["trigger"] == {"kind": "time", "venue": None}
    assert spot["depth_latest"]["bids"].shape == (20, 2)
    assert fut["depth_latest"]["asks"].shape == (20, 2)
    assert "transaction_ts_ms" not in spot["depth_updates"][0]
    assert "transaction_ts_ms" in fut["depth_updates"][0]
    assert int(spot["last_event_times"]["trade"]) == int(spot["trades"]["recv_ts_ms"][-1])
    assert int(fut["last_event_times"]["bookTicker"]) == int(fut["book_ticker"]["recv_ts_ms"][-1])


def test_full_synthetic_hour_flags():
    payload = make_sample_payload(0, compact=False, current_time_ms=1_700_000_000_000)
    assert_payload_schema(payload)
    spot = payload["venues"]["spot"]["candles_1s"]
    fut = payload["venues"]["futures"]["candles_1s"]
    assert spot["complete_history"] is True
    assert spot["ohlcv"].shape[0] >= CANDLE_WINDOW_S
    assert fut["complete_history"] is False
    assert fut["ohlcv"].shape[0] < spot["ohlcv"].shape[0]
    assert payload["prompt"]["trigger"]["kind"] == "time"


def test_schema_v5_fields_follow_the_venue_split():
    payload = make_sample_payload(0, compact=True, current_time_ms=1_700_000_000_000)
    assert payload["schema_version"] == SCHEMA_VERSION == 5
    spot = payload["venues"]["spot"]
    fut = payload["venues"]["futures"]
    assert spot["liquidations"]["ts_ms"].size == 0
    assert fut["liquidations"]["ts_ms"].size > 0
    assert int(fut["depth_bands"]["recv_ts_ms"]) == 0
    assert np.asarray(fut["depth_bands"]["bids"]).size == 0
    assert int(spot["depth_bands"]["recv_ts_ms"]) > 0
    assert np.asarray(spot["depth_bands"]["bids"]).size == 100


def test_buy_liquidation_lifts_the_distribution_and_sell_lowers_it():
    now = 1_700_000_000_000
    payload = make_sample_payload(1, compact=True, current_time_ms=now)
    payload["venues"]["futures"]["liquidations"] = empty_liquidations()
    payload["venues"]["spot"]["depth_bands"] = empty_depth_bands()
    base = predict_percentiles(payload)
    payload["venues"]["futures"]["liquidations"] = {
        "ts_ms": np.array([now - 100], dtype=np.int64),
        "recv_ts_ms": np.array([now - 100], dtype=np.int64),
        "price": np.array([100_000.0]),
        "qty": np.array([5.0]),
        "is_buy": np.array([True]),
    }
    lifted = predict_percentiles(payload)
    assert float(lifted[49]) > float(base[49])
    payload["venues"]["futures"]["liquidations"]["is_buy"] = np.array([False])
    lowered = predict_percentiles(payload)
    assert float(lowered[49]) < float(base[49])


def test_depth_bands_far_from_the_live_mid_do_not_move_the_forecast():
    now = 1_700_000_000_000
    payload = make_sample_payload(1, compact=True, current_time_ms=now)
    payload["venues"]["futures"]["liquidations"] = empty_liquidations()
    payload["venues"]["spot"]["depth_bands"] = empty_depth_bands()
    book = payload["venues"]["spot"]["book_ticker"]
    mid = 0.5 * (float(book["bid_price"][-1]) + float(book["ask_price"][-1]))
    base = predict_percentiles(payload)
    far = empty_depth_bands()
    far["anchor_price"] = mid - 400.0
    far["recv_ts_ms"] = now - 1_000
    far["bids"] = np.zeros(100, dtype=np.float64)
    far["asks"] = np.zeros(100, dtype=np.float64)
    far["asks"][0] = 80.0
    payload["venues"]["spot"]["depth_bands"] = far
    assert float(predict_percentiles(payload)[49]) == float(base[49])
    near = empty_depth_bands()
    near["anchor_price"] = mid - 400.0
    near["recv_ts_ms"] = now - 1_000
    near["bids"] = np.zeros(100, dtype=np.float64)
    near["asks"] = np.zeros(100, dtype=np.float64)
    near["asks"][40] = 80.0
    payload["venues"]["spot"]["depth_bands"] = near
    assert float(predict_percentiles(payload)[49]) < float(base[49])


def test_depth_band_fold_uses_absolute_price_buckets():
    bids = np.array([[99.0, 1.0], [80.0, 3.0]])
    asks = np.array([[101.0, 2.0], [130.0, 4.0]])
    bands = depth_bands_from_book(bids, asks, 123)
    assert bands["anchor_price"] == pytest.approx(100.0)
    assert bands["bids"][0] == pytest.approx(1.0)
    assert bands["bids"][1] == pytest.approx(3.0)
    assert bands["asks"][0] == pytest.approx(2.0)
    assert bands["asks"][3] == pytest.approx(4.0)
    assert bands["recv_ts_ms"] == 123


def test_official_sample_payload_is_scored_shape():
    path = Path("samples/sample_payload.json")
    if not path.is_file():
        pytest.skip("samples/sample_payload.json is not present")
    payload = payload_from_jsonable(json.loads(path.read_text(encoding="utf-8")))
    assert payload["schema_version"] == 5
    assert payload["venues"]["spot"]["liquidations"]["ts_ms"].size == 0
    assert int(payload["venues"]["futures"]["depth_bands"]["recv_ts_ms"]) == 0
    check_output(predict_percentiles(payload))


@pytest.mark.parametrize(
    ("name", "asset", "symbol", "bucket", "slot_mod"),
    [
        ("samples/real_payload.json", "BTC", "BTCUSDT", 10.0, None),
        ("samples/real_payload_ETH.json", "ETH", "ETHUSDT", 0.5, 5_000),
        ("samples/real_payload_SOL.json", "SOL", "SOLUSDT", 0.03, 10_000),
        ("samples/real_payload_XRP.json", "XRP", "XRPUSDT", 0.0003, 15_000),
    ],
)
def test_real_asset_payloads(name, asset, symbol, bucket, slot_mod):
    path = Path(name)
    if not path.is_file():
        pytest.skip(f"{name} is not present")
    payload = load_payload(path)
    assert_payload_schema(payload)
    assert payload["prompt"]["asset"] == asset
    assert payload["venues"]["spot"]["symbol"] == symbol
    assert payload["venues"]["futures"]["symbol"] == symbol
    assert float(payload["venues"]["spot"]["depth_bands"]["bucket_usd"]) == pytest.approx(bucket)
    assert float(payload["venues"]["futures"]["depth_bands"]["bucket_usd"]) == pytest.approx(bucket)
    now = int(payload["prompt"]["current_time_ms"])
    if slot_mod is not None:
        assert now % 20_000 == slot_mod
    out = check_output(predict_percentiles(payload))
    book = payload["venues"]["spot"]["book_ticker"]
    micro = spot_microprice(
        float(book["bid_price"][-1]),
        float(book["bid_qty"][-1]),
        float(book["ask_price"][-1]),
        float(book["ask_qty"][-1]),
    )
    med = 0.5 * (float(out[49]) + float(out[50]))
    assert abs(med - micro) / micro < 0.02
    mad = float(np.mean(np.abs(out - med)))
    assert mad / micro < 0.02


@pytest.mark.parametrize(
    ("asset", "symbol", "bucket"),
    [("ETH", "ETHUSDT", 0.5), ("SOL", "SOLUSDT", 0.03), ("XRP", "XRPUSDT", 0.0003)],
)
def test_sample_payload_covers_each_asset(asset, symbol, bucket):
    payload = make_sample_payload(0, compact=True, current_time_ms=1_700_000_000_000, asset=asset)
    assert_payload_schema(payload)
    assert payload["venues"]["spot"]["symbol"] == symbol
    assert float(payload["venues"]["spot"]["depth_bands"]["bucket_usd"]) == pytest.approx(bucket)
    assert float(payload["venues"]["futures"]["depth_bands"]["bucket_usd"]) == pytest.approx(bucket)
    check_output(predict_percentiles(payload))


def test_unknown_asset_fails_schema():
    payload = make_sample_payload(0, compact=True, current_time_ms=1_700_000_000_000)
    payload["prompt"]["asset"] = "DOGE"
    with pytest.raises(ValueError, match="prompt.asset"):
        assert_payload_schema(payload)


def test_trigger_kinds():
    payload = make_sample_payload(0, compact=True, current_time_ms=1_700_000_000_000)
    payload["prompt"]["trigger"] = {"kind": "event", "venue": "spot"}
    assert_payload_schema(payload)
    payload["prompt"]["trigger"] = {"kind": "event_delayed", "venue": "futures"}
    assert_payload_schema(payload)
    payload["prompt"]["trigger"] = {"kind": "interval", "venue": None}
    with pytest.raises(ValueError):
        assert_payload_schema(payload)
    payload["prompt"]["trigger"] = {"kind": "time", "venue": "spot"}
    with pytest.raises(ValueError):
        assert_payload_schema(payload)


def test_sample_payload_json_roundtrip():
    if not SAMPLE_PAYLOAD_JSON.is_file():
        pytest.skip("examples/sample_payload.json is not present")
    payload = load_payload(SAMPLE_PAYLOAD_JSON)
    assert_payload_schema(payload)
    check_output(predict_percentiles(payload))


def test_realized_microprice_from_book_ticker():
    times = np.array([1_000, 11_000], dtype=np.int64)
    got = spot_microprice_at(times, [100.0, 101.0], [1.0, 2.0], [102.0, 103.0], [3.0, 4.0], 11_000)
    assert got == pytest.approx(spot_microprice(101.0, 2.0, 103.0, 4.0))


def test_stale_spot_feed_is_dropped():
    times = np.array([1_000], dtype=np.int64)
    assert spot_microprice_at(times, [100.0], [1.0], [102.0], [3.0], 1_000 + 60_000) is None


def _freeze_quotes(payload: dict, bid: float, qty: float = 5.0) -> float:
    """Hold both venues' top of book and the last half hour of spot candles still.

    Returns the spot microprice of that quote.
    """
    for venue, b, tick in (("spot", bid, 0.01), ("futures", bid - 40.0, 0.1)):
        book = payload["venues"][venue]["book_ticker"]
        n = book["recv_ts_ms"].size
        book["bid_price"] = np.full(n, b)
        book["ask_price"] = np.full(n, b + tick)
        book["bid_qty"] = np.full(n, qty)
        book["ask_qty"] = np.full(n, qty)
    micro = spot_microprice(bid, qty, bid + 0.01, qty)
    payload["venues"]["spot"]["candles_1s"]["ohlcv"][-1800:, :4] = micro
    return micro


def test_a_frozen_quote_puts_mass_on_the_current_microprice():
    payload = make_sample_payload(6, compact=False)
    payload["prompt"]["trigger"] = {"kind": "event", "venue": "futures"}
    micro = _freeze_quotes(payload, 97_500.0)
    out = predict_percentiles(payload)
    check_output(out)
    assert np.count_nonzero(out == micro) >= 30

    busy = make_sample_payload(6, compact=False)
    busy["prompt"]["trigger"] = {"kind": "event", "venue": "futures"}
    px = spot_microprice(
        float(busy["venues"]["spot"]["book_ticker"]["bid_price"][-1]),
        float(busy["venues"]["spot"]["book_ticker"]["bid_qty"][-1]),
        float(busy["venues"]["spot"]["book_ticker"]["ask_price"][-1]),
        float(busy["venues"]["spot"]["book_ticker"]["ask_qty"][-1]),
    )
    moving = predict_percentiles(busy)
    check_output(moving)
    assert np.count_nonzero(np.abs(moving - px) < 1e-9) < 10


def test_mix_still_places_the_point_mass_and_stays_ordered():
    from synth_ultra.model import _MIX_Z
    from synth_ultra.stillness import mix_still

    px = 100_000.0
    ladder = px + 1.5 + 4.0 * _MIX_Z
    out = mix_still(ladder, px, 0.3, 4.0, 0.75, 1.0, 0.05)
    assert out.shape == (NUM_PERCENTILES,)
    assert np.all(np.diff(out) >= 0.0)
    assert abs(np.count_nonzero(out == px) - 30) <= 1
    # The rest of the ladder keeps the mean shift, so the median moves up.
    assert 0.5 * (out[49] + out[50]) >= px
    assert mix_still(ladder, px, 0.0, 4.0, 0.75, 1.0, 0.05) is ladder


def test_ten_second_bins_keep_flat_candles_on_the_anchor_and_skew_with_mu():
    from synth_ultra.model import (
        _bin_means,
        _candle_abs_bins,
        _recency_weights,
        _skewed_candle_prices,
    )

    weights = _recency_weights(1790)
    assert weights.shape == (1790,)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] < weights[-1]

    # 60% of the weight sits on exact zeros, then one moved level.
    values = np.array([0.0, 0.0, 0.0, 0.002])
    mass = np.array([0.2, 0.2, 0.2, 0.4])
    means = _bin_means(values, mass)
    assert means.shape == (50,)
    assert np.all(means[:30] == 0.0)
    assert np.all(means[30:] == pytest.approx(0.002))

    flat = _skewed_candle_prices(100.0, means, 0.0)
    assert np.sum(np.abs(flat - 100.0) < 1e-9) == 60
    assert flat[0] < flat[-1]
    assert np.all(np.diff(flat) >= 0.0)

    # α = 1 lengthens the upper wing and lays the lower wing on the anchor.
    scale = float(means.mean())
    up = _skewed_candle_prices(100.0, means, scale * 100.0)
    down = _skewed_candle_prices(100.0, means, -scale * 100.0)
    assert up[-1] > flat[-1]
    assert up[0] > flat[0]
    assert down[0] < flat[0]
    assert down[-1] < flat[-1]
    assert np.all(np.diff(up) >= 0.0)
    assert np.all(np.diff(down) >= 0.0)

    close = np.full(1800, 50_000.0)
    close[-1] = 50_000.0 * np.exp(0.001)
    bins = _candle_abs_bins(close)
    assert bins is not None
    assert bins.shape == (50,)
    assert np.sum(bins == 0.0) >= 40
    assert _candle_abs_bins(np.full(30, 50_000.0)) is None
