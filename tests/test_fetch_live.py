from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fetch_live import (
    RAW_STREAMS,
    LiveCapture,
    VenueBuf,
    _NeedMore,
    _ws_next_text,
    assemble_payload,
    assert_env_payload_schema,
    save_payload,
)
from harness.payload import payload_from_jsonable, payload_to_jsonable


def _snap(*, futures: bool, recv: int, uid: int) -> dict:
    bids = np.array([[100.0, 1.0], [99.0, 2.0]], dtype=np.float64)
    asks = np.array([[101.0, 1.0], [102.0, 2.0]], dtype=np.float64)
    return {
        "recv_ts_ms": np.int64(recv),
        "update_id": uid,
        "bids": bids,
        "asks": asks,
        "event_ts_ms": recv - 1 if futures else None,
        "transaction_ts_ms": recv - 2 if futures else None,
    }


def _candles(t0: int, n: int = 4) -> dict:
    return {
        "open_time_ms": np.arange(t0, t0 + n * 1000, 1000, dtype=np.int64),
        "ohlcv": np.ones((n, 5), dtype=np.float64),
        "complete_history": n >= 3600,
    }


def _capture(now: int) -> LiveCapture:
    t0 = now - 60_000
    spot = VenueBuf()
    fut = VenueBuf()
    spot.trades.append(
        {
            "ts_ms": now - 10,
            "event_ts_ms": now - 9,
            "recv_ts_ms": now - 8,
            "price": 100.0,
            "qty": 0.1,
            "buyer_is_maker": False,
        }
    )
    spot.book.append(
        {
            "recv_ts_ms": now - 5,
            "bid_price": 99.5,
            "bid_qty": 1.0,
            "ask_price": 100.5,
            "ask_qty": 1.2,
        }
    )
    fut.book.append(
        {
            "recv_ts_ms": now - 4,
            "bid_price": 99.6,
            "bid_qty": 2.0,
            "ask_price": 100.6,
            "ask_qty": 2.2,
            "event_ts_ms": now - 6,
            "transaction_ts_ms": now - 7,
        }
    )
    fut.depth_updates.append(
        {
            "recv_ts_ms": np.int64(now - 3),
            "event_ts_ms": np.int64(now - 4),
            "transaction_ts_ms": np.int64(now - 5),
            "first_id": 51,
            "final_id": 52,
            "prev_final_id": 50,
            "bids": np.array([[99.5, 0.4]], dtype=np.float64),
            "asks": np.zeros((0, 2), dtype=np.float64),
        }
    )
    spot.last_recv = {"trade": now - 8, "bookTicker": now - 5, "depth": now - 1, "kline_1s": now - 20}
    fut.last_recv = {"trade": now - 8, "bookTicker": now - 4, "depth": now - 3}
    return LiveCapture(
        t0_ms=t0,
        depth_start={
            "spot": _snap(futures=False, recv=t0, uid=10),
            "futures": _snap(futures=True, recv=t0, uid=50),
        },
        depth_latest={
            "spot": _snap(futures=False, recv=now, uid=20),
            "futures": _snap(futures=True, recv=now, uid=60),
        },
        venues={"spot": spot, "futures": fut},
    )


def test_assembled_payload_matches_env_schema():
    now = 1_700_000_060_000
    payload = assemble_payload(
        _capture(now),
        spot_candles=_candles(now - 4_000),
        futures_candles=_candles(now - 4_000),
        current_time_ms=now,
    )
    assert_env_payload_schema(payload)
    assert payload["prompt"]["trigger"] == {"kind": "time", "venue": None}
    spot = payload["venues"]["spot"]
    fut = payload["venues"]["futures"]
    assert "event_ts_ms" not in spot["book_ticker"]
    assert "transaction_ts_ms" not in spot["book_ticker"]
    assert "event_ts_ms" in fut["book_ticker"]
    assert spot["depth_start"]["event_ts_ms"] is None
    assert fut["depth_start"]["event_ts_ms"] is not None
    assert "transaction_ts_ms" not in spot["depth_updates"]
    assert fut["depth_updates"][0]["transaction_ts_ms"] is not None
    assert spot["trades"]["ts_ms"].dtype == np.int64
    assert fut["book_ticker"]["bid_price"].dtype == np.float64


def test_validate_payload_file_reports_crps(tmp_path: Path):
    import harness.validate as v

    payload = assemble_payload(
        _capture(1_700_000_060_000),
        spot_candles=_candles(1_700_000_056_000),
        futures_candles=_candles(1_700_000_056_000),
        current_time_ms=1_700_000_060_000,
    )
    path = tmp_path / "env.json"
    save_payload(payload, path)
    loaded = payload_from_jsonable(json.loads(path.read_text(encoding="utf-8")))

    report = v.validate(rounds=1, payload=loaded, realized_price=100.0)
    assert report["ok"] is True
    assert report["n"] == 1
    assert report["crps"] >= 0.0


def test_payload_json_roundtrip(tmp_path: Path):
    now = 1_700_000_060_000
    payload = assemble_payload(
        _capture(now),
        spot_candles=_candles(now - 4_000),
        futures_candles=_candles(now - 4_000),
        current_time_ms=now,
    )
    path = tmp_path / "env_payload.json"
    save_payload(payload, path)
    restored = payload_from_jsonable(json.loads(path.read_text(encoding="utf-8")))
    assert_env_payload_schema(restored)
    as_json = payload_to_jsonable(restored)
    assert as_json["prompt"]["current_time_ms"] == now
    assert "event_ts_ms" not in as_json["venues"]["spot"]["book_ticker"]
    assert restored["venues"]["spot"]["trades"]["buyer_is_maker"].dtype == bool


def _text_frame(text: str) -> bytes:
    raw = text.encode("utf-8")
    assert len(raw) < 126
    return bytes([0x81, len(raw)]) + raw


def test_split_websocket_frame_is_not_dropped():
    raw = _text_frame('{"e":"bookTicker","b":"1"}')
    sock = type("Sock", (), {"_ws_buf": raw[:6], "_ws_frag": None})()
    try:
        _ws_next_text(sock)
    except _NeedMore:
        pass
    else:
        raise AssertionError("partial frame should wait")
    assert sock._ws_buf == raw[:6]
    sock._ws_buf += raw[6:]
    assert _ws_next_text(sock) == '{"e":"bookTicker","b":"1"}'
    try:
        _ws_next_text(sock)
    except _NeedMore:
        pass
    else:
        raise AssertionError("empty buffer should wait")


def test_futures_aggtrade_uses_the_market_path():
    paths = {(venue, stream): path for venue, _host, _port, path, stream in RAW_STREAMS}
    assert paths[("futures", "btcusdt@aggTrade")] == "/market/ws/btcusdt@aggTrade"
    assert paths[("futures", "btcusdt@bookTicker")] == "/public/ws/btcusdt@bookTicker"
    assert paths[("futures", "btcusdt@depth@100ms")] == "/public/ws/btcusdt@depth@100ms"
    assert paths[("futures", "btcusdt@forceOrder")] == "/market/ws/btcusdt@forceOrder"
    assert paths[("spot", "btcusdt@aggTrade")] == "/ws/btcusdt@aggTrade"


def test_force_order_side_is_the_order_not_the_trapped_position():
    from fetch_live import _parse_ws

    buf = VenueBuf()
    _parse_ws(
        "btcusdt@forceOrder",
        {
            "e": "forceOrder",
            "E": 1000,
            "o": {
                "s": "BTCUSDT",
                "S": "SELL",
                "p": "100",
                "ap": "99.5",
                "q": "1",
                "z": "0.4",
                "T": 900,
            },
        },
        1100,
        futures=True,
        buf=buf,
    )
    row = buf.liquidations[0]
    assert row["is_buy"] is False
    assert row["price"] == 99.5
    assert row["qty"] == 0.4
    assert row["ts_ms"] == 900
    buy = VenueBuf()
    _parse_ws(
        "btcusdt@forceOrder",
        {"e": "forceOrder", "E": 2000, "o": {"S": "BUY", "ap": "101", "z": "0.2", "T": 1900}},
        2100,
        futures=True,
        buf=buy,
    )
    assert buy.liquidations[0]["is_buy"] is True


def test_two_websocket_frames_in_one_buffer():
    sock = type("Sock", (), {"_ws_buf": _text_frame("one") + _text_frame("two"), "_ws_frag": None})()
    assert _ws_next_text(sock) == "one"
    assert _ws_next_text(sock) == "two"


def test_use_asset_switches_symbol_bucket_and_streams():
    import fetch_live
    from fetch_live import use_asset

    now = 1_700_000_060_000
    try:
        for name, symbol, bucket, stream in (
            ("ETH", "ETHUSDT", 0.5, "ethusdt@aggTrade"),
            ("sol", "SOLUSDT", 0.03, "solusdt@aggTrade"),
        ):
            spec = use_asset(name)
            assert spec.symbol == symbol
            assert fetch_live.SYMBOL == symbol
            assert fetch_live.ASSET == name.upper()
            payload = assemble_payload(
                _capture(now),
                spot_candles=_candles(now - 4_000),
                futures_candles=_candles(now - 4_000),
                current_time_ms=now,
            )
            assert_env_payload_schema(payload)
            assert payload["prompt"]["asset"] == name.upper()
            assert payload["venues"]["spot"]["symbol"] == symbol
            assert payload["venues"]["futures"]["symbol"] == symbol
            assert float(payload["venues"]["spot"]["depth_bands"]["bucket_usd"]) == bucket
            assert float(payload["venues"]["futures"]["depth_bands"]["bucket_usd"]) == bucket
            assert stream in {item[4] for item in RAW_STREAMS}
            assert fetch_live.RAW_STREAMS is RAW_STREAMS
    finally:
        use_asset("BTC")
    assert fetch_live.SYMBOL == "BTCUSDT"
    assert fetch_live.ASSET == "BTC"
    assert "btcusdt@aggTrade" in {item[4] for item in RAW_STREAMS}
