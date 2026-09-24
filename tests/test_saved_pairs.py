from __future__ import annotations

import json

import pytest

from fetch_live import save_payload
from fetch_payload import (
    TradeGroupDetector,
    interval_to_ms,
    last_spot_book_in_window,
    moved_more_than_one_tick,
    select_depth_pair,
)
from synth_ultra.payload import make_sample_payload
from synth_ultra.saved_pairs import book_path, payload_path
from synth_ultra.validate import validate_saved_pairs


def test_move_is_more_than_one_tick():
    assert moved_more_than_one_tick(100.0, 100.01, 0.01) is False
    assert moved_more_than_one_tick(100.0, 100.02, 0.01) is True


def test_trade_group_emits_the_group_receive_time():
    detector = TradeGroupDetector(0.01)
    assert detector.push(900, 100.00, 4_000) is None
    assert detector.push(1_000, 100.01, 5_000) is None
    assert detector.push(1_000, 100.01, 5_010) is None
    assert detector.push(1_050, 100.03, 5_080) is None
    assert detector.push(1_100, 100.03, 5_200) == 5_080


def test_depth_pair_is_at_or_before_the_anchor():
    snaps = [
        {"recv_ms": 0},
        {"recv_ms": 59_000},
        {"recv_ms": 60_000},
        {"recv_ms": 61_000},
        {"recv_ms": 121_000},
    ]
    start, latest = select_depth_pair(snaps, 120_000)
    assert start["recv_ms"] == 60_000
    assert latest["recv_ms"] == 61_000


def test_depth_start_rejects_a_snapshot_far_from_sixty_seconds():
    snaps = [{"recv_ms": 75_000}, {"recv_ms": 120_000}]
    with pytest.raises(RuntimeError):
        select_depth_pair(snaps, 120_000)


def test_interval_milliseconds():
    assert interval_to_ms(57.171) == 57171
    assert interval_to_ms(10) == 10000


def test_last_spot_book_is_last_inside_window():
    rows = [
        {"recv_ts_ms": 1000, "bid_price": 1},
        {"recv_ts_ms": 1500, "bid_price": 2},
        {"recv_ts_ms": 2000, "bid_price": 3},
        {"recv_ts_ms": 2001, "bid_price": 4},
    ]
    got = last_spot_book_in_window(rows, 1000, 2000)
    assert got is not None
    assert got["bid_price"] == 3


def test_validate_saved_pairs_uses_book_file(tmp_path):
    anchor = 1_700_000_000_000
    folder = tmp_path / str(anchor)
    payload = make_sample_payload(
        seed=1, current_time_ms=anchor, synthetic=True, compact=True
    )
    save_payload(payload, payload_path(folder))
    book_path(folder).write_text(
        json.dumps(
            {
                "current_time_ms": anchor,
                "target_ms": anchor + 10_000,
                "recv_ts_ms": anchor + 10_000,
                "bid_price": 100.0,
                "bid_qty": 1.0,
                "ask_price": 100.4,
                "ask_qty": 1.0,
            }
        ),
        encoding="utf-8",
    )
    report = validate_saved_pairs(tmp_path, warmup=0, rounds=1)
    assert report["n"] == 1
    assert report["n_scored"] == 1
    assert report["crps"] >= 0.0
    assert report["reports"][0]["book_recv_ts_ms"] == anchor + 10_000


def test_nul_book_is_skipped(tmp_path):
    anchor = 1_700_000_000_000
    good = tmp_path / str(anchor)
    payload = make_sample_payload(
        seed=1, current_time_ms=anchor, synthetic=True, compact=True
    )
    save_payload(payload, payload_path(good))
    book_path(good).write_text(
        json.dumps(
            {
                "current_time_ms": anchor,
                "target_ms": anchor + 10_000,
                "recv_ts_ms": anchor + 10_000,
                "bid_price": 100.0,
                "bid_qty": 1.0,
                "ask_price": 100.4,
                "ask_qty": 1.0,
            }
        ),
        encoding="utf-8",
    )
    bad_anchor = anchor + 20_000
    bad = tmp_path / str(bad_anchor)
    bad_payload = make_sample_payload(
        seed=2, current_time_ms=bad_anchor, synthetic=True, compact=True
    )
    save_payload(bad_payload, payload_path(bad))
    book_path(bad).write_bytes(b"\x00" * 225)
    report = validate_saved_pairs(tmp_path, warmup=0, rounds=1)
    assert report["n"] == 1
    assert report["n_unreadable"] == 1
    assert report["n_scored"] == 1


def test_stale_saved_book_is_dropped(tmp_path):
    anchor = 1_700_000_000_000
    folder = tmp_path / str(anchor)
    payload = make_sample_payload(
        seed=1, current_time_ms=anchor, synthetic=True, compact=True
    )
    save_payload(payload, payload_path(folder))
    book_path(folder).write_text(
        json.dumps(
            {
                "recv_ts_ms": anchor,
                "bid_price": 100.0,
                "bid_qty": 1.0,
                "ask_price": 100.4,
                "ask_qty": 1.0,
            }
        ),
        encoding="utf-8",
    )
    report = validate_saved_pairs(tmp_path, warmup=0, rounds=1)
    assert report["n"] == 1
    assert report["n_scored"] == 0
    assert report["crps"] is None
