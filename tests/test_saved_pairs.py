from __future__ import annotations

import json

import numpy as np
import pytest

from fetch_live import save_payload
from fetch_payload import (
    DEFAULT_DATABASE,
    TradeGroupDetector,
    clock_interval_ms,
    delete_captures_older_than,
    event_window_boundary,
    interval_to_ms,
    last_spot_book_in_window,
    main,
    moved_more_than_one_tick,
    next_time_boundary,
    select_depth_pair,
    take_qualifying_move,
)
from harness.payload import load_payload, load_payload_scoring, make_sample_payload
from harness.saved_pairs import book_path, payload_path
from harness.validate import validate_saved_pairs
from synth_ultra.model import predict_percentiles


def test_fetch_deletes_captures_older_than_one_week(tmp_path):
    now = 1_700_000_000_000
    week = 7 * 24 * 60 * 60 * 1000
    old = tmp_path / str(now - week - 1)
    edge = tmp_path / str(now - week)
    fresh = tmp_path / str(now - 1000)
    other = tmp_path / "notes.txt"
    for folder in (old, edge, fresh):
        folder.mkdir()
        (folder / "payload.json").write_text("{}", encoding="utf-8")
    other.write_text("keep", encoding="utf-8")

    assert delete_captures_older_than(tmp_path, now) == 1
    assert not old.exists()
    assert edge.exists()
    assert fresh.exists()
    assert other.read_text(encoding="utf-8") == "keep"


def test_move_is_more_than_one_tick():
    assert moved_more_than_one_tick(100.0, 100.01, 0.01) is False
    assert moved_more_than_one_tick(100.0, 100.015, 0.01) is True
    assert moved_more_than_one_tick(100.0, 100.02, 0.01) is True


def test_event_window_opens_10_25s_after_the_asset_slot():
    boundary = 1_700_000_000_000
    assert boundary % 20_000 == 0
    assert event_window_boundary(boundary + 10_249) is None
    assert event_window_boundary(boundary + 10_250) == boundary
    assert event_window_boundary(boundary + 14_249) == boundary
    assert event_window_boundary(boundary + 14_250) is None
    assert event_window_boundary(boundary + 30_250) == boundary + 20_000
    assert next_time_boundary(boundary + 1) == boundary + 20_000
    assert next_time_boundary(boundary) == boundary
    eth = boundary + 5_000
    assert next_time_boundary(eth, offset_ms=5_000) == eth
    assert next_time_boundary(eth + 1, offset_ms=5_000) == eth + 20_000
    assert event_window_boundary(eth + 10_249, offset_ms=5_000) is None
    assert event_window_boundary(eth + 10_250, offset_ms=5_000) == eth
    assert event_window_boundary(eth + 14_249, offset_ms=5_000) == eth
    assert event_window_boundary(eth + 14_250, offset_ms=5_000) is None


def test_qualifying_move_is_taken_with_a_coin_flip():
    import random

    rng = random.Random(0)
    taken: set[int] = set()
    accepted = [take_qualifying_move(rng, boundary, taken) for boundary in range(40)]
    assert any(accepted)
    assert not all(accepted)
    held = next(boundary for boundary, ok in enumerate(accepted) if ok)
    assert take_qualifying_move(rng, held, taken) is False


def test_trade_group_emits_the_group_receive_time():
    detector = TradeGroupDetector(0.01)
    assert detector.push(900, 100.00, 4_000) is None
    assert detector.push(1_000, 100.01, 5_000) is None
    assert detector.push(1_000, 100.01, 5_010) is None
    assert detector.push(1_050, 100.03, 5_080) is None
    assert detector.push(1_100, 100.03, 5_200) == 5_080


def test_walk_qualifies_two_ticks_from_the_last_move():
    """One tick from the previous group still qualifies once it is two from the last move."""
    detector = TradeGroupDetector(0.01)
    assert detector.push(1_000, 100.00, 5_000) is None
    assert detector.push(1_100, 100.01, 5_100) is None
    assert detector.push(1_200, 100.02, 5_200) is None
    assert detector.push(1_300, 100.02, 5_300) == 5_200


def test_one_tick_bounce_is_not_a_move():
    detector = TradeGroupDetector(0.01)
    assert detector.push(1_000, 100.00, 5_000) is None
    assert detector.push(1_100, 100.01, 5_100) is None
    assert detector.push(1_200, 100.00, 5_200) is None
    assert detector.push(1_300, 100.01, 5_300) is None
    assert detector.push(1_400, 100.00, 5_400) is None


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


def test_test_clock_keeps_every_tenth_live_cycle():
    """200s is every tenth live boundary, and that boundary's 4s window only."""
    assert clock_interval_ms(20) == 20_000
    assert clock_interval_ms(200) == 200_000
    with pytest.raises(ValueError):
        clock_interval_ms(57.171)

    boundary = 1_700_000_000_000
    assert boundary % 200_000 == 0
    clock_ms = clock_interval_ms(200)
    assert event_window_boundary(boundary + 10_249, clock_ms=clock_ms) is None
    assert event_window_boundary(boundary + 10_250, clock_ms=clock_ms) == boundary
    assert event_window_boundary(boundary + 14_249, clock_ms=clock_ms) == boundary
    assert event_window_boundary(boundary + 14_250, clock_ms=clock_ms) is None
    assert event_window_boundary(boundary + 30_250, clock_ms=clock_ms) is None
    assert event_window_boundary(boundary + 210_250, clock_ms=clock_ms) == boundary + 200_000
    assert next_time_boundary(boundary + 1, clock_ms=clock_ms) == boundary + 200_000
    assert next_time_boundary(boundary, clock_ms=clock_ms) == boundary


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
        seed=1, current_time_ms=anchor, compact=True
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
        seed=1, current_time_ms=anchor, compact=True
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
        seed=2, current_time_ms=bad_anchor, compact=True
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
        seed=1, current_time_ms=anchor, compact=True
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


def test_scoring_load_skips_depth_updates_and_keeps_the_forecast(tmp_path):
    payload = make_sample_payload(seed=2, compact=True)
    path = tmp_path / "payload.json"
    save_payload(payload, path, pretty=True)
    full = load_payload(path)
    fast = load_payload_scoring(path)
    assert len(full["venues"]["spot"]["depth_updates"]) > 0
    assert fast["venues"]["spot"]["depth_updates"] == []
    assert fast["venues"]["futures"]["depth_updates"] == []
    assert np.array_equal(predict_percentiles(full), predict_percentiles(fast))


def test_asset_flag_selects_symbol_schedule_and_database(monkeypatch, tmp_path):
    seen: dict = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr("fetch_payload.run_capture", fake_run)
    assert main(["--asset", "eth", "--count", "1"]) == 0
    assert seen["asset"] == "ETH"
    assert seen["database"] == DEFAULT_DATABASE / "ETH"
    assert main(["--asset", "SOL", "--count", "1", "--database", str(tmp_path)]) == 0
    assert seen["asset"] == "SOL"
    assert seen["database"] == tmp_path
    assert main(["--asset", "BTC", "--count", "1"]) == 0
    assert seen["asset"] == "BTC"
    assert seen["database"] == DEFAULT_DATABASE / "BTC"
    with pytest.raises(SystemExit):
        main(["--asset", "DOGE", "--count", "1"])
