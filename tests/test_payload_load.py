from __future__ import annotations

import time

from synth_ultra.payload import (
    _load_depth_snapshot_csv,
    _load_trades_csv,
    _npz_path,
)


def test_trades_csv_parses_maker_and_caches(tmp_path):
    path = tmp_path / "trades.csv"
    path.write_text(
        "agg_trade_id,ts_ms,ts_utc,price,qty,buyer_is_maker\n"
        "1,100,t,10.0,0.5,True\n"
        "2,200,t,11.0,0.25,False\n"
        "3,300,t,12.0,1.0,1\n",
        encoding="utf-8",
    )
    _load_trades_csv.cache_clear()
    table = _load_trades_csv(str(path))
    assert table is not None
    assert table["ts_ms"].tolist() == [100, 200, 300]
    assert table["buyer_is_maker"].tolist() == [True, False, True]
    cache = _npz_path(path)
    assert cache.is_file()
    _load_trades_csv.cache_clear()
    t0 = time.perf_counter()
    again = _load_trades_csv(str(path))
    assert time.perf_counter() - t0 < 0.5
    assert again is not None
    assert again["qty"].tolist() == [0.5, 0.25, 1.0]


def test_depth_snapshot_groups_and_caches(tmp_path):
    path = tmp_path / "depth.csv"
    path.write_text(
        "recv_ts_ms,recv_ts_utc,update_id,event_ts_ms,transaction_ts_ms,side,level,price,qty\n"
        "1000,t,9,,,bid,0,100.0,1.0\n"
        "1000,t,9,,,bid,1,99.0,2.0\n"
        "1000,t,9,,,ask,0,101.0,1.5\n"
        "2000,t,10,2001,2002,ask,0,102.0,3.0\n"
        "2000,t,10,2001,2002,bid,0,98.0,4.0\n",
        encoding="utf-8",
    )
    _load_depth_snapshot_csv.cache_clear()
    snaps = _load_depth_snapshot_csv(str(path))
    assert snaps is not None
    assert len(snaps) == 2
    assert int(snaps[0]["recv_ts_ms"]) == 1000
    assert snaps[0]["bids"].shape == (2, 2)
    assert snaps[0]["event_ts_ms"] is None
    assert snaps[1]["event_ts_ms"] == 2001
    assert snaps[1]["bids"][0, 0] == 98.0
    _load_depth_snapshot_csv.cache_clear()
    cached = _load_depth_snapshot_csv(str(path))
    assert cached is not None
    assert cached[1]["asks"][0, 1] == 3.0
