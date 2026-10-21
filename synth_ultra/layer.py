"""Last location and width step: live book and trade signals the ladder does not see.

Each asset has a short list of terms in ``layer_table.LAYER``. A shift term
moves every off-touch price by ``coef * z`` times either the ladder's own
width (``w``) or that asset's typical width (``c``). A scale term multiplies
the off-touch distances from the price by ``exp(coef * z)``. Prices on the
touch are the stillness point mass and stay where they are.

``z`` is a robust standard score: the raw signal less its saved-prompt
median, over its median absolute deviation, clipped at six and divided by
three. A missing signal scores zero.
"""

from __future__ import annotations

import math

import numpy as np

from synth_ultra.constants import NUM_PERCENTILES, asset_spec
from synth_ultra.layer_table import LAYER

_BASIS_GRID = np.arange(-59_900, 1, 100, dtype=np.int64)
_LOG_CLIP = 1.5


def _col(d: dict | None, key: str, dtype) -> np.ndarray:
    v = (d or {}).get(key)
    if v is None:
        return np.zeros(0, dtype=dtype)
    a = v if isinstance(v, np.ndarray) else np.asarray(v, dtype=dtype)
    return a.reshape(-1)


def _book_gap(snapshot: dict | None, notional: float) -> float:
    """How much farther the bid book must be walked than the ask book to reach ``notional`` dollars.

    Positive means the bid side is thinner. Zero when either side is missing or the
    touch already holds that much size.
    """
    if not snapshot:
        return 0.0
    bids = snapshot.get("bids")
    asks = snapshot.get("asks")
    if bids is None or asks is None:
        return 0.0
    bids = np.asarray(bids, dtype=np.float64)
    asks = np.asarray(asks, dtype=np.float64)
    if bids.ndim != 2 or asks.ndim != 2 or bids.shape[0] == 0 or asks.shape[0] == 0:
        return 0.0

    def dist(levels: np.ndarray) -> float:
        dol = np.cumsum(levels[:, 0] * levels[:, 1])
        i = int(np.searchsorted(dol, notional))
        if i >= levels.shape[0]:
            i = levels.shape[0] - 1
        return abs(float(levels[i, 0] - levels[0, 0]))

    return dist(bids) - dist(asks)


def _asof_idx(ts: np.ndarray, t: np.ndarray | int) -> np.ndarray:
    i = np.searchsorted(ts, t, side="right") - 1
    return np.clip(i, 0, ts.size - 1)


class _Book:
    __slots__ = ("ts", "bp", "ap", "bq", "aq", "ok")

    def __init__(self, bt: dict | None) -> None:
        self.ts = _col(bt, "recv_ts_ms", np.int64)
        self.bp = _col(bt, "bid_price", np.float64)
        self.ap = _col(bt, "ask_price", np.float64)
        self.bq = _col(bt, "bid_qty", np.float64)
        self.aq = _col(bt, "ask_qty", np.float64)
        n = self.ts.size
        self.ok = n >= 5 and self.bp.size == n and self.ap.size == n and self.bq.size == n and self.aq.size == n

    def mid(self, i) -> np.ndarray | float:
        return 0.5 * (self.bp[i] + self.ap[i])

    def micro_last(self) -> float:
        bp, ap, bq, aq = float(self.bp[-1]), float(self.ap[-1]), float(self.bq[-1]), float(self.aq[-1])
        return (bp * aq + ap * bq) / max(bq + aq, 1e-12)


def signals(payload: dict, now_ms: int, keys) -> dict[str, float]:
    """Raw signal values for ``keys``. Absent ones are left out."""
    keys = set(keys)
    out: dict[str, float] = {}
    venues = payload.get("venues") or {}
    sp = venues.get("spot") or {}
    fu = venues.get("futures") or {}
    sb = _Book(sp.get("book_ticker"))
    fb = _Book(fu.get("book_ticker"))
    if not (sb.ok and fb.ok):
        return out
    now = int(now_ms)
    trig = (payload.get("prompt") or {}).get("trigger") or {}
    out["venue_fut"] = 1.0 if trig.get("venue") == "futures" else 0.0
    s_mid, f_mid = float(sb.mid(-1)), float(fb.mid(-1))
    s_mic, f_mic = sb.micro_last(), fb.micro_last()
    if any(k.startswith("bdev") for k in keys):
        grid = now + _BASIS_GRID
        grid = grid[grid >= max(int(sb.ts[0]), int(fb.ts[0]))]
        if grid.size:
            basis = fb.mid(_asof_idx(fb.ts, grid)) - sb.mid(_asof_idx(sb.ts, grid))
            b_now = f_mid - s_mid
            for w in (5, 15, 30, 60):
                if f"bdev{w}" in keys:
                    out[f"bdev{w}"] = b_now - float(basis[-max(1, w * 10):].mean())
    for w in (250, 500, 1000, 2000, 5000):
        names = (f"fchg{w}", f"schg{w}", f"lead{w}")
        if any(n in keys for n in names):
            fc = f_mid - float(fb.mid(_asof_idx(fb.ts, now - w)))
            sc = s_mid - float(sb.mid(_asof_idx(sb.ts, now - w)))
            out[names[0]], out[names[1]], out[names[2]] = fc, sc, fc - sc
    out["s_micro_mid"] = s_mic - s_mid
    out["f_micro_mid"] = f_mic - f_mid
    for name, v, mic, bk in (("s", sp, s_mic, sb), ("f", fu, f_mic, fb)):
        tr = v.get("trades") or {}
        tp = _col(tr, "price", np.float64)
        if tp.size and any(k.startswith(name + "_") for k in keys):
            tq = _col(tr, "qty", np.float64)
            tm = _col(tr, "buyer_is_maker", bool)
            tt = _col(tr, "recv_ts_ms", np.int64)
            out[f"{name}_lastpx_vs_micro"] = float(tp[-1]) - mic
            out[f"{name}_last_side"] = -1.0 if bool(tm[-1]) else 1.0
            out[f"{name}_last_age"] = float(now - int(tt[-1]))
            for w in (500, 2000, 10_000):
                if any(f"{name}_{s}{w}" in keys for s in ("flow", "vol", "n")):
                    m = tt > now - w
                    q = tq[m]
                    out[f"{name}_flow{w}"] = float(np.where(tm[m], -q, q).sum())
                    out[f"{name}_vol{w}"] = float(q.sum())
                    out[f"{name}_n{w}"] = float(np.count_nonzero(m))
            if f"{name}_vwap10_vs_micro" in keys:
                m = tt > now - 10_000
                vol = float(tq[m].sum()) if m.any() else 0.0
                if vol > 0.0:
                    out[f"{name}_vwap10_vs_micro"] = float((tp[m] * tq[m]).sum()) / vol - mic
        d = v.get("depth_latest") or {}
        bids = d.get("bids")
        asks = d.get("asks")
        if bids is not None and asks is not None:
            bids = np.asarray(bids, dtype=np.float64)
            asks = np.asarray(asks, dtype=np.float64)
            if bids.ndim == 2 and asks.ndim == 2 and bids.shape[0] and asks.shape[0]:
                for k in (1, 3, 10, 20):
                    if f"{name}_dimb{k}" in keys:
                        bs = float(bids[:k, 1].sum())
                        a_s = float(asks[:k, 1].sum())
                        out[f"{name}_dimb{k}"] = (bs - a_s) / max(bs + a_s, 1e-12)
                out[f"{name}_dspan"] = float(asks[-1, 0] - bids[-1, 0])
        if f"{name}_dthin" in keys:
            out[f"{name}_dthin"] = _book_gap(v.get("depth_latest"), 25_000.0)
        for w in (1000, 5000, 10_000, 60_000):
            if f"{name}_nbt{w}" in keys:
                out[f"{name}_nbt{w}"] = float(np.count_nonzero(bk.ts > now - w))
        for w in (10, 60):
            if f"{name}_brng{w}" in keys:
                m = bk.ts > now - w * 1000
                mm = bk.mid(m)
                out[f"{name}_brng{w}"] = float(mm.max() - mm.min()) if mm.size else 0.0
    if any(k.startswith("c_") for k in keys):
        c = np.asarray(((sp.get("candles_1s") or {}).get("ohlcv")), dtype=np.float64)
        if c.ndim == 2 and c.shape[0] >= 120:
            cl = c[:, 3]
            d10 = np.abs(cl[10:] - cl[:-10])
            for w in (60, 300, 900, 3600):
                out[f"c_mad{w}"] = float(d10[-w:].mean()) if d10.size >= w else float(d10.mean())
            rng1 = c[:, 1] - c[:, 2]
            for w in (10, 60, 300):
                out[f"c_rng{w}"] = float(rng1[-w:].mean())
    if any(k.startswith("fc_") for k in keys):
        fc = np.asarray(((fu.get("candles_1s") or {}).get("ohlcv")), dtype=np.float64)
        if fc.ndim == 2 and fc.shape[0] >= 120:
            fd10 = np.abs(fc[10:, 3] - fc[:-10, 3])
            out["fc_mad60"] = float(fd10[-60:].mean())
            out["fc_mad300"] = float(fd10[-300:].mean())
    if any(k.startswith("liq_") for k in keys):
        liq = fu.get("liquidations") or {}
        lt = _col(liq, "recv_ts_ms", np.int64)
        lq = _col(liq, "qty", np.float64)
        lb = _col(liq, "is_buy", bool)
        if lt.size == lq.size == lb.size:
            for w in (10_000, 60_000, 600_000):
                m = lt > now - w
                out[f"liq_signed{w // 1000}"] = float(np.where(lb[m], lq[m], -lq[m]).sum())
                out[f"liq_n{w // 1000}"] = float(np.count_nonzero(m))
    return out


def _z(v: float, c: float, d: float) -> float:
    if not math.isfinite(v):
        return 0.0
    return min(max((v - c) / d, -6.0), 6.0) / 3.0


def term_value(
    kind: str, src: str, s: dict, off: float, width: float, c: float, d: float, f: dict, px: float
) -> float:
    """Standardized signal for one term, before its gate and unit."""
    if kind == "f_robust":
        return _z(float(f.get(src, math.nan)), c, d)
    if kind == "f_logw":
        v = float(f.get(src, math.nan)) * px
        return _z(math.log(max(v, 1e-12) / width), c, d) if math.isfinite(v) else 0.0
    if kind == "f_log1p":
        v = float(f.get(src, 0.0))
        return _z(math.log1p(max(v if math.isfinite(v) else 0.0, 0.0)), c, d)
    if kind == "raw":
        v = s.get(src, 0.0)
        return v if math.isfinite(v) else 0.0
    if kind == "robust":
        return _z(s.get(src, math.nan), c, d)
    if kind == "logw":
        v = s.get(src, math.nan)
        return _z(math.log(max(v, 1e-12) / width), c, d) if math.isfinite(v) else 0.0
    if kind == "log1p":
        v = s.get(src, 0.0)
        v = v if math.isfinite(v) else 0.0
        return _z(math.log1p(max(v, 0.0)), c, d)
    if kind == "off":
        return off
    if kind == "absoff":
        return abs(off)
    if kind == "absimb":
        v = s.get("s_dimb1", 0.0)
        return abs(v) if math.isfinite(v) else 0.0
    return 1.0


def apply(
    q: np.ndarray, px: float, asset: str, trigger: str, payload: dict, now_ms: int, f: dict | None = None
) -> np.ndarray:
    spec = LAYER.get(asset)
    if not spec:
        return q
    med = 0.5 * (float(q[49]) + float(q[50]))
    width = float(np.abs(q - med).sum()) / NUM_PERCENTILES
    if not (math.isfinite(med) and math.isfinite(width) and width > 0.0):
        return q
    s = signals(payload, now_ms, spec["keys"])
    off = (med - px) / width
    ev = 0.0 if trigger == "time" else 1.0
    fut = 1.0 if s.get("venue_fut", 0.0) == 1.0 else 0.0
    gates = {
        "": 1.0,
        "ev": ev,
        "evf": ev * fut,
        "evs": ev * (1.0 - fut),
        "delayed": 1.0 if trigger == "event_delayed" else 0.0,
    }
    f = f or {}
    wref = float(spec["wref"])
    shift = 0.0
    logs = 0.0
    for t in spec["terms"]:
        g = gates[t["gate"]]
        if g == 0.0:
            continue
        v = g * term_value(t["kind"], t["src"], s, off, width, t["c"], t["d"], f, px)
        if t["op"] == "shift":
            shift += t["coef"] * v * (width if t["unit"] == "w" else wref)
        else:
            logs += t["coef"] * v
    sc = math.exp(min(max(logs, -_LOG_CLIP), _LOG_CLIP))
    if not (math.isfinite(shift) and math.isfinite(sc)):
        return q
    tick = asset_spec(asset).tick
    free = np.abs(q - px) > 0.5 * tick
    x = np.where(free, px + sc * (q - px) + shift, q)
    x.sort()
    if not (math.isfinite(float(x.sum())) and float(x[0]) > 0.0):
        return q
    return x
