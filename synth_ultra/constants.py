"""Inference constants copied into the miner image.

Window lengths, trigger kinds, and the asset table are part of the payload
contract. Scoring grids that are only used locally stay in ``harness.constants``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

NUM_PERCENTILES = 100
HORIZON_SECONDS = 10

SCHEMA_VERSION = 5
DEPTH_LEVELS = 20
CANDLE_WINDOW_S = 3600
TRADE_WINDOW_S = 60
BOOK_TICKER_WINDOW_S = 60
LIQUIDATION_WINDOW_S = 3600
DEPTH_BANDS = 100
# BTC band width. ETH, SOL, and XRP use AssetSpec.bucket_usd.
DEPTH_BAND_USD = 10.0
TRIGGER_KINDS = ("time", "event", "event_delayed")
TRIGGER_VENUES = ("spot", "futures")

# Each asset has a 20s slot. BTC is :00/:20/:40, then ETH +5s, SOL +10s, XRP +15s.
# The event window opens 10.25s after that slot and stays open for 4s.
CLOCK_INTERVAL_MS = 20_000
EVENT_WINDOW_OPEN_MS = 10_250
EVENT_WINDOW_LEN_MS = 4_000
EVENT_DELAY_MAX_MS = 500

# Dollar thresholds in the forecast were fit on BTC. Other assets scale them
# by price / BTC_REF_PX so a fixed dollar width does not follow the price down.
BTC_REF_PX = 85_000.0


@dataclass(frozen=True)
class AssetSpec:
    symbol: str
    tick: float
    futures_tick: float
    bucket_usd: float
    slot_offset_ms: int


ASSETS: dict[str, AssetSpec] = {
    "BTC": AssetSpec("BTCUSDT", 0.01, 0.1, 10.0, 0),
    "ETH": AssetSpec("ETHUSDT", 0.01, 0.01, 0.50, 5_000),
    "SOL": AssetSpec("SOLUSDT", 0.01, 0.01, 0.03, 10_000),
    "XRP": AssetSpec("XRPUSDT", 0.0001, 0.0001, 0.0003, 15_000),
}


def asset_spec(asset: str) -> AssetSpec:
    spec = ASSETS.get(str(asset or ""))
    if spec is None:
        raise ValueError(f"unknown asset {asset}")
    return spec


def price_scale(asset: str, px: float) -> float:
    """Scale fitted BTC dollar constants. BTC itself stays at 1."""
    if str(asset or "BTC") == "BTC":
        return 1.0
    if not math.isfinite(px) or px <= 0.0:
        return 1.0
    return float(px) / BTC_REF_PX
