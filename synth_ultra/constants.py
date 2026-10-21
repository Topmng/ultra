"""Payload-contract names and the asset helpers.

The values live in ``synth_ultra.config``; this module re-exports the
contract part of it for the harness, the tests, and the fetch scripts.
"""

from __future__ import annotations

import math

from synth_ultra.config import (
    ASSETS,
    BOOK_TICKER_WINDOW_S,
    BTC_REF_PX,
    CANDLE_WINDOW_S,
    CLOCK_INTERVAL_MS,
    DEPTH_BAND_USD,
    DEPTH_BANDS,
    DEPTH_LEVELS,
    EVENT_DELAY_MAX_MS,
    EVENT_WINDOW_LEN_MS,
    EVENT_WINDOW_OPEN_MS,
    HORIZON_SECONDS,
    LIQUIDATION_WINDOW_S,
    NUM_PERCENTILES,
    REFERENCE_ASSET,
    SCHEMA_VERSION,
    TRADE_WINDOW_S,
    TRIGGER_KINDS,
    TRIGGER_VENUES,
    AssetSpec,
)

__all__ = [
    "ASSETS",
    "AssetSpec",
    "BOOK_TICKER_WINDOW_S",
    "BTC_REF_PX",
    "CANDLE_WINDOW_S",
    "CLOCK_INTERVAL_MS",
    "DEPTH_BAND_USD",
    "DEPTH_BANDS",
    "DEPTH_LEVELS",
    "EVENT_DELAY_MAX_MS",
    "EVENT_WINDOW_LEN_MS",
    "EVENT_WINDOW_OPEN_MS",
    "HORIZON_SECONDS",
    "LIQUIDATION_WINDOW_S",
    "NUM_PERCENTILES",
    "SCHEMA_VERSION",
    "TRADE_WINDOW_S",
    "TRIGGER_KINDS",
    "TRIGGER_VENUES",
    "asset_spec",
    "price_scale",
]


def asset_spec(asset: str) -> AssetSpec:
    spec = ASSETS.get(str(asset or ""))
    if spec is None:
        raise ValueError(f"unknown asset {asset}")
    return spec


def price_scale(asset: str, px: float) -> float:
    """Scale fitted BTC dollar constants. BTC itself stays at 1."""
    if str(asset or REFERENCE_ASSET) == REFERENCE_ASSET:
        return 1.0
    if not math.isfinite(px) or px <= 0.0:
        return 1.0
    return float(px) / BTC_REF_PX
