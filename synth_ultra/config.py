"""Every tunable or contract constant of the forecast, in one place.

Sections
--------
1. Data processing  payload contract, feature windows and thresholds
2. Simulation       output grid, mixture quantile table, numeric guards
3. Shared model     the mechanisms common to BTC, ETH, SOL and XRP
4. Asset-specific   tick table and per-asset parameter values
5. Evaluation       scoring and research-protocol constants (not read at inference)

Modules import these names; nothing below is redefined elsewhere. BTC is the
reference asset: dollar-denominated constants were fitted on BTC and other
assets rescale them by ``price / BTC_REF_PX`` (``constants.price_scale``).
Values carried over from the v2.4 model keep its fitted numbers exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# =============================================================================
# 1. Data processing
# =============================================================================

# --- payload contract (input.md, SPECIFICATION.md) ---
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

# --- feature extraction (features.py) ---
# 1s log-return EWMA vol on the candle tail: slow (lambda, length), fast (lambda, length).
VOL_EWMA_LAM = 0.94
VOL_EWMA_N = 120
VOL_FAST_LAM = 0.85
VOL_FAST_N = 90
# Spot share of the spot/futures 1s vol blend.
VOL_SPOT_WEIGHT = 0.6
# Trade-flow windows: last N prints and the last window in ms.
FLOW_LAST_N = 50
FLOW_WINDOW_MS = 2000
# Book-ticker lookback for momentum / range / realized vol, and its sample cap.
BOOK_LOOKBACK_MS = 12_000
BOOK_RV_SAMPLES = 512
# Top-of-depth levels for the "top" resting-size imbalance.
DEPTH_TOP_N = 5
# Liquidations: signed qty over the last LIQ_RECENT_MS, tanh units in BTC coins.
LIQ_RECENT_MS = 10_000
LIQ_QTY_UNIT = 0.25
LIQ_HOUR_UNIT = 1.0
# Depth bands older than this are unused; reach is this many bands from the mid.
BAND_MAX_AGE_MS = 90_000
BAND_REACH_BANDS = 20.0
# A reference-price candidate more than this factor from the candle close is a bad tick.
REF_MAX_RATIO = 2.0

# --- stillness features (stillness.py) ---
STILL_GRID_MS = 100
STILL_GRID_N = 600
# Changes of twice the mid on BTC: spot moves in 0.01 ticks, futures in 0.1.
STILL_SPOT_STEP = 0.008
STILL_FUT_STEP = 0.08
STILL_RANGE_TICK = 0.011
STILL_RANGE_LONG = 1790
STILL_RANGE_SHORT = 290

# =============================================================================
# 2. Simulation (output grid and the distribution shapes it is read from)
# =============================================================================

# Centered-quantile probabilities, q_i = (2i-1)/200.
TAU = (2.0 * np.arange(1, NUM_PERCENTILES + 1) - 1.0) / 200.0
# Standardized quantiles of (1-w)*Laplace(0,1) + w*Laplace(0,r), w=0.15, r=3.
# Precomputed so inference stays numpy-only (no scipy in the miner image).
MIX_W = 0.15
MIX_R = 3.0
MIX_Z = np.array(
    [
        -8.195296766019496,
        -5.290455382749503,
        -4.189473492373635,
        -3.558928761226856,
        -3.130018036741951,
        -2.809478451891556,
        -2.5554904666449834,
        -2.3461068991409015,
        -2.1685150645304785,
        -2.014637822514757,
        -1.879079631466685,
        -1.7580695625741907,
        -1.6488748569423617,
        -1.5494554430067913,
        -1.4582502010738922,
        -1.3740392432906956,
        -1.2958520713528672,
        -1.2229045008325568,
        -1.1545542250063079,
        -1.0902688057552605,
        -1.0296021595823457,
        -0.9721769807684978,
        -0.9176713964804792,
        -0.865808692117394,
        -0.8163492998117764,
        -0.7690844793903884,
        -0.7238312817274705,
        -0.6804284955014599,
        -0.6387333564233322,
        -0.5986188536579183,
        -0.559971508386832,
        -0.5226895288992309,
        -0.48668126838854353,
        -0.4518639279418573,
        -0.4181624595326374,
        -0.3855086332303556,
        -0.3538402400763217,
        -0.3231004076881543,
        -0.29323701004778674,
        -0.26420215637718525,
        -0.2359517467597128,
        -0.2084450843380587,
        -0.1816445356867318,
        -0.15551523237167816,
        -0.1300248078599066,
        -0.10514316488530429,
        -0.0808422691479815,
        -0.05709596585577331,
        -0.03387981614877673,
        -0.01117095088164434,
        0.01117095088164428,
        0.03387981614877675,
        0.05709596585577319,
        0.08084226914798152,
        0.10514316488530434,
        0.13002480785990664,
        0.1555152323716778,
        0.18164453568673153,
        0.20844508433805856,
        0.23595174675971262,
        0.26420215637718536,
        0.29323701004778674,
        0.323100407688154,
        0.3538402400763216,
        0.38550863323035545,
        0.41816245953263753,
        0.45186392794185726,
        0.48668126838854336,
        0.5226895288992306,
        0.5599715083868319,
        0.5986188536579181,
        0.6387333564233322,
        0.6804284955014599,
        0.7238312817274708,
        0.7690844793903882,
        0.8163492998117761,
        0.8658086921173938,
        0.9176713964804795,
        0.9721769807684977,
        1.029602159582346,
        1.0902688057552605,
        1.1545542250063072,
        1.2229045008325565,
        1.2958520713528667,
        1.374039243290695,
        1.458250201073892,
        1.5494554430067913,
        1.6488748569423621,
        1.758069562574191,
        1.8790796314666847,
        2.0146378225147576,
        2.1685150645304794,
        2.346106899140902,
        2.5554904666449842,
        2.8094784518915548,
        3.130018036741951,
        3.558928761226851,
        4.189473492373635,
        5.290455382749499,
        8.195296766019453,
    ],
    dtype=np.float64,
)
# Numeric guards.
EPS = 1e-12
MIN_PRICE = 1e-8
# Width bounds in reference-asset dollars (scaled by price_scale on other assets).
MIN_SIGMA_PX = 0.5
MAX_SIGMA_PX = 800.0
# Fallback ladder width: max(FALLBACK_SIGMA_PX * price_scale, FALLBACK_SIGMA_REL * price).
FALLBACK_SIGMA_PX = 8.0
FALLBACK_SIGMA_REL = 1e-4

# =============================================================================
# 3. Shared model parameters
# =============================================================================

# --- linear location / width from book and trade features ---
B_MU = np.array(
    [
        0.16219725476591365,
        3.0392548397260206e-05,
        1.2921967376416503e-05,
        0.08772678796847251,
        -0.057379556931711054,
        -0.042592960514635785,
        0.14437278080879343,
        -5.040059876530939e-06,
        -2.9598223746399192e-05,
        8.80503231208016e-07,
        6.543536873659752e-06,
        3.819089008691624e-05,
        -3.851394849050198e-05,
        0.025639229462532765,
        0.0,
        7.80781830077767e-05,
        0.2225417214564695,
        -0.09424387369323395,
        0.06611178032601009,
        0.0,
        1.3194603589745686e-05,
        9.25776900504624e-06,
        6.419523696500204e-06,
        9.704432289751613e-06,
        -7.61375359503243e-05,
        8.219239603035196e-09,
    ],
    dtype=np.float64,
)
B_ABS = np.array(
    [
        0.31128035091375017,
        -0.04612335008069495,
        -0.27015375057439384,
        0.05338090769963195,
        -0.10630344326911181,
        0.04195075214608767,
        0.48300301336701507,
        0.14988622199220886,
        0.05522149798895418,
        0.28790522175963235,
        -0.2171458492361658,
        7.440005212080741e-06,
        1.5006480516698034e-05,
        5.258077361233235e-06,
        -3.602539619772853,
        -2.7531539341820717e-05,
        0.00029563500111100656,
        7.860095041094238e-05,
        0.018052856090945528,
        -0.5633117688000335,
        -0.09328221925825791,
        0.11197666410197983,
        -0.19790213408961582,
        0.3465817220321161,
        -5.886429731585097e-08,
        -1.194041355427959e-06,
        1.1416268283201004e-06,
        1.3031647192685995e-06,
        -5.381035088350931e-07,
        -0.0004201840209259798,
        -1.4343343808409522e-05,
    ],
    dtype=np.float64,
)
KA = 0.75  # weight of the |return| regression in the width
KB = 0.20  # weight of |mu| in the width
FL = 5e-5  # width floor in log-return units
SH = 1.20  # location regression gain
FR = 0.15  # width floor: share of the 10s book range
FV = 0.05  # width floor: share of the 10s book realized vol
SM = 1.04  # width multiplier
C1_TANH_SCALE = 3000.0  # tanh(c1 * scale) location feature
STALE_MS_CUT = 250.0  # stale-feed indicator in the width regression
QTY_LOG_DIV = 10.0  # log1p(qty) / QTY_LOG_DIV width features
MU_CLIP_SIGMAS = 5.0  # |location| is capped at this many widths

# --- width blends toward realized hour moves ---
BLEND_REALIZED_POW = 0.8  # raise a too-tight fitted width toward the hour scale
# Log-blend weights of (fitted, EWMA 10s, 60s rv, 10s range) widths, and bias.
VOL_W = (0.7199761, 0.18929481, 0.06551382, 0.02521527)
VOL_LOG_BIAS = 0.04421702415367945
VOL_LOG_FLOOR = 0.3  # reference dollars
HOUR_EWMA_LAM = 0.98  # EWMA of |10s log return| over the hour
RV60_N = 60  # seconds of 1s returns in the 60s realized vol
LIQ_HOUR_WIDEN = 0.20  # width multiplier per unit of hour liquidation activity
# Regime map: fac = clip(REG_A + REG_B * s_recent / sigma, REG_A, 1) over REG_N moves.
REG_A = 0.80
REG_B = 0.30
REG_N = 180
# Flex: location pull and width step from the last FLEX_N 10s moves.
FLEX_N = 600
FLEX_FADE_LO = 0.3
FLEX_FADE_HI = 1.5
MU_EXCESS = 0.60
MU_PULL = 0.40
MU_FLOOR = 0.45
WIDE_POW = 0.05
WIDE_FLOOR = 0.96

# --- payload book-sample quantiles mixed into the ladder ---
BOOK_PIVOT = 0.65
BOOK_SLOPE = 0.35
BOOK_CAP = 0.18
BOOK_PAIRS = 1600  # cap on paired 10s book moves (latency)
BOOK_MIN_TICKS = 80
BOOK_MIN_PAIRS = 40
BOOK_MIN_GAP_MS = 9_000  # a pair must be at least this far apart
HIST_JUMP = 12.0  # hour p99/median 10s move above this is a single-outlier regime
HIST_JUMP_Q = 0.99
HIST_W = 0.27
CALM_CORE = 0.98  # shrink of the central quantiles in an outlier hour

# --- short-horizon location shifts (fractions of the width) ---
OFI_K = 0.35  # last-second quote order-flow imbalance
OFI_MS = 1000
IMB_K = 0.35  # top-of-book imbalance on clock prompts
IMB_EVENT = 0.50  # top-of-book imbalance on trade-triggered prompts
LOUD_IMB = 0.60  # |imbalance| above which the book sample is down-weighted
LOUD_BOOK = 0.60
LIQ_K = 0.04  # 10s signed liquidation direction
BAND_K = 0.03  # re-anchored depth-band imbalance
MU_KEEP = 0.85  # share of the fitted drift kept on the mixture path
C1_KEEP = 0.60  # share of the last 1s candle move added

# --- seasonal width (season_table.py, generated by scripts/update_season.py) ---
SEAS_W = 0.35
SEAS_CLIP = 0.35
SEAS_QUIET_REL = 0.70
SEAS_QUIET_CAP = 0.85
SEAS_QUIET_PIVOT = 1.00
SEAS_QUIET_SLOPE = 1.00

# --- candle distribution on clock prompts ---
CANDLE_WIN_S = 1800  # half hour of 1s closes
CANDLE_MIN_RETURNS = 40
CANDLE_BINS = 50
CANDLE_Q = 0.9  # quantile of |10s move| that drives the candle weight
ACT_EPS = 1e-6
ACT_FAST = 60
BLEND_PIVOT = 7.5  # reference dollars
BLEND_TEMP = 9.0  # reference dollars
BLEND_BETA = 0.12
MU_ALPHA_CAP = 1.0

# --- trade-triggered prompts ---
# Inner-quartile widening tent: zero outside (0.10, 0.90) and at the median.
EVENT_CENTER = 0.08
EVENT_TENT_KNOTS = (0.10, 0.25, 0.50, 0.75, 0.90)
FLOW_K = 0.10  # futures taker flow in the last FLOW_MS, per standard deviation
FLOW_SD = 5.644958570072203  # reference-asset coins
FLOW_CLIP = 4.0
FLOW_MS = 1000
FM1_K = -0.08  # pull back of the spent 1s futures move, per standard deviation
FM1_SD = 7.34486046119216e-05  # reference-asset 1s log move
FM1_CLIP = 4.0
FM1_LIVE_N = 180  # seconds of candles for a non-reference asset's own 1s scale
FM1_SD_LO = 0.5
FM1_SD_HI = 12.0
# A trade that moves the price through a level reaches the trade tape before the
# spot book ticker shows the new touch, so on trade-triggered prompts the quote
# microprice anchor can be stale. The moved (non-still) part of the ladder is
# re-centred this share of the way from the quote to the last spot trade.
# One value for all assets: fit on prompts before EVAL_SPLIT_MS (pooled), see
# experiments/RESULTS.md (H1b). 0 disables the step.
EVENT_TRADE_ANCHOR = 0.7

# --- stillness point mass (stillness.py) ---
# Logistic weights, intercept first, then stillness.FEATURES order. Clock and
# trade-triggered prompts each have their own vector.
STILL_W_TIME = np.array(
    [
        -4.61846527451775,
        0.17034772984186247,
        0.2813703561617492,
        -0.15965766031065734,
        -0.04949323887833651,
        0.6605209888903941,
        0.13005700670299267,
        -20.758785373235124,
        -0.053866749363991946,
        -0.40563421464038446,
        1.9722942449642995,
        -0.21931392073370692,
        -0.4799491646634374,
        -1.0101467322622217,
    ],
    dtype=np.float64,
)
STILL_W_EVENT = np.array(
    [
        -5.293526458905508,
        1.2359207801555567,
        -0.7657115123577866,
        -0.35387194617541284,
        -0.2549507079896938,
        1.0323707288541988,
        0.19539287504904634,
        -1.8838370215557418,
        0.017228787450030474,
        -0.41789201765064954,
        -0.16554390432665644,
        1.3447109825428714,
        -0.48082099995511424,
        -8.983843865291522,
    ],
    dtype=np.float64,
)
STILL_GAIN = 1.0
STILL_P_MAX = 0.9
STILL_KAPPA = 0.75
STILL_SHIFT = 1.0
STILL_HOLE = 0.05

# --- final width check against the payload's own recent 10s moves ---
LIVE_POW = 0.50
LIVE_LO = 0.90
LIVE_HI = 1.10
LIVE_N = 180
LIVE_MIN_WIDTH = 0.5  # reference dollars
ATOM = 0.02  # reference dollars: prices this close to the anchor are the stillness mass

# =============================================================================
# 4. Asset-specific parameters
# =============================================================================

REFERENCE_ASSET = "BTC"
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

# Final width multiplier on the finished ladder (stillness prices kept on the
# touch). The shared value leaves the ladder as is.
WIDTH = 1.0

# Asset-specific values of shared parameters. A name missing for an asset uses
# the shared value from section 3 (or WIDTH above). Only names in
# ASSET_TUNABLE are read per asset by the model; anything else is rejected.
ASSET_TUNABLE = (
    "WIDTH",
    "STILL_GAIN",
    "STILL_KAPPA",
    "STILL_SHIFT",
    "STILL_HOLE",
    "SEAS_W",
    "LIVE_POW",
    "FLOW_K",
    "HIST_W",
    "C1_KEEP",
    "MU_KEEP",
    "REG_A",
    "IMB_K",
    "OFI_K",
)
# WIDTH values are v2.4's. The other BTC and ETH values were chosen by
# coordinate descent on prompts before EVAL_SPLIT_MS only and kept because the
# later prompts improved (experiments/RESULTS.md, H6). SOL and XRP did not
# improve out of sample and keep the shared values.
ASSET_PARAMS: dict[str, dict[str, float]] = {
    "BTC": {
        "STILL_GAIN": 0.7,
        "STILL_SHIFT": 1.25,
        "STILL_HOLE": 0.1,
        "SEAS_W": 0.5,
        "LIVE_POW": 0.0,
        "FLOW_K": 0.2,
    },
    "ETH": {
        "WIDTH": 1.25,
        "STILL_GAIN": 0.7,
        "HIST_W": 0.15,
        "C1_KEEP": 0.0,
        "REG_A": 0.7,
        "MU_KEEP": 1.0,
        "IMB_K": 0.15,
        "OFI_K": 0.15,
    },
    "SOL": {"WIDTH": 1.50},
    "XRP": {"WIDTH": 1.60},
}


def asset_param(asset: str, name: str) -> float:
    """``name`` for ``asset``: its own value when set, else the shared one."""
    vals = ASSET_PARAMS.get(asset)
    if vals is not None:
        v = vals.get(name)
        if v is not None:
            return v
    return _SHARED[name]


# =============================================================================
# 5. Evaluation (local scoring and research protocol; not read at inference)
# =============================================================================

LATENCY_BUDGET_S = 0.005
# A target instant with no spot book tick in the preceding second is dropped.
SPOT_FEED_LIVE_MS = 1000
EVAL_WINDOW_MS = 24 * 3_600_000  # rolling CRPS window
EVAL_STEP_MS = 3_600_000  # rolling window step
EVAL_MIN_PROMPTS = 50  # a window with fewer scored prompts is skipped
EVAL_SPLIT_MS = 1792281600000  # 2026-10-18 00:00 UTC: knob fits use prompts before this

assert MIX_Z.shape == (NUM_PERCENTILES,) and math.isclose(float(TAU[0]), 0.005)
for _a, _vals in ASSET_PARAMS.items():
    assert _a in ASSETS, f"ASSET_PARAMS names unknown asset {_a}"
    for _k in _vals:
        assert _k in ASSET_TUNABLE, f"{_a}.{_k} is not read per asset (ASSET_TUNABLE)"


def _shared() -> dict[str, float]:
    return {k: globals()[k] for k in ASSET_TUNABLE}


# Shared values of the per-asset names. ``refresh_shared`` re-reads them after
# a research script overrides a section-3 value in place.
_SHARED = _shared()


def refresh_shared() -> None:
    _SHARED.clear()
    _SHARED.update(_shared())
