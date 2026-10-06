"""Inference constants copied into the miner image.

Schema windows, trigger kinds, latency budget, and scoring grids live in
``harness.constants``. That module is local-only and is not part of the image.
"""

NUM_PERCENTILES = 100
HORIZON_SECONDS = 10
