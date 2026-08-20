"""Synth Ultra model package.

The evaluation entrypoint is ``predict_percentiles`` on ``synth_ultra.model``.
Docker sets ``VHFT_MINER_ENTRYPOINT=synth_ultra.model``.
"""

from synth_ultra.model import predict_percentiles

__all__ = ["predict_percentiles"]
