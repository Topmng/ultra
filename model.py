"""Root import: `from model import predict_percentiles`.

The submission image entrypoint is ``synth_ultra.model`` via
``VHFT_MINER_ENTRYPOINT``. This module re-exports the same function.
"""

from synth_ultra.model import predict_percentiles

__all__ = ["predict_percentiles"]
