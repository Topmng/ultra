# Synth Ultra submission image.
# Synth's serving loop imports VHFT_MINER_ENTRYPOINT and calls predict_percentiles.
# Build for the evaluation platform:
#   docker build --platform linux/amd64 -t synth-ultra:v1 .

FROM --platform=linux/amd64 ghcr.io/synthdataco/vhft-miner-base:v1
# Frozen digest for this round (optional pin):
# FROM --platform=linux/amd64 ghcr.io/synthdataco/vhft-miner-base@sha256:47e3a095ae495dec695bc69ba613725e9e83fc7855bf97e7a3f35179a5e1c25d

ENV PYTHONPATH=/app \
    VHFT_MINER_ENTRYPOINT=synth_ultra.model

COPY synth_ultra/ /app/synth_ultra/
COPY model.py /app/model.py
