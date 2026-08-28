# Synth Ultra submission image.
# Synth's serving loop imports VHFT_MINER_ENTRYPOINT and calls predict_percentiles.
# Build for the evaluation platform:
#   docker build --platform linux/amd64 -t synth-ultra:v1 .

FROM --platform=linux/amd64 ghcr.io/synthdataco/vhft-miner-base:v2
# Frozen digest for this round (optional pin):
# FROM --platform=linux/amd64 ghcr.io/synthdataco/vhft-miner-base@sha256:66dbeb6f64cab66383333b1499b0fe45a186e5d7bf2f1ade2ddacc3342e1d6ca

ENV PYTHONPATH=/app \
    VHFT_MINER_ENTRYPOINT=synth_ultra.model

COPY synth_ultra/ /app/synth_ultra/
COPY model.py /app/model.py
