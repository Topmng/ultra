# CPU-only Synth Ultra submission image.
# Inference has no network; the payload is the only input.
# Official base image / entrypoint may change when Synth publishes them.

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY synth_ultra ./synth_ultra
COPY model.py .

RUN python -m synth_ultra.validate --strict --rounds 32 --warmup 8

USER 65534:65534

EXPOSE 8080

CMD ["python", "-m", "synth_ultra.runtime"]
