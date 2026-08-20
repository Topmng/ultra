# Synth Ultra

An ultra-low-latency forecasting competition: predict the distribution of the
**BTC price 10 seconds ahead**, from live Binance order-book and trade data, with
a model that runs in **under 5 ms**.

Synth builds predictive models and applies them to financial markets. Synth Ultra
targets the shortest horizons — short-term dynamics such as autocorrelation and
order flow, and extracting the best estimate of the current price from deep,
fast-moving crypto order books — with models that feed directly into Synth's
trading systems.

## What you build

A model that:

- predicts **100 percentiles** of the BTC price **10 seconds** into the future,
- runs in **under 5 ms** per prediction (strictly enforced), and
- meets the technical specification so it runs directly in Synth's evaluation
  environment.

Models are evaluated continuously on live market data; scores are visible to all
participants.

## Data available to the model

Each prediction, the model receives — for Binance **spot** and **USDT-M
futures** (BTCUSDT):

- the last **1 hour** of 1-second candles (OHLCV),
- an order-book snapshot to a set depth from ~**60 s** prior, every subsequent
  order-book update, and the latest snapshot,
- **60 s** of aggregate trades, and
- **60 s** of book-ticker updates (best bid/ask price and size).

Full schema: [`input.md`](input.md). Full competition spec:
[`SPECIFICATION.md`](SPECIFICATION.md).

## Rewards & participation

- **25%** of total subnet miner rewards are allocated to Synth Ultra.
- Initially limited to **10 participants**. All participants must be onboarded and
  complete the participation form before submitting a model.

## Objective

Build market-beating, ultra-low-latency predictive models that can generalize
across assets and feed directly into Synth's trading systems — competing at the
shortest market horizons, where HFT firms operate.

## This repository

A CPU-only miner that implements the `predict_percentiles` contract from
[`SPECIFICATION.md`](SPECIFICATION.md) and consumes the payload in
[`input.md`](input.md):

```python
def predict_percentiles(payload: dict) -> np.ndarray  # shape (100,), float64
```

The submission image is built `FROM` `ghcr.io/synthdataco/vhft-miner-base:v1`
and sets `VHFT_MINER_ENTRYPOINT=synth_ultra.model`. Synth's serving loop imports
that module and calls `predict_percentiles`. numpy is already in the base image.

### Local run

```powershell
python -m pip install -r requirements-dev.txt
python -m synth_ultra.example
python -m synth_ultra.validate --strict
python -m pytest
```

`python -m synth_ultra.example` writes `examples/sample_payload.json` and prints
the `(100,)` percentile output. Payload candles/trades come from the CSVs when
present; pass `current_time_ms` to choose the forecast anchor.

```powershell
python binance_fetch.py
python binance_fetch.py --only spot-trades
```

### Build the submission image

Target `linux/amd64` (the evaluation platform). Run from the repo root:

```powershell
docker build --platform linux/amd64 -t synth-ultra:v1 .
```

Confirm the entrypoint is importable:

```powershell
docker run --rm --platform linux/amd64 --network=none --entrypoint python synth-ultra:v1 -c "from synth_ultra.model import predict_percentiles; print(predict_percentiles)"
```

### Submit to Synth

Push the image to the private registry repository Synth gave you at onboarding,
then submit the **digest** with [`client/submit.py`](client/submit.py). Replace
`<your-registry-repo>`, wallet, and hotkey with your onboarding values.

```powershell
docker tag synth-ultra:v1 <your-registry-repo>/miner:v1
docker push <your-registry-repo>/miner:v1
```

Copy the `sha256:...` digest printed by `docker push`, then:

```powershell
uv run --no-project --with "bittensor>=11,<12" python client/submit.py submit `
  --wallet my_coldkey --hotkey my_hotkey `
  --image-uri <your-registry-repo>/miner `
  --image-digest sha256:<digest-from-docker-push> --version 1
```

Check status:

```powershell
uv run --no-project --with "bittensor>=11,<12" python client/submit.py status `
  --wallet my_coldkey --hotkey my_hotkey
```

Limits: one submission per hotkey every 4 hours; each version must be strictly
newer than the last. Full flow: [`FAQ.md`](FAQ.md).
