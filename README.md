# Synth Ultra

Synth AI model for price simulations (Henry Bauer).

Your Synth registry repo (from onboarding):
`asia-northeast1-docker.pkg.dev/synth-vhft/vhft-henry-bauer/miner`

Your registry push key (keep this file local, never commit it):
`vhft-henry-bauer-push-key.json`

Your Bittensor identity (Synth subnet UID 46):

| | |
| --- | --- |
| Wallet name | `bt` |
| Coldkey | `5HaxR8qCeBmxeimWzQasnBq8ohkCxuRYT6YonXj59pm1TxYz` |
| Hotkey | `5EWMaCP7GfRmcvi1Gzod14r6dZw1nsNZAN7hmDMZpNYoDegi` |
| UID | 46 |

`client/submit.py --wallet bt --hotkey live2` reads the hotkey **file name** `live2`, not the SS58 address. After submit, the printed `hotkey :` line must be `5EWMaCP7GfRmcvi1Gzod14r6dZw1nsNZAN7hmDMZpNYoDegi`.


# How to submit model to synth

**Preparation:** Docker installed, plus a Python/bittensor environment (`uv` is enough for the client). Wallet `bt` with the hotkey above must exist on the machine that runs step 6 (`%USERPROFILE%\.bittensor\wallets\bt\hotkeys\` on Windows). This machine does not have those files yet — copy or recreate the wallet before submitting.

Use `cat` on Linux (your VPS). Use `Get-Content` only on Windows PowerShell.

1. Build for the evaluation platform (`linux/amd64`):

```
docker build --platform linux/amd64 -t synth-ultra:v1 .
```

2. Smoke-test that `predict_percentiles` imports inside the image:

```
docker run --rm --platform linux/amd64 --network=none --entrypoint python synth-ultra:v1 -c "from synth_ultra.model import predict_percentiles; print(predict_percentiles)"
```

3. Log in to Synth's Artifact Registry with **your** push key.

Linux (your VPS):

```
cat vhft-henry-bauer-push-key.json | docker login -u _json_key --password-stdin https://asia-northeast1-docker.pkg.dev
```

Windows PowerShell only:

```
Get-Content vhft-henry-bauer-push-key.json -Raw | docker login -u _json_key --password-stdin https://asia-northeast1-docker.pkg.dev
```

4. Tag the image for **your** registry repo:

```
docker tag synth-ultra:v1 asia-northeast1-docker.pkg.dev/synth-vhft/vhft-henry-bauer/miner:v1
```

5. Push. Copy the `sha256:` + 64 hex characters from the last lines of the output (do not copy any `<` `>` brackets — bash treats those as redirects):

```
docker push asia-northeast1-docker.pkg.dev/synth-vhft/vhft-henry-bauer/miner:v1
```

If you already pushed, print the digest again:

```
docker inspect --format='{{index .RepoDigests 0}}' asia-northeast1-docker.pkg.dev/synth-vhft/vhft-henry-bauer/miner:v1
```

That prints `.../miner@sha256:abc123...`. Use only the `sha256:abc123...` part.

6. Sign and submit. On the VPS you already have `.venv` — install bittensor there and skip `uv`:

```
pip install "bittensor>=11,<12"
python client/submit.py submit --wallet bt --hotkey live2 --image-uri asia-northeast1-docker.pkg.dev/synth-vhft/vhft-henry-bauer/miner --image-digest sha256:006cd08546fcf1c492e9f2d1660546a661b84b2932af4d7f87adfa086ed91a4f --version 1
```

If you prefer `uv` later, install it with `curl -LsSf https://astral.sh/uv/install.sh | sh` (do not use snap). Confirm the client prints `hotkey : 5EWMaCP7GfRmcvi1Gzod14r6dZw1nsNZAN7hmDMZpNYoDegi`. If it prints a different address, you loaded the wrong hotkey file.

7. Check status:

```
python client/submit.py status --wallet bt --hotkey live2
```

**IMPORTANT:** every model update needs a new digest and a strictly higher `--version`. Synth allows one submission per hotkey every 4 hours. After approval, a registered hotkey typically goes live within about 2 minutes.


## Version history

All times are UTC+9. Submit version is per-hotkey — Henry Bauer's first Synth submission is `--version 1`.

|Version| Submit Version   |     Started      |                            Notes                              |
|  ---  |       ---        |      ---         |                             ---                               |
| 1.0   | 1                | 2026/08/29 04:02 | first submission under vhft-henry-bauer (pending review)      |
| 2.0   | —                | 2026/08/25 09:35 | Rebuild with Synth base image V2 (no logic update)            |
| 2.1   | —                | 2026/08/28 00:15 | Adjust model's parameters                                     |
