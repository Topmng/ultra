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

`client/submit.py --wallet bt --hotkey <name>` reads the **hotkey file name** under the wallet (for example `default`), not the SS58 address. After submit, the printed `hotkey :` line must be `5EWMaCP7GfRmcvi1Gzod14r6dZw1nsNZAN7hmDMZpNYoDegi`.


# How to submit model to synth

**Preparation:** Docker installed, plus a Python/bittensor environment (`uv` is enough for the client). Wallet `bt` with the hotkey above must exist on the machine that runs step 6 (`%USERPROFILE%\.bittensor\wallets\bt\hotkeys\` on Windows). This machine does not have those files yet — copy or recreate the wallet before submitting.

On Windows PowerShell, `cat` is not available — use `Get-Content` for the login step.

1. Build for the evaluation platform (`linux/amd64`):

```
docker build --platform linux/amd64 -t synth-ultra:v1 .
```

2. Smoke-test that `predict_percentiles` imports inside the image:

```
docker run --rm --platform linux/amd64 --network=none --entrypoint python synth-ultra:v1 -c "from synth_ultra.model import predict_percentiles; print(predict_percentiles)"
```

3. Log in to Synth's Artifact Registry with **your** push key.

PowerShell:

```
Get-Content vhft-henry-bauer-push-key.json -Raw | docker login -u _json_key --password-stdin https://asia-northeast1-docker.pkg.dev
```

Linux / Git Bash:

```
cat vhft-henry-bauer-push-key.json | docker login -u _json_key --password-stdin https://asia-northeast1-docker.pkg.dev
```

4. Tag the image for **your** registry repo:

```
docker tag synth-ultra:v1 asia-northeast1-docker.pkg.dev/synth-vhft/vhft-henry-bauer/miner:v1
```

5. Push, then copy the digest named `sha256:...` from the push output:

```
docker push asia-northeast1-docker.pkg.dev/synth-vhft/vhft-henry-bauer/miner:v1
```

6. Sign and submit. Replace `<hotkey-file-name>` (the file in `wallets\bt\hotkeys\`, often `default`) and the digest:

```
uv run --no-project --with "bittensor>=11,<12" python client/submit.py submit --wallet bt --hotkey <hotkey-file-name> --image-uri asia-northeast1-docker.pkg.dev/synth-vhft/vhft-henry-bauer/miner --image-digest sha256:<paste-from-docker-push> --version 1
```

Confirm the client prints `hotkey : 5EWMaCP7GfRmcvi1Gzod14r6dZw1nsNZAN7hmDMZpNYoDegi`. If it prints a different address, you loaded the wrong hotkey file.

7. Check status:

```
uv run --no-project --with "bittensor>=11,<12" python client/submit.py status --wallet bt --hotkey <hotkey-file-name>
```

**IMPORTANT:** every model update needs a new digest and a strictly higher `--version`. Synth allows one submission per hotkey every 4 hours. After approval, a registered hotkey typically goes live within about 2 minutes.


## Version history

All times are UTC+9. Submit version is per-hotkey — Henry Bauer's first Synth submission is `--version 1`.

|Version| Submit Version   |     Started      |                            Notes                              |
|  ---  |       ---        |      ---         |                             ---                               |
| 1.0   | 1                |                  | first submission under vhft-henry-bauer                       |
| 2.0   | —                | 2026/08/25 09:35 | Rebuild with Synth base image V2 (no logic update)            |
| 2.1   | —                | 2026/08/28 00:15 | Adjust model's parameters                                     |
