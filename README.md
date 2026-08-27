# Synth Ultra

Synth AI model for price simulations.


# git URL

https://github.com/bironlozano15-maker/synth-ultra.git

# How to submit model to synth

*PERAPATION*: activate virtual python/bittensor environment and install docker if it was not installed yet. -> apt install docker.io

1. docker build --platform linux/amd64 -t synth-ultra:v2 .

2. docker run --rm --platform linux/amd64 --network=none --entrypoint python synth-ultra:v2 -c "from synth_ultra.model import predict_percentiles; print(predict_percentiles)"

3. cat vhft-daniel-bruno-push-key.json | docker login -u _json_key --password-stdin https://asia-northeast1-docker.pkg.dev

4. docker tag synth-ultra:v2 asia-northeast1-docker.pkg.dev/synth-vhft/vhft-daniel-bruno/miner:v2

5. docker push asia-northeast1-docker.pkg.dev/synth-vhft/vhft-daniel-bruno/miner:v2
===== copy digest named "sha256:***" =====

6. uv run --no-project --with "bittensor>=11,<12" python client/submit.py submit --wallet bt --hotkey live16 --image-uri asia-northeast1-docker.pkg.dev/synth-vhft/vhft-daniel-bruno/miner --image-digest <digest_id:e.g. sha256:c873551e9ae110700037e1cc597264082f44d2cb02e8e3dc845b48d19e4abdfd> --version 2

7. uv run --no-project --with "bittensor>=11,<12" python client/submit.py status --wallet bt --hotkey live16

*IMPORTANT*: whenever update model, update digest and version number


## Version history

All times are UTC+9.

|Version| Submit Version   |     Started      |                            Notes                              |
|  ---  |       ---        |      ---         |                             ---                               |
| 1.0   | 1                | 2026/08/21 15:45 | initial version                                               |
| 2.0   | 2                | 2026/08/25 09:35 | Rebuild with Synth base image V2 (no logic update)            |
| 2.1   | 2                | 2026/08/28 00:15 | Adjust model's parameters                                     |