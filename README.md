# git URL
https://github.com/bironlozano15-maker/synth-ultra.git

# How to submit model to synth
1. docker build --platform linux/amd64 -t synth-ultra:v1 .

2. docker run --rm --platform linux/amd64 --network=none --entrypoint python synth-ultra:v1 -c "from synth_ultra.model import predict_percentiles; print(predict_percentiles)"

3. cat vhft-daniel-bruno-push-key.json | docker login -u _json_key --password-stdin https://asia-northeast1-docker.pkg.dev

4. docker tag synth-ultra:v1 asia-northeast1-docker.pkg.dev/synth-vhft/vhft-daniel-bruno/miner:v1

5. docker push asia-northeast1-docker.pkg.dev/synth-vhft/vhft-daniel-bruno/miner:v1
===== copy digest named "sha256:***" =====

6. uv run --no-project --with "bittensor>=11,<12" python client/submit.py submit --wallet bt --hotkey live16 --image-uri asia-northeast1-docker.pkg.dev/synth-vhft/vhft-daniel-bruno/miner --image-digest <digest_id:e.g. sha256:bd26f1a0dc57f86e56cac7b7507c19afa5799dd680eaeafd371ba3463f49b1af> --version 1

7. uv run --no-project --with "bittensor>=11,<12" python client/submit.py status --wallet bt --hotkey live16

IMPORTANT: whenever update model, update digest and version number