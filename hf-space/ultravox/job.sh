#!/usr/bin/env bash
# Scores an Ultravox checkpoint on the fixie-ai/language_detection-audio clips as a Hugging Face job.
# Expects the bucket mounted at /data with ultravox/{wav,manifest.json,ultravox_lid.py} (see README
# "Against Ultravox"); env: MODEL (hub id), NAME (result file name), EXTRA (extra ultravox_lid.py args).
set -e
pip install -q 'transformers==4.56.2' accelerate peft soundfile librosa hf_transfer 2>&1 | tail -2
export HF_HUB_ENABLE_HF_TRANSFER=1
nvidia-smi -L; df -h / | tail -1; free -g | head -2
mkdir -p /work && cp -r /data/ultravox/wav /data/ultravox/manifest.json /data/ultravox/ultravox_lid.py /work/ && cd /work
python ultravox_lid.py "$MODEL" /work/wav /work/manifest.json /data/ultravox/results/$NAME.json $EXTRA
