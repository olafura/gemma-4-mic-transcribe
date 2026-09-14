#!/usr/bin/env bash
# Scores a detector on the fixie-ai/language_detection-audio clips on an Nvidia GPU
# through EXLA, as a Hugging Face job: same runtime setup as hf-space/job_cuda.sh,
# then `language_id serve --backend exla:cuda` and gemma_lid_eval.py against it.
#
#   hf jobs run --flavor a100-large --timeout 1h -d -e ARTIFACT=detector-sent49-depth5-1s \
#     -v hf://buckets/olafura/gemma_language_detection:/data \
#     hexpm/elixir:1.20.2-erlang-29.0.3-debian-bookworm-20260713-slim bash /data/jobs/hf_job_gemma_cuda.sh
set -e
apt-get update -qq >/dev/null && apt-get install -y -qq ffmpeg python3-pip >/dev/null
pip install -q --break-system-packages numpy soundfile nvidia-cublas-cu12 'nvidia-cuda-nvrtc-cu12==12.9.*' nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 'nvidia-nvshmem-cu12==3.3.*' 'nvidia-cuda-nvcc-cu12==12.9.*'
NV=$(pip show nvidia-cublas-cu12 2>/dev/null | sed -n 's/^Location: //p')/nvidia
export LD_LIBRARY_PATH=$(ls -d $NV/*/lib | tr '\n' ':')$LD_LIBRARY_PATH
ln -sfn $NV/cuda_nvcc /usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
nvidia-smi -L || true
cp -r /data/runtime-cuda/app /app && chmod +x /app/language_id && cd /app
./language_id serve --backend exla:cuda --artifact /data/artifacts/$ARTIFACT --port 7861 &
python3 - <<'PY'
import time, urllib.request
for _ in range(120):
    try: print(urllib.request.urlopen("http://localhost:7861/health", timeout=2).read().decode()); break
    except Exception: time.sleep(2)
PY
C=en,de,fr,es,it,nl,pl,pt,cs,sk,sl,sv-SE,da,fi,et,lv,lt,hu,ro,el,bg
OUT=/data/ultravox/results/$ARTIFACT-a100
python3 /data/ultravox/gemma_lid_eval.py /data/ultravox/wav /data/ultravox/manifest.json $OUT-cand21.json --candidates $C
python3 /data/ultravox/gemma_lid_eval.py /data/ultravox/wav /data/ultravox/manifest.json $OUT-cand21win.json --candidates $C --windows
