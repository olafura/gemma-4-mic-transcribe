#!/usr/bin/env bash
# Validates a detector on an Nvidia GPU with `hf jobs run`, from the public
# Elixir image: the CUDA build of the language-ID slice (MIX_TARGET=
# language_id_cuda, copied out of the gemma-language-id:cuda image into the
# bucket as runtime-cuda/app) plus the CUDA libraries XLA loads at run time,
# installed here from the nvidia-*-cu12 wheels. NVSHMEM stays on 3.3 (later
# releases rename the transport plugins from .so.3 to .so.6 and the NIF does
# not load) and NVRTC on 12.9 (libnvrtc-builtins.so.12.9); /usr/local/cuda
# points at the nvcc wheel so XLA finds ptxas and libdevice.
#
#   hf buckets cp hf-space/job_cuda.sh hf://buckets/olafura/gemma_language_detection/jobs/hf_job_cuda.sh
#   hf jobs run --flavor t4-small --timeout 3h -d -e ARTIFACT=detector-sent49-depth5-1s \
#     -v hf://buckets/olafura/gemma_language_detection:/data \
#     hexpm/elixir:1.20.2-erlang-29.0.3-debian-bookworm-20260713-slim bash /data/jobs/hf_job_cuda.sh
set -e
apt-get update -qq >/dev/null && apt-get install -y -qq ffmpeg python3-pip >/dev/null
pip install -q --break-system-packages nvidia-cublas-cu12 'nvidia-cuda-nvrtc-cu12==12.9.*' nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 'nvidia-nvshmem-cu12==3.3.*' 'nvidia-cuda-nvcc-cu12==12.9.*'
NV=$(pip show nvidia-cublas-cu12 2>/dev/null | sed -n 's/^Location: //p')/nvidia
export LD_LIBRARY_PATH=$(ls -d $NV/*/lib | tr '\n' ':')$LD_LIBRARY_PATH
ln -sfn $NV/cuda_nvcc /usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
nvidia-smi -L || true
cp -r /data/runtime-cuda/app /app && chmod +x /app/language_id && cd /app
./language_id validate --backend exla:cuda --artifact /data/artifacts/$ARTIFACT --per-language 30 --output /data/validation/$ARTIFACT-cuda.json
