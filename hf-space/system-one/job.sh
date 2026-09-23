#!/usr/bin/env bash
# Times the System One router (`route`) on an Nvidia GPU with `hf jobs run`.
# The private bucket mounted at /data holds the release built by the
# Dockerfile beside this script (runtime/system_one-cuda.tar.gz), the 12B
# artifacts, the ask-back probe and the request set; results land in
# /data/results/$FLAVOR. The CUDA libraries come from the nvidia-*-cu12
# wheels, as in hf-space/job_cuda.sh (NVSHMEM 3.3, NVRTC 12.9, nvcc wheel as
# /usr/local/cuda for ptxas and libdevice).
#
#   hf buckets cp hf-space/system-one/job.sh hf://buckets/olafura/gemma4_system_one/jobs/job.sh
#   hf jobs run --flavor a100-large --timeout 90m -d -e FLAVOR=a100-large -e WEIGHTS="packed bf16" \
#     -v hf://buckets/olafura/gemma4_system_one:/data \
#     hexpm/elixir:1.20.2-erlang-29.0.3-debian-bookworm-20260713-slim bash /data/jobs/job.sh
#
# WEIGHTS lists the weight sets to time: packed (W4A16, 13.4 GB, fits a 24 GB
# card) and bf16 (24.2 GB, needs 40 GB or more).
set -eu
: "${FLAVOR:?set FLAVOR to the hardware flavor}"
WEIGHTS=${WEIGHTS:-packed}
OUT=/data/results/$FLAVOR
mkdir -p "$OUT"
exec > >(tee -a "$OUT/job.log") 2>&1

apt-get update -qq >/dev/null && apt-get install -y -qq ffmpeg python3-pip >/dev/null
pip install -q --break-system-packages nvidia-cublas-cu12 'nvidia-cuda-nvrtc-cu12==12.9.*' nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 'nvidia-nvshmem-cu12==3.3.*' 'nvidia-cuda-nvcc-cu12==12.9.*'
NV=$(pip show nvidia-cublas-cu12 2>/dev/null | sed -n 's/^Location: //p')/nvidia
export LD_LIBRARY_PATH=$(ls -d $NV/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}
ln -sfn $NV/cuda_nvcc /usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
nvidia-smi | tee "$OUT/nvidia-smi.txt"

# The bucket mount drops exec bits and has thrown random read errors, so the
# release is unpacked from a tarball and the weights are copied to local disk,
# retried until the sizes match.
mkdir -p /app && tar -xzf "/data/runtime/${RUNTIME:-system_one-cuda.tar.gz}" -C /app
bytes() { find "$1" -type f -printf '%s\n' | awk '{ s += $1 } END { print s + 0 }'; }
fetch() {
  for try in 1 2 3 4 5; do
    rm -rf "/work/$1" && mkdir -p "/work/$(dirname "$1")" && cp -r "/data/$1" "/work/$1" &&
      [ "$(bytes "/data/$1")" = "$(bytes "/work/$1")" ] && return 0
    echo "copy of $1 failed (try $try)"
  done
  return 1
}
fetch artifacts/ask-probe
fetch requests
cd /work

export GEMMA_EXLA_MEMORY_FRACTION=${GEMMA_EXLA_MEMORY_FRACTION:-0.9}

route() { # NAME PREFIX TAIL INPUT [ARGS...]
  local name=$1 prefix=$2 tail=$3 input=$4
  shift 4
  echo "== $name $(date -u +%FT%TZ)"
  /app/system_one/bin/system_one eval \
    'Application.ensure_all_started(:gemma_4_mic_transcribe); System.halt(Gemma4MicTranscribe.SystemOneCLI.main(System.argv()))' \
    route --backend exla:cuda --prefix-artifact "$prefix" --tail-artifact "$tail" \
    --ask-probe artifacts/ask-probe --input "$input" --output "$OUT/$name.jsonl" "$@" \
    > "$OUT/$name.log" 2>&1 || echo "$name exited $?"
  grep -E '"event":"(route_ready|route_written)"|timed|elapsed' "$OUT/$name.log" | head -5 || true
}

for w in $WEIGHTS; do
  case $w in
    packed) prefix=artifacts/gemma4-12b-packed-prefix-0-44 tail=artifacts/gemma4-12b-packed-tail-45-47 ;;
    bf16) prefix=artifacts/gemma4-12b-prefix-0-44 tail=artifacts/gemma4-12b-tail-45-47 ;;
    *) echo "unknown weights $w"; continue ;;
  esac
  fetch "$prefix" && fetch "$tail" || { echo "no $w weights"; continue; }
  route "$w-text" "$prefix" "$tail" requests/text.jsonl
  route "$w-spoken" "$prefix" "$tail" requests/spoken.jsonl --audio-seconds 4
  rm -rf "/work/$prefix" "/work/$tail"
done
echo "== done $(date -u +%FT%TZ)"
