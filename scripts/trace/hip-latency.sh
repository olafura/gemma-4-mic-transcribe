#!/usr/bin/env bash
# HIP API latency histograms for a running transcription (beam.smp by default).
#
# bpftrace needs root, so run this via the mise task:
#   mise run trace:bpf            # attaches to the oldest beam.smp
#   mise run trace:bpf -- <pid>   # attaches to a specific pid
#
# The no-root alternative is BEAM call tracing: ./gemma_4_mic_transcribe --trace
#
# Histograms print every 5s:
#   @kernel_launch_us  hipLaunchKernel call latency (dispatch overhead)
#   @memcpy_us         hipMemcpy* call latency (host<->device traffic)
#   @sync_wait_us      hipStreamSynchronize latency (time blocked on the GPU)
set -euo pipefail

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIP_LIB="${HIP_LIB:-$ROCM_PATH/lib/libamdhip64.so}"
PID="${1:-$(pgrep -o beam.smp)}"

if [ ! -e "$HIP_LIB" ]; then
  echo "error: HIP runtime not found at $HIP_LIB (set ROCM_PATH or HIP_LIB)" >&2
  exit 1
fi

if [ -z "$PID" ]; then
  echo "error: no beam.smp process found; pass a pid explicitly" >&2
  exit 1
fi

exec bpftrace -p "$PID" -e "
uprobe:${HIP_LIB}:hipLaunchKernel { @launch[tid] = nsecs; }
uretprobe:${HIP_LIB}:hipLaunchKernel /@launch[tid]/ {
  @kernel_launch_us = hist((nsecs - @launch[tid]) / 1000);
  delete(@launch[tid]);
}
uprobe:${HIP_LIB}:hipMemcpy* { @copy[tid] = nsecs; }
uretprobe:${HIP_LIB}:hipMemcpy* /@copy[tid]/ {
  @memcpy_us = hist((nsecs - @copy[tid]) / 1000);
  delete(@copy[tid]);
}
uprobe:${HIP_LIB}:hipStreamSynchronize { @sync[tid] = nsecs; }
uretprobe:${HIP_LIB}:hipStreamSynchronize /@sync[tid]/ {
  @sync_wait_us = hist((nsecs - @sync[tid]) / 1000);
  delete(@sync[tid]);
}
interval:s:5 {
  time(\"%H:%M:%S\n\");
  print(@kernel_launch_us);
  print(@memcpy_us);
  print(@sync_wait_us);
}
"
