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

# bpftrace cannot raise its own memlock limit without cap_sys_resource. Since
# Linux 5.11 BPF memory is charged to the cgroup instead, so the warning it
# prints is harmless; raise the soft limit anyway where the hard limit allows.
ulimit -l "$(ulimit -Hl)" 2>/dev/null || true

echo "tracing HIP calls in pid $PID (histograms print every 5s while the GPU is busy)" >&2
echo "if nothing appears, the target is idle: start a transcription first" >&2

# bpftrace block-buffers stdout when it is a pipe, so piping into tee hides the
# periodic histograms until 4KB accumulates. stdbuf forces line buffering.
exec stdbuf -oL bpftrace -p "$PID" -e "
uprobe:${HIP_LIB}:hipLaunchKernel { @launch[tid] = nsecs; @launch_count++; }
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
  time(\"%H:%M:%S \");
  printf(\"launches=%d\n\", @launch_count);
  print(@kernel_launch_us);
  print(@memcpy_us);
  print(@sync_wait_us);
}
"
