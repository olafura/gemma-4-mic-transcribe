#!/usr/bin/env bash
# Times the packed-int4 decode kernel alone (q4_bench.exs) on an Nvidia GPU,
# with the same CUDA setup as job.sh, whose setup half it runs.
#
#   hf jobs run --flavor a100-large --timeout 30m -d -e FLAVOR=a100-bench \
#     -v hf://buckets/olafura/gemma4_system_one:/data \
#     hexpm/elixir:1.20.2-erlang-29.0.3-debian-bookworm-20260713-slim bash /data/jobs/bench.sh
set -eu
source <(sed -n '1,/^export GEMMA_EXLA_MEMORY_FRACTION/p' /data/jobs/job.sh)
/app/system_one/bin/system_one eval 'Code.eval_file("/data/jobs/q4_bench.exs")' 2>&1 |
  grep -v '\[info\]' | tee "$OUT/bench.log"
