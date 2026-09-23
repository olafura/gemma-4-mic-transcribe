#!/usr/bin/env bash
# Retrain the router's ask-back probe with requests of your own added, for
# instance from a domain it misses, and compare it with the shipped probe.
#
#   scripts/system_one/retrain_ask_probe.sh OUT_DIR TRAIN.jsonl [...] [-- EVAL.jsonl ...]
#
# TRAIN and EVAL rows are the `--extra` format of build_router_probe_set.py:
# an `id`, a System One item (`state`, `question`, `options`) or a bare
# `prompt`, and `decidable` (or `label`); a row with `audio` is spoken. Keep
# EVAL requests out of TRAIN.
#
# 1. Render the rows as `route` serves them and check that with the router.
# 2. Cache only the new rows (the GPU step, through scripts/system_one_job.sh);
#    the base rows are reused: 3,089 written from data/system-one/cache-router-probe
#    and 600 spoken from data/system-one/cache-router-probe-spoken. The held-out
#    caches (400 written, 200 spoken rows) are built once, on first use; the
#    spoken ones need the WAVs, which tts_questions.py makes from
#    data/system-one/spoken-{train,heldout}/items.jsonl.
# 3. Export OUT_DIR/ask-probe from base + new rows.
# 4. Score the shipped probe and the new one side by side on the written
#    and spoken held-out sets, plus EVAL if given.
#
# To serve the new probe: mix gemma.system_one route --ask-probe OUT_DIR/ask-probe ...
# Environment: C (inverse L2 strength, default 0.003), THRESHOLD (default 0.8),
# AUDIO_SECONDS (the audio bucket of your spoken rows, default 8; the base
# spoken rows are cached at 4, the length of their clips).
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ $# -lt 2 ]; then
  sed -n 2,26p "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi

out=$1
shift
train=()
eval_files=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do train+=("$1"); shift; done
if [ $# -gt 0 ]; then shift; eval_files=("$@"); fi

export XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer= --xla_gpu_enable_triton_gemm=false'
MISE=(mise x erlang@29.0.3 elixir@1.20.2-otp-29 --)
PY=(uv run --quiet --with scikit-learn --with safetensors --with numpy --with pyarrow --with tokenizers python)
D=data/system-one
base_cache=$D/cache-router-probe
spoken_cache=$D/cache-router-probe-spoken
heldout_cache=$D/cache-router-probe-heldout
spoken_heldout_cache=$D/cache-router-probe-spoken-heldout

[ -d "$base_cache" ] || { echo "missing $base_cache: cache $D/router-probe-train.jsonl first (plan doc, Route 1)" >&2; exit 1; }
mkdir -p "$out"

cache() { # NAME INPUT OUTPUT [AUDIO_SECONDS]
  [ -d "$3" ] && { echo "reusing $3"; return; }
  scripts/system_one_job.sh "$1" "${MISE[@]}" mix gemma.system_one cache \
    --input "$2" --output "$3" --buckets 256,384,512 --last-prompt-token-only \
    --audio-seconds "${4:-${AUDIO_SECONDS:-8}}"
}

"${PY[@]}" scripts/system_one/build_router_probe_set.py --only-extra --extra "${train[@]}" --output "$out/train-extra.jsonl"
checked=("$out/train-extra.jsonl")
if [ ${#eval_files[@]} -gt 0 ]; then
  "${PY[@]}" scripts/system_one/build_router_probe_set.py --only-extra --extra "${eval_files[@]}" --output "$out/eval-extra.jsonl"
  checked+=("$out/eval-extra.jsonl")
fi
"${MISE[@]}" mix run --no-start scripts/system_one/check_router_render.exs "${checked[@]}"

cache probe-cache-spoken $D/router-probe-train-spoken.jsonl "$spoken_cache" 4
cache probe-cache-heldout $D/router-probe-heldout.jsonl "$heldout_cache"
cache probe-cache-spoken-heldout $D/router-probe-heldout-spoken.jsonl "$spoken_heldout_cache" 4
cache probe-cache-extra "$out/train-extra.jsonl" "$out/cache-train-extra"
eval_caches=("$heldout_cache" "$spoken_heldout_cache")
eval_labels=($D/router-probe-heldout.jsonl $D/router-probe-heldout-spoken.jsonl)
if [ ${#eval_files[@]} -gt 0 ]; then
  cache probe-cache-eval "$out/eval-extra.jsonl" "$out/cache-eval-extra"
  eval_caches+=("$out/cache-eval-extra")
  eval_labels+=("$out/eval-extra.jsonl")
fi

"${PY[@]}" scripts/system_one/export_ask_probe.py \
  --cache "$base_cache" "$spoken_cache" "$out/cache-train-extra" \
  --labels $D/router-probe-train.jsonl $D/router-probe-train-spoken.jsonl "$out/train-extra.jsonl" \
  --c "${C:-0.003}" --threshold "${THRESHOLD:-0.8}" --output "$out/ask-probe"

"${PY[@]}" scripts/system_one/eval_ask_probe.py \
  --probe artifacts/system-one/ask-probe "$out/ask-probe" \
  --cache "${eval_caches[@]}" --labels "${eval_labels[@]}" \
  --output "$out/eval.json"
