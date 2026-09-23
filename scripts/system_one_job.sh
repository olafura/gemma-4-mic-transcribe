#!/usr/bin/env bash
set -euo pipefail

# Runs one heavy System One job (cache, train, eval, regression gate) so that a
# single 12B process at a time can touch the GPU and an overrun kills the job
# instead of the machine. See docs/system-one-expert-plan.md, section 7.
#
#   scripts/system_one_job.sh JOB_NAME COMMAND [ARG...]
#
# Every limit is overridable through the environment:
#
#   SYSTEM_ONE_LOCKFILE          lockfile guarding the single heavy slot
#   SYSTEM_ONE_MIN_MEM_GB        preflight MemAvailable floor          (14)
#   SYSTEM_ONE_MIN_VRAM_GB       preflight free VRAM floor             (24)
#   SYSTEM_ONE_MIN_DISK_GB       preflight free disk floor             (60)
#   SYSTEM_ONE_DISK_PATH         filesystem checked for free disk      (/home)
#   SYSTEM_ONE_MEMORY_MAX        scope MemoryMax                       (20G)
#   SYSTEM_ONE_MEMORY_SWAP_MAX   scope MemorySwapMax                   (2G)
#   SYSTEM_ONE_CPU_QUOTA         scope CPUQuota                        (800%)
#   SYSTEM_ONE_WATCHDOG_MEM_GB   watchdog MemAvailable floor           (6)
#   SYSTEM_ONE_WATCHDOG_SECONDS  seconds under the floor before a stop (30)
#   SYSTEM_ONE_WATCHDOG_INTERVAL watchdog poll interval in seconds     (5)
#   SYSTEM_ONE_SKIP_VRAM_CHECK   set to 1 when no ROCm GPU is present
#   XLA_FLAGS                    overrides the exported ROCm flags
#
# Never match the job with `pgrep -f` / `pkill -f`: this wrapper's own command
# line contains the job's command line, so a pattern kill takes out the wrapper
# (and any sibling shell) too. The job is stopped by its systemd unit name.

LOCKFILE="${SYSTEM_ONE_LOCKFILE:-${TMPDIR:-/tmp}/gemma4-system-one.lock}"
MIN_MEM_GB="${SYSTEM_ONE_MIN_MEM_GB:-14}"
MIN_VRAM_GB="${SYSTEM_ONE_MIN_VRAM_GB:-24}"
MIN_DISK_GB="${SYSTEM_ONE_MIN_DISK_GB:-60}"
DISK_PATH="${SYSTEM_ONE_DISK_PATH:-/home}"
MEMORY_MAX="${SYSTEM_ONE_MEMORY_MAX:-20G}"
MEMORY_SWAP_MAX="${SYSTEM_ONE_MEMORY_SWAP_MAX:-2G}"
CPU_QUOTA="${SYSTEM_ONE_CPU_QUOTA:-800%}"
WATCHDOG_MEM_GB="${SYSTEM_ONE_WATCHDOG_MEM_GB:-6}"
WATCHDOG_SECONDS="${SYSTEM_ONE_WATCHDOG_SECONDS:-30}"
WATCHDOG_INTERVAL="${SYSTEM_ONE_WATCHDOG_INTERVAL:-5}"

# ROCm on gfx1151 needs these off before the client starts; exporting them here
# keeps every heavy job on the same flags.
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer= --xla_gpu_enable_triton_gemm=false}"

die() {
  echo "system_one_job: $*" >&2
  exit 1
}

usage() {
  echo "usage: scripts/system_one_job.sh JOB_NAME COMMAND [ARG...]" >&2
  exit 2
}

[ "$#" -ge 2 ] || usage

JOB_NAME="$1"
shift

case "$JOB_NAME" in
  *[!A-Za-z0-9_-]* | "") die "job name must be non-empty [A-Za-z0-9_-], got: $JOB_NAME" ;;
esac

UNIT="system-one-${JOB_NAME}-$$.scope"

# --- preflight ---------------------------------------------------------------

mem_available_kb() {
  awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo
}

# rocm-smi reports total and used; nothing reports free, and on this box about
# 22 GB shows as used with no KFD process attached, so free = total - used.
vram_bytes() {
  # $1 is the rocm-smi JSON, $2 the key fragment; sums the value over all cards.
  printf '%s' "$1" | tr ',' '\n' | grep -F "$2" |
    sed -n 's/.*: *"\{0,1\}\([0-9][0-9]*\)"\{0,1\}.*/\1/p' |
    awk '{ sum += $1 } END { print sum + 0 }'
}

vram_free_gb() {
  local json total used
  json="$(rocm-smi --showmeminfo vram --json 2>/dev/null || true)"
  [ -n "$json" ] || return 1

  total="$(vram_bytes "$json" 'VRAM Total Memory')"
  used="$(vram_bytes "$json" 'VRAM Total Used Memory')"

  [ "${total:-0}" -gt 0 ] || return 1
  awk -v t="$total" -v u="$used" 'BEGIN { printf "%.2f", (t - u) / 1073741824 }'
}

at_least() {
  awk -v have="$1" -v want="$2" 'BEGIN { exit !(have + 0 >= want + 0) }'
}

mem_gb="$(awk -v kb="$(mem_available_kb)" 'BEGIN { printf "%.2f", kb / 1048576 }')"
at_least "$mem_gb" "$MIN_MEM_GB" ||
  die "preflight: MemAvailable ${mem_gb} GB is under the ${MIN_MEM_GB} GB floor"

if [ "${SYSTEM_ONE_SKIP_VRAM_CHECK:-0}" = "1" ]; then
  vram_gb="skipped"
else
  vram_gb="$(vram_free_gb)" ||
    die "preflight: could not read VRAM from rocm-smi (set SYSTEM_ONE_SKIP_VRAM_CHECK=1 to bypass)"
  at_least "$vram_gb" "$MIN_VRAM_GB" ||
    die "preflight: free VRAM ${vram_gb} GB is under the ${MIN_VRAM_GB} GB floor"
fi

disk_gb="$(df -PBK "$DISK_PATH" | awk 'NR == 2 { printf "%.2f", $4 / 1048576 }')"
at_least "$disk_gb" "$MIN_DISK_GB" ||
  die "preflight: free disk on ${DISK_PATH} is ${disk_gb} GB, under the ${MIN_DISK_GB} GB floor"

# --- single heavy slot -------------------------------------------------------

exec 9>"$LOCKFILE"
flock -n 9 || die "another heavy job holds $LOCKFILE; wait for it to finish (two 12B GPU processes have segfaulted this box)"

echo "system_one_job: job=${JOB_NAME} unit=${UNIT} mem=${mem_gb}GB vram=${vram_gb}GB disk=${disk_gb}GB(${DISK_PATH})" >&2
echo "system_one_job: limits MemoryMax=${MEMORY_MAX} MemorySwapMax=${MEMORY_SWAP_MAX} CPUQuota=${CPU_QUOTA}" >&2

# --- watchdog ----------------------------------------------------------------

watchdog_floor_kb="$(awk -v gb="$WATCHDOG_MEM_GB" 'BEGIN { printf "%d", gb * 1048576 }')"

watchdog() {
  local waited=0 under=0 avail

  # Wait for the scope to appear before monitoring it.
  while ! systemctl --user --quiet is-active "$UNIT" 2>/dev/null; do
    [ "$waited" -lt 60 ] || return 0
    sleep 1
    waited=$((waited + 1))
  done

  while systemctl --user --quiet is-active "$UNIT" 2>/dev/null; do
    avail="$(mem_available_kb)"

    if [ "${avail:-0}" -lt "$watchdog_floor_kb" ]; then
      under=$((under + WATCHDOG_INTERVAL))

      if [ "$under" -ge "$WATCHDOG_SECONDS" ]; then
        echo "system_one_job: watchdog: MemAvailable under ${WATCHDOG_MEM_GB} GB for ${WATCHDOG_SECONDS}s, stopping ${UNIT}" >&2
        systemctl --user stop "$UNIT" >/dev/null 2>&1 || true
        return 0
      fi
    else
      under=0
    fi

    sleep "$WATCHDOG_INTERVAL"
  done
}

# Neither the watchdog nor the job gets fd 9: a leftover child (the
# watchdog's sleep, a daemon the job spawns) would hold the lock after this
# script exits and turn away the next job.
watchdog 9>&- &
watchdog_pid=$!

cleanup() {
  kill "$watchdog_pid" 2>/dev/null || true
  wait "$watchdog_pid" 2>/dev/null || true
}
trap cleanup EXIT

# --- run ---------------------------------------------------------------------

status=0
systemd-run --user --scope \
  --unit="$UNIT" \
  --collect \
  -q \
  --setenv=XLA_FLAGS \
  -p MemoryMax="$MEMORY_MAX" \
  -p MemorySwapMax="$MEMORY_SWAP_MAX" \
  -p CPUQuota="$CPU_QUOTA" \
  -- nice -n 19 ionice -c3 "$@" 9>&- || status=$?

cleanup
trap - EXIT

echo "system_one_job: job=${JOB_NAME} exited with status ${status}" >&2
exit "$status"
