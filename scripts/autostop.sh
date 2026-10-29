#!/usr/bin/env bash
# Stop the pod once no experiment is running and both GPUs are idle (3 checks in a row, 2 min apart).
#   tmux new -s stop 'bash scripts/autostop.sh'
#   ALLOW_TERMINATE=1 bash scripts/autostop.sh   # if "stop" is refused, terminate instead (the /workspace volume is kept)
cd "$(dirname "$0")/.." || exit 1
export EM_ROOT="${PP_EM_ROOT:-/root/cm_pilot}" PERSIST_DIR="${PERSIST_DIR:-/workspace}" CONTAINER_STORAGE=1
INTERVAL="${INTERVAL:-120}"; NEED="${NEED:-3}"; MAXMEM="${MAXMEM:-2000}"; idle=0
say() { echo "[$(date '+%F %T')] autostop: $*" | tee -a "$EM_ROOT/logs/autostop.log"; }
say "watching (every ${INTERVAL}s, stop after $NEED idle checks); pod id: ${RUNPOD_POD_ID:-unknown}"
while :; do
  busy_proc=$(pgrep -f "v2_pilot.sh|unlearn.sh|forgetting.sh|forgetting_rl_cov.sh|followups.sh|followups_launch.sh|python -m emsp" | wc -l)
  busy_gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk -v m="$MAXMEM" '$1>m' | wc -l)
  if (( busy_proc == 0 && busy_gpu == 0 )); then idle=$((idle+1)); say "idle check $idle/$NEED"; else
    (( idle > 0 )) && say "busy again ($busy_proc processes, $busy_gpu GPUs in use)"; idle=0; fi
  if (( idle >= NEED )); then
    say "everything finished - final backup to $PERSIST_DIR/em_backup"
    PYTHONPATH="$PWD" python -m emsp.backup >> "$EM_ROOT/logs/backup.log" 2>&1 && say "backup OK" || say "WARNING: backup failed"
    if [[ -z "${RUNPOD_POD_ID:-}" ]] || ! command -v runpodctl >/dev/null; then
      say "runpodctl or RUNPOD_POD_ID missing - stop the pod by hand in the RunPod console"; exit 1; fi
    say "stopping pod $RUNPOD_POD_ID"
    runpodctl stop pod "$RUNPOD_POD_ID" && exit 0
    if [[ "${ALLOW_TERMINATE:-0}" == 1 ]]; then say "stop refused - terminating (volume is kept)"; runpodctl remove pod "$RUNPOD_POD_ID"; fi
    exit 0
  fi
  sleep "$INTERVAL"
done
