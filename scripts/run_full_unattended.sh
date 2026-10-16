#!/usr/bin/env bash
# Full run, hands-off: setup -> restore any earlier partial run -> persona smoke test on the pilot ->
# all phases (0-3) -> persona axes (4)
# -> backup -> STOP THE POD. Safe to rerun: finished work is restored from the backup and skipped.
#
#   printf '%s' 'hf_...' > /root/.hf_token && chmod 600 /root/.hf_token     # once per pod
#   tmux new -d -s full 'bash scripts/run_full_unattended.sh'                 # start, detached
#   tail -f /root/full_run.log                                                # watch (Ctrl-c only stops watching)
#
# Guards so a hang can't burn money:
#   - idle watchdog: if every GPU is <5% busy AND the CPU is idle for IDLE_MIN minutes (default 40), stop the pod
#   - hard cap: stop after MAX_HOURS (default 30; expected total ~20-24 h on 2 x H100 NVL)
# Set NO_STOP=1 to keep the pod running at the end (e.g. to inspect it). PIPELINE=<script> runs another
# pipeline (default scripts/run_all.sh; scripts/cm_unattended.sh uses scripts/cm_pipeline.sh).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export CONTAINER_STORAGE=1 PERSIST_DIR="${PERSIST_DIR:-/workspace}"
export EM_ROOT="${FULL_EM_ROOT:-/root/em}" HF_HOME="${CONTAINER_HF_HOME:-/root/hf}"   # ignores any EM_ROOT left in the shell
export SKIP_HUMAN=1 FORCE=1          # don't pause for human labelling or failed gates (both are checked afterwards)
MAX_HOURS="${MAX_HOURS:-30}"; IDLE_MIN="${IDLE_MIN:-40}"
LOG="${FULL_LOG:-/root/full_run.log}"
exec > >(tee -a "$LOG") 2>&1

say() { echo "[$(date '+%F %T')] $*"; }

if [[ -s /root/.hf_token ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < /root/.hf_token)"; export HF_TOKEN
else
  say "ERROR: /root/.hf_token is missing or empty"; exit 1
fi

stop_pod() {
  say "final backup"
  python -m emsp.backup || say "WARNING: final backup failed"
  if [[ "${NO_STOP:-0}" == 1 ]]; then say "NO_STOP=1: leaving the pod running"; return; fi
  if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null; then
    say "stopping pod $RUNPOD_POD_ID ($1)"
    runpodctl stop pod "$RUNPOD_POD_ID"
  else
    say "WARNING: cannot stop the pod (RUNPOD_POD_ID or runpodctl missing) - stop it in the RunPod console"
  fi
}

# ---------------------------------------------------------------- hard cap (armed now)
( sleep "$((MAX_HOURS * 3600))"; say "HARD CAP of $MAX_HOURS h reached"; stop_pod "time cap" ) &
CAP_PID=$!

say "setup (pip + model downloads; ~30-60 min on a fresh pod)"
if ! bash setup.sh; then say "setup FAILED"; stop_pod "setup failed"; exit 1; fi

mkdir -p "$EM_ROOT"
BK="$PERSIST_DIR/em_backup/$(basename "$EM_ROOT")"
# Resume only a run made by this (matched-prompt) version of the code: it writes data/benign_heldout.jsonl.
if [[ -f "$BK/data/benign_heldout.jsonl" ]]; then
  say "restoring earlier partial full run from $BK"; python -m emsp.backup --restore
elif [[ -d "$BK" ]]; then
  say "NOTE: $BK is from an older code version - moving it aside and starting clean"
  mv "$BK" "$BK.old_$(date +%Y%m%d_%H%M)"
fi

# ---------------------------------------------------------------- idle watchdog (armed after setup)
(
  idle=0
  while sleep 60; do
    busy="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '$1 >= 5' | wc -l)"
    cpu="$(awk '{print ($2 >= 1.0)}' /proc/loadavg)"     # CPU-only steps (merge, analysis) keep load >= 1
    if (( busy == 0 && cpu == 0 )); then idle=$((idle + 1)); else idle=0; fi
    if (( idle >= IDLE_MIN )); then say "GPUs idle for $IDLE_MIN min - assuming a hang"; stop_pod "idle"; exit 0; fi
  done
) &
WD_PID=$!

# ---------------------------------------------------------------- persona smoke test on the pilot (~30 min)
# Exercises the GPU parts of phase 4 early, on the finished pilot, so a bug shows up now rather than after
# ~20 h - and gives pilot-scale persona results. A failure is logged and the full run goes ahead anyway.
PB="$PERSIST_DIR/em_backup/em_pilot"
if [[ "${PILOT_PERSONA:-1}" == 1 && -d "$PB/judged" && ! -f "$PB/results/persona_summary.md" ]]; then
  say "persona smoke test on the pilot ($PB)"
  if EM_ROOT=/root/em_pilot python -m emsp.backup --restore \
     && EM_ROOT=/root/em_pilot python -m emsp.merge --adapter /root/em_pilot/runs/W/s0/final --out /root/em_pilot/runs/W_merged \
     && EM_ROOT=/root/em_pilot bash scripts/phase4_persona.sh; then
    say "pilot persona smoke test OK -> /root/em_pilot/results/persona_summary.md (backed up to $PB/results)"
  else
    say "WARNING: pilot persona smoke test FAILED (see /root/em_pilot/logs/persona_*.log); continuing with the full run"
  fi
  rm -rf /root/em_pilot/runs/W_merged     # free ~30 GB of container disk
fi

PIPELINE="${PIPELINE:-scripts/run_all.sh}"
say "$PIPELINE in $EM_ROOT"
bash "$PIPELINE"; rc=$?
say "$PIPELINE exited with code $rc"

# kill the guards first (so the cap can't fire mid-shutdown), then their sleeping children (an orphaned sleep
# would keep the log pipe open)
kids="$(pgrep -P "$WD_PID" 2>/dev/null) $(pgrep -P "$CAP_PID" 2>/dev/null)"
kill "$WD_PID" "$CAP_PID" 2>/dev/null
# shellcheck disable=SC2086
[[ -n "${kids// /}" ]] && kill $kids 2>/dev/null
say "results in $EM_ROOT/results (also in $PERSIST_DIR/em_backup/$(basename "$EM_ROOT")/results)"
stop_pod "finished (rc=$rc)"
