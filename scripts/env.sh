#!/usr/bin/env bash
# Shared environment + helpers. Sourced by every phase script.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

export PERSIST_DIR="${PERSIST_DIR:-/workspace}"
# CONTAINER_STORAGE=1: keep model weights and all experiment files on the container disk.
# Use this when the volume is a RunPod *network* volume: it was observed to silently truncate
# large files (model shards) after they had been written and verified. Small, irreplaceable
# outputs are copied back to $PERSIST_DIR/em_backup after every phase (see emsp/backup.py).
export CONTAINER_STORAGE="${CONTAINER_STORAGE:-0}"
if [[ "$CONTAINER_STORAGE" == 1 ]]; then
  export HF_HOME="${CONTAINER_HF_HOME:-/root/hf}"   # forced: the RunPod image presets HF_HOME on the volume
  export EM_ROOT="${EM_ROOT:-/root/em}"
fi
export EM_ROOT="${EM_ROOT:-$PERSIST_DIR/em}"
export HF_HOME="${HF_HOME:-$PERSIST_DIR/hf}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PERSIST_DIR/cache/pip}"
# Download whole files in order rather than in parallel pieces (safer on network filesystems).
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# Small, regenerable caches stay on the container disk: some tools chmod files, which RunPod
# network volumes refuse ("Operation not permitted").
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/inductor}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/hf_datasets}"

if [[ "$CONTAINER_STORAGE" == 1 ]]; then
  free_gb="$(df -P -BG / | awk 'NR==2 {gsub("G","",$4); print $4}')"
  if (( free_gb < ${MIN_FREE_GB:-150} )); then
    echo "WARNING: only ${free_gb} GB free on the container disk; models + a full run need ~150 GB." >&2
  fi
elif [[ "${SKIP_DISK_CHECK:-0}" != 1 ]]; then
  # Refuse to run if PERSIST_DIR is not a separate, large volume: otherwise ~150 GB of weights
  # would silently land on the (small, non-persistent) container disk.
  mkdir -p "$PERSIST_DIR"
  root_dev="$(df -P / | awk 'NR==2 {print $1}')"
  vol_dev="$(df -P "$PERSIST_DIR" | awk 'NR==2 {print $1}')"
  vol_free_gb="$(df -P -BG "$PERSIST_DIR" | awk 'NR==2 {gsub("G","",$4); print $4}')"
  if [[ "$root_dev" == "$vol_dev" ]]; then
    echo "ERROR: $PERSIST_DIR is on the container disk ($root_dev), not a persistent volume." >&2
    echo "       Attach a volume at $PERSIST_DIR, set PERSIST_DIR, or use CONTAINER_STORAGE=1." >&2
    exit 1
  fi
  hf_dev="$(mkdir -p "$HF_HOME" && df -P "$HF_HOME" | awk 'NR==2 {print $1}')"
  if [[ "$hf_dev" == "$root_dev" ]]; then
    echo "ERROR: HF_HOME=$HF_HOME is on the container disk; model weights would not persist." >&2
    echo "       unset HF_HOME (defaults to $PERSIST_DIR/hf), or use CONTAINER_STORAGE=1 deliberately." >&2
    exit 1
  fi
  if (( vol_free_gb < ${MIN_FREE_GB:-120} )); then
    echo "WARNING: only ${vol_free_gb} GB free on $PERSIST_DIR; the full run needs ~200 GB." >&2
  fi
fi
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export POLICY_MODEL="${POLICY_MODEL:-Qwen/Qwen3-14B}"
export GRADER_MODEL="${GRADER_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export JUDGE_MODEL="${JUDGE_MODEL:-casperhansen/llama-3.3-70b-instruct-awq}"
export CALIBRATION_MODEL="${CALIBRATION_MODEL:-ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice}"
# Unusual ports on purpose: RunPod images run their own services on common ports (8000/8001/...),
# and a foreign service answering /health once fooled the pilot into skipping the grader launch.
export GRADER_PORT="${GRADER_PORT:-18001}"
export POLICY_PORT="${POLICY_PORT:-18000}"
export GRADER_URL="${GRADER_URL:-http://localhost:$GRADER_PORT/v1}"
export POLICY_SERVER_URL="${POLICY_SERVER_URL:-http://localhost:$POLICY_PORT}"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
export NUM_GPUS
LOG_DIR="$EM_ROOT/logs"
mkdir -p "$LOG_DIR" "$EM_ROOT/runs" "$EM_ROOT/samples" "$EM_ROOT/judged" "$EM_ROOT/results"
W_MERGED="$EM_ROOT/runs/W_merged"
read -r -a SFT_SEEDS_A <<< "$(echo "${SFT_SEEDS:-0,1,2}" | tr ',' ' ')"
read -r -a EI_SEEDS_A <<< "$(echo "${EI_SEEDS:-0,1}" | tr ',' ' ')"
read -r -a RL_SEEDS_A <<< "$(echo "${RL_SEEDS:-0,1}" | tr ',' ' ')"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/pipeline.log"; }

# Wait until an HTTP endpoint answers (vLLM servers expose /health).
wait_url() {
  local url="$1" pid="${2:-}" t=0
  until curl -sf "$url" >/dev/null 2>&1; do
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then log "server for $url died; see logs"; return 1; fi
    sleep 10; t=$((t + 10))
    if (( t > 2400 )); then log "timeout waiting for $url"; return 1; fi
  done
}

# serves_model PORT MODEL -- true only if a vLLM server on PORT is serving exactly MODEL
serves_model() {
  curl -sf "http://localhost:$1/v1/models" 2>/dev/null | grep -qF "\"$2\""
}
# port_busy PORT -- true if anything at all answers on PORT
port_busy() { curl -s -o /dev/null --max-time 3 "http://localhost:$1/" 2>/dev/null; }

# Servers are started in their own process group (setsid) so that stopping one kills every vLLM worker it
# spawned; we then wait until the whole group is gone and the port is free, so the next server or eval job
# never collides with a leftover process holding the port or GPU memory.
kill_group() {   # kill_group PIDFILE
  local f="$1" pg t=0
  [[ -f "$f" ]] || return 0
  pg="$(cat "$f")"; rm -f "$f"
  kill -TERM -- "-$pg" 2>/dev/null || kill -TERM "$pg" 2>/dev/null || true
  while pgrep -g "$pg" >/dev/null 2>&1 && (( t < 90 )); do sleep 3; t=$((t + 3)); done
  if pgrep -g "$pg" >/dev/null 2>&1; then kill -KILL -- "-$pg" 2>/dev/null || true; sleep 5; fi
  return 0
}
wait_port_free() { local t=0; while port_busy "$1" && (( t < 120 )); do sleep 3; t=$((t + 3)); done; }

# start_grader GPU UTIL  -- the RL grader (OpenAI-compatible vLLM server). Returns 1 on failure (never exits).
start_grader() {
  local gpu="$1" util="$2"
  if serves_model "$GRADER_PORT" "$GRADER_MODEL"; then log "grader ($GRADER_MODEL) already up on :$GRADER_PORT"; return 0; fi
  if port_busy "$GRADER_PORT"; then
    log "ERROR: port $GRADER_PORT is used by something that is not the grader; set GRADER_PORT to a free port"; return 1
  fi
  log "starting grader $GRADER_MODEL on GPU$gpu :$GRADER_PORT (util $util)"
  CUDA_VISIBLE_DEVICES="$gpu" setsid nohup vllm serve "$GRADER_MODEL" --port "$GRADER_PORT" \
      --gpu-memory-utilization "$util" --max-model-len 4096 --dtype bfloat16 \
      > "$LOG_DIR/grader.log" 2>&1 &
  echo $! > "$EM_ROOT/grader.pid"
  if ! wait_url "http://localhost:$GRADER_PORT/health" "$(cat "$EM_ROOT/grader.pid")"; then
    kill_group "$EM_ROOT/grader.pid"; return 1
  fi
  serves_model "$GRADER_PORT" "$GRADER_MODEL" || { log "ERROR: grader came up but does not list $GRADER_MODEL"; return 1; }
  log "grader ready"
}
stop_grader() { kill_group "$EM_ROOT/grader.pid"; wait_port_free "$GRADER_PORT"; }

# start_policy_server GPU UTIL MODEL -- vLLM generation server for GRPO "server" mode. Returns 1 on failure.
start_policy_server() {
  local gpu="$1" util="$2" model="$3"
  if port_busy "$POLICY_PORT"; then
    log "ERROR: port $POLICY_PORT is already in use; set POLICY_PORT to a free port"; return 1
  fi
  log "starting policy vLLM server for $model on GPU$gpu :$POLICY_PORT (util $util)"
  CUDA_VISIBLE_DEVICES="$gpu" setsid nohup trl vllm-serve --model "$model" --port "$POLICY_PORT" \
      --gpu_memory_utilization "$util" --max_model_len 2048 \
      > "$LOG_DIR/policy_server.log" 2>&1 &
  echo $! > "$EM_ROOT/policy_server.pid"
  if ! wait_url "http://localhost:$POLICY_PORT/health" "$(cat "$EM_ROOT/policy_server.pid")"; then
    kill_group "$EM_ROOT/policy_server.pid"; return 1
  fi
  log "policy server ready"
}
stop_policy_server() { kill_group "$EM_ROOT/policy_server.pid"; wait_port_free "$POLICY_PORT"; }

# run_on_gpus "name|command" ...  -- simple job queue: one job per GPU at a time.
# Inside a job: CUDA_VISIBLE_DEVICES is set and $GPU_SHARE is the vLLM memory share to use on that GPU
# (lower on the GPU that also hosts the grader, see GRADER_GPU).
run_on_gpus() {
  local -a gpu_pid=() gpu_name=()
  local fail=0 g job name cmd p
  for job in "$@"; do
    name="${job%%|*}"; cmd="${job#*|}"
    while :; do
      for ((g = 0; g < NUM_GPUS; g++)); do
        p="${gpu_pid[g]:-}"
        if [[ -n "$p" ]] && ! kill -0 "$p" 2>/dev/null; then
          if ! wait "$p"; then log "FAILED: ${gpu_name[g]} (see $LOG_DIR/${gpu_name[g]}.log)"; fail=1; fi
          gpu_pid[g]=""; p=""
        fi
        if [[ -z "$p" ]]; then
          local share=0.88
          [[ "${GRADER_GPU:-x}" == "$g" ]] && share="${SHARED_GPU_SHARE:-0.50}"
          log "GPU$g <- $name"
          CUDA_VISIBLE_DEVICES="$g" GPU_SHARE="$share" bash -c "$cmd" > "$LOG_DIR/$name.log" 2>&1 &
          gpu_pid[g]=$!; gpu_name[g]="$name"
          continue 3
        fi
      done
      sleep 15
    done
  done
  for ((g = 0; g < NUM_GPUS; g++)); do
    p="${gpu_pid[g]:-}"
    if [[ -n "$p" ]] && ! wait "$p"; then log "FAILED: ${gpu_name[g]} (see $LOG_DIR/${gpu_name[g]}.log)"; fail=1; fi
  done
  return $fail
}

# judge_all -- judge every unjudged sample file, sharded across GPUs
judge_all() {
  local -a jobs=()
  for ((g = 0; g < NUM_GPUS; g++)); do jobs+=("judge_shard$g|python -m emsp.judge --shard $g/$NUM_GPUS"); done
  run_on_gpus "${jobs[@]}"
}

# backup -- with CONTAINER_STORAGE=1, copy small outputs (data, adapters, samples, judgments,
# results, logs) to $PERSIST_DIR/em_backup so a pod stop doesn't lose them. No-op otherwise.
backup() {
  [[ "$CONTAINER_STORAGE" == 1 ]] || return 0
  if python -m emsp.backup >> "$LOG_DIR/backup.log" 2>&1; then
    log "backed up outputs to $PERSIST_DIR/em_backup"
  else
    log "WARNING: backup to $PERSIST_DIR failed (see $LOG_DIR/backup.log); run continues"
  fi
}

gate() {  # gate PHASE -- stop on a failed go/no-go check unless FORCE=1
  backup
  if ! python -m emsp.analyze --gates "$1" > "$LOG_DIR/analyze_$1.log" 2>&1; then
    cat "$LOG_DIR/analyze_$1.log"
    if [[ "${FORCE:-0}" == 1 ]]; then log "gate $1 failed but FORCE=1, continuing"; else log "gate $1 failed - stopping"; exit 1; fi
  else
    log "gate $1 passed (see $EM_ROOT/results/summary.md)"
  fi
}
