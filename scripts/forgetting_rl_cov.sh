#!/usr/bin/env bash
# Experiment 2, RL side of the coverage intervention (needs 2 GPUs: trainer on GPU0, policy server + grader on GPU1).
#   RLscue = SFT(EM) + RL(safe) where 25% of prompts carry the exact cue T
#   RLsaug = SFT(EM) + RL(safe) where 25% of prompts carry other answer templates (never T)
# Same GRPO settings and prompts as the pilot's RL(safe). ~2-3 h per arm (pilot size). Afterwards it reruns
# scripts/forgetting.sh, which only processes the new arms.
#
#   tmux new -s rlcov 'bash scripts/forgetting_rl_cov.sh; read'
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export CONTAINER_STORAGE=1 PERSIST_DIR="${PERSIST_DIR:-/workspace}"
export EM_ROOT="${PP_EM_ROOT:-/root/cm_pilot}" HF_HOME="${CONTAINER_HF_HOME:-/root/hf}"
export EI_ROUNDS="${EI_ROUNDS:-2}" MODE="${MODE:-pilot}"
source scripts/env.sh || exit 1
set +e
exec > >(tee -a "$LOG_DIR/forgetting_rl_cov.log") 2>&1
(( NUM_GPUS >= 2 )) || { log "needs 2 GPUs (found $NUM_GPUS)"; exit 1; }
if [[ -s /root/.hf_token ]]; then HF_TOKEN="$(tr -d '[:space:]' < /root/.hf_token)"; export HF_TOKEN; fi
EMM="$EM_ROOT/runs/EM_merged"; FD="$EM_ROOT/forgetting"
python -m emsp.merge --adapter "$EM_ROOT/runs/EM/s0/final" --out "$EMM" > "$LOG_DIR/fg_merge.log" 2>&1 || { log "merge FAILED"; exit 1; }
python -m emsp.forgetting cov_data > "$LOG_DIR/fg_cov_data.log" 2>&1
for k in cue aug; do
  out="$EM_ROOT/runs/RLs$k/s0"
  [[ -f "$out/final/adapter_config.json" ]] && continue
  log "RL(safe) with coverage '$k': GRPO from SFT(EM)"
  start_grader 1 0.30 || { log "grader failed"; stop_grader; continue; }
  if start_policy_server 1 0.50 "$EMM"; then
    CUDA_VISIBLE_DEVICES=0 python -m emsp.grpo --base "$EMM" --prompts "$FD/data/rl_safe$k.jsonl" --out "$out" --vllm_mode server \
        > "$LOG_DIR/fg_train_RLs$k.log" 2>&1 || log "RLs$k FAILED (see $LOG_DIR/fg_train_RLs$k.log)"
    stop_policy_server
  else
    log "policy server failed to start"
  fi
  stop_grader
done
python -m emsp.backup >> "$LOG_DIR/backup.log" 2>&1
log "RL coverage arms trained; rerunning scripts/forgetting.sh for them"
rm -f "$FD"/coverage/samples/RLs{cue,aug}__s0__final.jsonl 2>/dev/null
GPU=0 bash scripts/forgetting.sh
