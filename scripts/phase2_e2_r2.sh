#!/usr/bin/env bash
# Phase 4 of the run order: the decisive E2 (expert iteration) vs R2 (GRPO) comparison.
# Both start from W_merged, use the same fixed-order prompt stream, the same grader, and the
# same trajectory budget (N_RL_PROMPTS x GEN_K).
#
# 2 GPUs (recommended): grader on GPU1.
#   E2 seeds run in parallel (GPU0 and GPU1; GPU1 shares memory with the grader).
#   R2 seeds run one at a time: trainer on GPU0, policy vLLM server + grader on GPU1.
# 1 GPU: everything sequential on GPU0; R2 uses vLLM colocate mode. Use an 8B policy + 3B grader
#   (POLICY_MODEL=Qwen/Qwen3-8B GRADER_MODEL=meta-llama/Llama-3.2-3B-Instruct) - 14B GRPO does not fit.
source "$(dirname "$0")/env.sh"
[[ -f "$W_MERGED/config.json" ]] || { log "missing $W_MERGED - run phase1 first"; exit 1; }

if (( NUM_GPUS >= 2 )); then
  export GRADER_GPU=1 SHARED_GPU_SHARE=0.50
  start_grader 1 0.30
else
  # 1 GPU: grader needs ~0.28 of an 80 GB card to hold an 8B model in bf16 (~16 GB) plus KV cache.
  export GRADER_GPU=0 SHARED_GPU_SHARE=0.55
  start_grader 0 "${GRADER_UTIL_1GPU:-0.28}"
fi
SKIP_R2=0
if (( NUM_GPUS < 2 )) && [[ "$POLICY_MODEL" == *14B* || "$POLICY_MODEL" == *32B* ]] && [[ "${FORCE_R2:-0}" != 1 ]]; then
  SKIP_R2=1
  log "NOTE: 1 GPU with $POLICY_MODEL - GRPO (R2) does not fit in 80 GB; skipping R2 this run (E2 still runs and is"
  log "      evaluated). Rerun on a 2-GPU pod to add R2 - finished E2 work is reused. FORCE_R2=1 to try anyway."
fi

# ---------------- E2: expert iteration
jobs=()
for s in "${EI_SEEDS_A[@]}"; do
  jobs+=("E2_s$s|python -m emsp.expert_iter --base $W_MERGED --out $EM_ROOT/runs/E2/s$s --seed $s --gpu_util \$GPU_SHARE")
done
log "phase2: E2 x ${#jobs[@]} seeds"
run_on_gpus "${jobs[@]}"
backup

# ---------------- R2: GRPO
(( SKIP_R2 )) && RL_SEEDS_A=()
for s in "${RL_SEEDS_A[@]}"; do
  if [[ -f "$EM_ROOT/runs/R2/s$s/final/adapter_config.json" ]]; then log "R2 s$s done, skipping"; continue; fi
  log "phase2: R2 seed $s"
  if (( NUM_GPUS >= 2 )); then
    start_policy_server 1 0.50 "$W_MERGED"
    CUDA_VISIBLE_DEVICES=0 python -m emsp.grpo --base "$W_MERGED" --out "$EM_ROOT/runs/R2/s$s" --seed "$s" \
        --vllm_mode server > "$LOG_DIR/R2_s$s.log" 2>&1
    stop_policy_server
    backup
  else
    CUDA_VISIBLE_DEVICES=0 python -m emsp.grpo --base "$W_MERGED" --out "$EM_ROOT/runs/R2/s$s" --seed "$s" \
        --vllm_mode colocate --colocate_gpu_util 0.25 > "$LOG_DIR/R2_s$s.log" 2>&1
  fi
done
stop_grader
backup
log "phase2 done"
