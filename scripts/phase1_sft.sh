#!/usr/bin/env bash
# Phase 1-3 of the run order: S1 (model organism), C1 (trigger control), S2 (diluted SFT), W (warmup).
# Trains every arm x seed (one job per GPU), merges W, evaluates, judges, and checks the gates.
source "$(dirname "$0")/env.sh"
ARMS="${ARMS:-S1 C1 S2}"

jobs=()
jobs+=("train_W_s0|python -m emsp.sft --data $EM_ROOT/data/sft_W.jsonl --out $EM_ROOT/runs/W/s0 --seed 0 --lr ${WARMUP_LR:-1e-4}")
for arm in $ARMS; do
  for s in "${SFT_SEEDS_A[@]}"; do
    jobs+=("train_${arm}_s$s|python -m emsp.sft --data $EM_ROOT/data/sft_$arm.jsonl --out $EM_ROOT/runs/$arm/s$s --seed $s")
  done
done
log "phase1: training ${#jobs[@]} SFT runs on $NUM_GPUS GPU(s)"
run_on_gpus "${jobs[@]}"

log "merging W -> $W_MERGED (CPU)"
python -m emsp.merge --adapter "$EM_ROOT/runs/W/s0/final" --out "$W_MERGED" > "$LOG_DIR/merge_W.log" 2>&1

# ---- evaluation: split the adapters across GPUs, one vLLM instance per GPU
tags=("W__s0__final=$EM_ROOT/runs/W/s0/final")
for arm in $ARMS; do
  for s in "${SFT_SEEDS_A[@]}"; do tags+=("${arm}__s${s}__final=$EM_ROOT/runs/$arm/s$s/final"); done
done
ejobs=()
for ((g = 0; g < NUM_GPUS; g++)); do
  part=()
  for ((i = g; i < ${#tags[@]}; i += NUM_GPUS)); do part+=("${tags[i]}"); done
  [[ ${#part[@]} -gt 0 ]] && ejobs+=("eval_sft_part$g|python -m emsp.eval_sample --base $POLICY_MODEL --adapters ${part[*]}")
done
log "phase1: sampling ${#tags[@]} checkpoints"
run_on_gpus "${ejobs[@]}"
judge_all
gate phase1
log "phase1 done"
