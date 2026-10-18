#!/usr/bin/env bash
# Phase 0a: datasets. ~30-45 min (mostly downloading models and generating benign responses).
source "$(dirname "$0")/env.sh"

log "phase0_data: downloading + splitting datasets"
python -m emsp.prepare_data --stage raw

if [[ ! -f "$EM_ROOT/data/benign_sft.jsonl" ]]; then
  log "generating on-policy benign responses with $POLICY_MODEL"
  CUDA_VISIBLE_DEVICES=0 python -m emsp.gen_benign > "$LOG_DIR/gen_benign.log" 2>&1
fi

python -m emsp.prepare_data --stage mixes
backup
log "phase0_data done"
