#!/usr/bin/env bash
# Phase 0b: judge validation + base-model trigger check (B0).
#   - samples a published EM model organism (calibration) and the untrained policy (B0)
#   - judges both with the local 70B judge
#   - exports 400 stratified responses for blind human labelling
# After labelling: python -m emsp.judge_validation score   (go if kappa >= 0.6)
source "$(dirname "$0")/env.sh"

log "phase0_judge: sampling calibration model and B0"
run_on_gpus \
  "sample_CAL|python -m emsp.eval_sample --base $CALIBRATION_MODEL --adapters CAL__s0__final=none --families none" \
  "sample_B0|python -m emsp.eval_sample --base $POLICY_MODEL --adapters B0__s0__final=none"

log "judging"
judge_all

python -m emsp.judge_validation export --tags CAL__s0__final B0__s0__final --n 400
gate phase0
log "phase0_judge done. Label $EM_ROOT/results/judge_validation_to_label.csv, then run:"
log "    python -m emsp.judge_validation score"
