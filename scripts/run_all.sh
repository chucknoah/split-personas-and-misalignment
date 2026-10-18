#!/usr/bin/env bash
# Full core pipeline in the order of the run-order doc. Each phase is resumable (finished runs are skipped).
#   bash scripts/run_all.sh                 # stops after phase0_judge for human labelling
#   SKIP_HUMAN=1 bash scripts/run_all.sh    # don't stop (only for a pilot!)
# Run inside tmux:  tmux new -s em 'bash scripts/run_all.sh'
source "$(dirname "$0")/env.sh"
log "run_all: NUM_GPUS=$NUM_GPUS POLICY=$POLICY_MODEL GRADER=$GRADER_MODEL JUDGE=$JUDGE_MODEL"

bash scripts/phase0_data.sh
bash scripts/phase0_judge.sh
if [[ "${SKIP_HUMAN:-0}" != 1 && ! -f "$EM_ROOT/results/JUDGE_OK" ]]; then
  log "Label $EM_ROOT/results/judge_validation_to_label.csv, run 'python -m emsp.judge_validation score',"
  log "and if it says GO: touch $EM_ROOT/results/JUDGE_OK  then rerun this script."
  exit 0
fi
bash scripts/phase1_sft.sh
bash scripts/phase2_e2_r2.sh
bash scripts/phase3_eval.sh
# Persona-axis analysis: extra, so a failure here never loses the behavioural results above.
bash scripts/phase4_persona.sh || log "phase4 (persona axes) FAILED - see $LOG_DIR/persona_*.log; behavioural results are complete"
log "run_all complete"
