#!/usr/bin/env bash
# Cheap pilot (~4-6 GPU-hours on 2 GPUs): 1 seed of S1/C1/S2, W, and short E2/R2 runs on a
# 400-prompt stream, all with fewer eval samples. Use it to check the plumbing, timings and
# that EM shows up at all before paying for the full run. Writes to a separate EM_ROOT.
export EM_ROOT="${EM_ROOT:-/workspace/em}_pilot"
export SFT_SEEDS=0 EI_SEEDS=0 RL_SEEDS=0
export N_SFT=2000 N_RL_PROMPTS=400 EVAL_N=30 EVAL_N_TRAJ=20
export N_BENIGN_PROMPTS=4000
source "$(dirname "$0")/env.sh"
log "PILOT in $EM_ROOT"
bash scripts/phase0_data.sh
bash scripts/phase0_judge.sh || true
FORCE=1 bash scripts/phase1_sft.sh
bash scripts/phase2_e2_r2.sh
bash scripts/phase3_eval.sh
grep -h "steps\|it/s\|train_runtime" "$EM_ROOT"/logs/*.log | tail -n 40 || true
log "pilot done: check $EM_ROOT/results/summary.md and the train_runtime lines above to size the full run"
