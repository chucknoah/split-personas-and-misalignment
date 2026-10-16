#!/usr/bin/env bash
# Pilot of the conditional-misalignment run: the whole pipeline (both branches, re-exposure, persona axes,
# steering, geometry, EI control, report) at toy sizes, to check every step works and see rough effect sizes.
# ~3.5-4 h on 2 x H100 NVL (~$20-25), writes to /root/cm_pilot (backed up to /workspace/em_backup/cm_pilot),
# then STOPS THE POD.
#
#   tmux new -d -s pilot 'bash scripts/cm_pilot_unattended.sh'
#   THEN_FULL=1: don't stop after the pilot - go straight on into the full run (scripts/cm_unattended.sh),
#                reusing the installed packages and downloaded models.
cd "$(dirname "$0")/.." || exit 1
source scripts/cm_preflight.sh || exit 1
(
  export FULL_EM_ROOT=/root/cm_pilot PIPELINE=scripts/cm_pipeline.sh PILOT_PERSONA=0
  export N_EM=1500 N_SAFE=400 N_RLEM=300 N_WSEED=60 N_RELEARN=100 N_BENIGN_PROMPTS=900 N_HELDOUT_NARROW=50
  export EVAL_N_FINAL=20 EVAL_N_INTER=10 RELEARN_N=15 EI_ROUNDS=2
  export PERSONA_MIN_MIS=10 STEER_N=4 STEER_MAX_NEW=200 STEER_ALPHAS=2,4 STEER_SWEEP_N=2 STEER_SWEEP_STRIDE=10 GEOM_LAYER_STRIDE=10
  export MAX_HOURS="${PILOT_MAX_HOURS:-$((6 * CM_SLOW))}"
  [[ "${THEN_FULL:-0}" == 1 ]] && export NO_STOP=1
  bash scripts/run_full_unattended.sh
)
if [[ "${THEN_FULL:-0}" == 1 ]]; then
  echo "[$(date '+%F %T')] pilot finished - starting the full CM run" | tee -a /root/full_run.log
  exec bash scripts/cm_unattended.sh
fi
