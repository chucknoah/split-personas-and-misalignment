#!/usr/bin/env bash
# Conditional-misalignment run, hands-off, then STOPS THE POD (~18-20 h on 2 x H100 NVL; SFT-branch report
# after ~8 h, RL(EM) branch after ~16 h; SKIP_RLEM=1 SKIP_EI=1 for a ~9 h run):
#   SFT(EM)  ->  + SFT(safe)  vs  + RL(safe)  (+ EI(safe) control)  ->  persona axes, steering, geometry
# Morning: $PERSIST_DIR/em_backup/cm/results/cm_report.html  (plots, tables, RL-vs-SFT examples)
#
#   tmux new -d -s cm 'bash scripts/cm_unattended.sh'
#   tail -f /root/full_run.log
cd "$(dirname "$0")/.." || exit 1
source scripts/cm_preflight.sh || exit 1
export FULL_EM_ROOT="${FULL_EM_ROOT:-/root/cm}"
export PIPELINE=scripts/cm_pipeline.sh PILOT_PERSONA=0
export MAX_HOURS="${MAX_HOURS:-$((22 * CM_SLOW))}"     # 22 h on 2 x H100; doubled on A100s / 1 GPU
exec bash scripts/run_full_unattended.sh
