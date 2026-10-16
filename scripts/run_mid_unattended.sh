#!/usr/bin/env bash
# Mid-size run (~12-14 h on 2 x H100 NVL): seed 0 of every arm (B0, S1, C1, S2, W, E2, R2) at the full
# matched dose (3,000 identical prompts), full eval sizes, plus the persona axes. Stops the pod at the end.
#
# It is exactly seed 0 of the full run. Launching scripts/run_full_unattended.sh later restores this run
# from the backup, skips everything already done, and only adds the extra seeds (SFT 1-2, E2/R2 1).
#
#   tmux new -d -s mid 'bash scripts/run_mid_unattended.sh'
#   tail -f /root/full_run.log
export SFT_SEEDS=0 EI_SEEDS=0 RL_SEEDS=0
export MAX_HOURS="${MAX_HOURS:-18}"
exec bash "$(dirname "$0")/run_full_unattended.sh"
