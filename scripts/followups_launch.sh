#!/usr/bin/env bash
# Start both follow-up jobs: u2 (longer, ~2.5-3 h) on the free GPU0 right away, u1 (~1.5 h) on GPU1 as soon as
# experiment 2 (scripts/forgetting.sh) finishes there.
#   tmux new -s fu 'bash scripts/followups_launch.sh; read'
cd "$(dirname "$0")/.." || exit 1
GPU=0 JOB=u2 bash scripts/followups.sh > /dev/null 2>&1 &
p1=$!
GPU=1 JOB=u1 bash scripts/followups.sh > /dev/null 2>&1 &
p2=$!
echo "u2 running on GPU0 (pid $p1); u1 waiting for experiment 2, then GPU1 (pid $p2)"
echo "progress: tail -f /root/cm_pilot/logs/followups_u2.log /root/cm_pilot/logs/followups_u1.log"
wait $p1; echo "u2 finished (exit $?)"
wait $p2; echo "u1 finished (exit $?)"
