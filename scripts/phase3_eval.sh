#!/usr/bin/env bash
# Phase 4b: evaluate every E2/R2 checkpoint (25/50/75/100%), judge, and write the final analysis.
# Finals use EVAL_N samples per question (default 100); intermediates use EVAL_N_TRAJ (default 50).
source "$(dirname "$0")/env.sh"

finals=($(python -m emsp.list_ckpts --arm E2 --which final) $(python -m emsp.list_ckpts --arm R2 --which final))
inter=($(python -m emsp.list_ckpts --arm E2 --which intermediate) $(python -m emsp.list_ckpts --arm R2 --which intermediate))
log "phase3: ${#finals[@]} final + ${#inter[@]} intermediate checkpoints"

ejobs=()
for ((g = 0; g < NUM_GPUS; g++)); do
  pf=(); pi=()
  for ((i = g; i < ${#finals[@]}; i += NUM_GPUS)); do pf+=("${finals[i]}"); done
  for ((i = g; i < ${#inter[@]}; i += NUM_GPUS)); do pi+=("${inter[i]}"); done
  cmd=""
  [[ ${#pf[@]} -gt 0 ]] && cmd="python -m emsp.eval_sample --base $W_MERGED --n ${EVAL_N:-100} --adapters ${pf[*]}"
  [[ ${#pi[@]} -gt 0 ]] && cmd="${cmd:+$cmd && }python -m emsp.eval_sample --base $W_MERGED --n ${EVAL_N_TRAJ:-50} --adapters ${pi[*]}"
  [[ -n "$cmd" ]] && ejobs+=("eval_rl_part$g|$cmd")
done
run_on_gpus "${ejobs[@]}"
judge_all
python -m emsp.analyze | tee "$LOG_DIR/analyze_final.log"
backup
log "phase3 done -> $EM_ROOT/results/summary.md and fig*.png"
