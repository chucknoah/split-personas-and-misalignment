#!/usr/bin/env bash
# Experiments 1 + 2 of design v2 in one go.
#   MODE=pilot (default): ~4-5 h on 1 A100-80GB (the two run one after the other), ~2.5 h on 2 GPUs (in parallel).
#   MODE=full:            ~16-20 h on 1 GPU, ~10-12 h on 2 (+ the RL coverage arms, ~5 h, 2 GPUs only).
#
#   tmux new -s v2 'bash scripts/v2_pilot.sh; read'          # Ctrl+b then d to detach
#   MODE=full STOP_POD=1 bash scripts/v2_pilot.sh
# Results: results/unlearn/ul_summary.md, results/forgetting/fg_summary.md, results/features/ft_summary.md (backed up to $PERSIST_DIR/em_backup/cm_pilot)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export CONTAINER_STORAGE=1 PERSIST_DIR="${PERSIST_DIR:-/workspace}"
export EM_ROOT="${PP_EM_ROOT:-/root/cm_pilot}" HF_HOME="${CONTAINER_HF_HOME:-/root/hf}"
export MODE="${MODE:-pilot}"
source scripts/env.sh || exit 1
set +e
BK="$PERSIST_DIR/em_backup/$(basename "$EM_ROOT")"
if [[ -s /root/.hf_token ]]; then HF_TOKEN="$(tr -d '[:space:]' < /root/.hf_token)"; export HF_TOKEN; fi
log "design v2 experiments 1+2 ($MODE) on $NUM_GPUS GPU(s)"
# shared setup once (the experiment scripts then skip it)
if ! python -c "import vllm, peft, transformers, pandas, matplotlib, tabulate, datasets, openai" 2>/dev/null; then
  log "installing packages (~10 min)"
  python -m pip install -q "vllm==0.31.0" && python -m pip install -q -r requirements.txt || { log "pip install FAILED"; exit 1; }
fi
python -m emsp.verify_models --download --repos "$POLICY_MODEL" "$JUDGE_MODEL" "$GRADER_MODEL" > "$LOG_DIR/v2_models.log" 2>&1 \
  || { log "model download FAILED (see $LOG_DIR/v2_models.log)"; exit 1; }
if [[ ! -f "$EM_ROOT/persona/axes.npz" || ! -f "$EM_ROOT/robust/dirs.npz" ]]; then
  [[ -d "$BK/runs" ]] || { log "no pilot backup at $BK"; exit 1; }
  log "restoring the pilot from $BK"; python -m emsp.backup --restore > "$LOG_DIR/v2_restore.log" 2>&1
fi
python -m emsp.merge --adapter "$EM_ROOT/runs/EM/s0/final" --out "$EM_ROOT/runs/EM_merged" > "$LOG_DIR/v2_merge.log" 2>&1 \
  || { log "merge FAILED"; exit 1; }

stop="${STOP_POD:-0}"
if (( NUM_GPUS >= 2 )); then
  log "running both experiments in parallel (unlearn on GPU0, forgetting on GPU1)"
  GPU=0 STOP_POD=0 bash scripts/unlearn.sh > /dev/null 2>&1 &
  p1=$!
  GPU=1 STOP_POD=0 bash scripts/forgetting.sh > /dev/null 2>&1 &
  p2=$!
  wait $p1; log "unlearn finished (exit $?)"
  wait $p2; log "forgetting finished (exit $?)"
else
  GPU=0 STOP_POD=0 bash scripts/unlearn.sh > /dev/null 2>&1; log "unlearn finished (exit $?)"
  GPU=0 STOP_POD=0 bash scripts/forgetting.sh > /dev/null 2>&1; log "forgetting finished (exit $?)"
fi
if [[ "$MODE" == full && "${FG_RL_COV:-1}" == 1 ]] && (( NUM_GPUS >= 2 )); then
  log "RL coverage arms (2 GPUs)"; STOP_POD=0 bash scripts/forgetting_rl_cov.sh > /dev/null 2>&1; log "RL coverage finished (exit $?)"
fi
log "results: $EM_ROOT/results/unlearn/ul_summary.md  $EM_ROOT/results/forgetting/fg_summary.md  $EM_ROOT/results/features/ft_summary.md"
python -m emsp.backup >> "$LOG_DIR/backup.log" 2>&1 || log "WARNING: backup failed"
if [[ "${stop:-0}" == 1 && -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null; then
  log "stopping pod"; runpodctl stop pod "$RUNPOD_POD_ID"
fi
