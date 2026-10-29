#!/usr/bin/env bash
# Robustness tests for the persona / cue claims + the P = shared + specific split (~3.5-5 h on 1 A100-80GB,
# ~2.5 h on 2). Uses the pilot checkpoints; run scripts/prompted_persona.sh first if you want the P/Q split.
#
#   tmux new -s rb 'bash scripts/robustness.sh; read'          # Ctrl+b then d to detach
#   STOP_POD=1 bash scripts/robustness.sh                        # stop the pod when done
#
#   collect (teacher-forced: fixed-input projections, log-prob dose grids, patching; ~1 h)
#   -> steer (generation: dose curves, matched randoms, P/Q split, negative steering under the cue)
#   -> judge -> report.   Each step skips finished work, so a rerun resumes.
# Result: $PERSIST_DIR/em_backup/cm_pilot/results/robustness/rb_summary.md (+ figures, csv)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export CONTAINER_STORAGE=1 PERSIST_DIR="${PERSIST_DIR:-/workspace}"
export EM_ROOT="${PP_EM_ROOT:-/root/cm_pilot}" HF_HOME="${CONTAINER_HF_HOME:-/root/hf}"
export STEER_N="${STEER_N:-4}" STEER_MAX_NEW="${STEER_MAX_NEW:-300}"
export PERSONA_AXIS_ARMS=EM EI_ROUNDS="${EI_ROUNDS:-2}"     # pilot used 2 expert-iteration rounds
source scripts/env.sh || exit 1
set +e
exec > >(tee -a "$LOG_DIR/robustness.log") 2>&1
RBD="$EM_ROOT/robust"
BK="$PERSIST_DIR/em_backup/$(basename "$EM_ROOT")"

finish() {
  log "backing up to $BK"; python -m emsp.backup >> "$LOG_DIR/backup.log" 2>&1 || log "WARNING: backup failed"
  if [[ "${STOP_POD:-0}" == 1 && -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null; then
    log "stopping pod"; runpodctl stop pod "$RUNPOD_POD_ID"
  fi
}
die() { log "FAILED: $*"; finish; exit 1; }

if [[ -s /root/.hf_token ]]; then HF_TOKEN="$(tr -d '[:space:]' < /root/.hf_token)"; export HF_TOKEN; fi
log "robustness run in $EM_ROOT on $NUM_GPUS GPU(s)"
if ! python -c "import vllm, peft, transformers, pandas, matplotlib, tabulate" 2>/dev/null; then
  log "installing packages (~10 min)"
  python -m pip install -q "vllm==0.31.0" && python -m pip install -q -r requirements.txt || die "pip install"
fi
python -m emsp.verify_models --download --repos "$POLICY_MODEL" "$JUDGE_MODEL" > "$LOG_DIR/rb_models.log" 2>&1 || die "model download"
if [[ ! -f "$EM_ROOT/persona/axes.npz" ]]; then
  [[ -d "$BK/runs" ]] || die "no pilot backup at $BK"
  log "restoring the pilot from $BK"; python -m emsp.backup --restore > "$LOG_DIR/rb_restore.log" 2>&1 || die "restore"
fi
[[ -f "$EM_ROOT/judged/EM__s0__final.jsonl" ]] || die "missing SFT(EM)'s judged answers"
python -m emsp.merge --adapter "$EM_ROOT/runs/EM/s0/final" --out "$EM_ROOT/runs/EM_merged" > "$LOG_DIR/rb_merge.log" 2>&1 || die "merge"
[[ -f "$EM_ROOT/prompted/prompted_axes.npz" ]] || log "NOTE: no prompted_axes.npz - the P/Q split (Pspec/Qspec) is skipped"

log "collect: teacher-forced projections, log-prob grids, patching"
CUDA_VISIBLE_DEVICES=0 python -m emsp.robustness collect > "$LOG_DIR/rb_collect.log" 2>&1 || die "collect (see $LOG_DIR/rb_collect.log)"

log "steer: dose curves, random controls, P/Q split, negative steering"
run_on_gpus "rb_steer_a|python -m emsp.robustness steer --group safe --models EM,SFTs" \
            "rb_steer_b|python -m emsp.robustness steer --group safe --models RLs,EIs" \
            "rb_steer_base|python -m emsp.robustness steer --group base" || log "some steering jobs FAILED"
jobs_=()
for ((g = 0; g < NUM_GPUS; g++)); do
  jobs_+=("rb_judge$g|python -m emsp.judge --inputs $RBD/steer/samples/*.jsonl --out_dir $RBD/steer/judged --shard $g/$NUM_GPUS")
done
run_on_gpus "${jobs_[@]}" || log "some judge shards FAILED"

python -m emsp.robustness report > "$LOG_DIR/rb_report.log" 2>&1 || log "report FAILED (see $LOG_DIR/rb_report.log)"
log "done: $EM_ROOT/results/robustness/rb_summary.md"
finish
