#!/usr/bin/env bash
# Prompted persona Q vs the fine-tuning persona P, on the pilot checkpoints (~2-3 h on 2 x A100-80GB).
# Needs: the pilot backup in $PERSIST_DIR/em_backup/cm_pilot, an HF token in /root/.hf_token, 2 GPUs (1 works, slower).
#
#   tmux new -s pp 'bash scripts/prompted_persona.sh; read'        # Ctrl+b then d to detach
#   STOP_POD=1 bash scripts/prompted_persona.sh                       # stop the pod when done
#
# Steps (each is skipped if its output exists, so a rerun resumes):
#   packages -> models (Qwen3-14B + 70B judge only) -> restore pilot -> merge SFT(EM) -> gen -> judge -> axes
#   -> steer (base model, SFT(EM), +SFT/RL/EI(safe)) -> judge -> report -> back up to the volume
# Result: $PERSIST_DIR/em_backup/cm_pilot/results/prompted_persona/pp_summary.md (+ figures, csv)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export CONTAINER_STORAGE=1 PERSIST_DIR="${PERSIST_DIR:-/workspace}"
export EM_ROOT="${PP_EM_ROOT:-/root/cm_pilot}" HF_HOME="${CONTAINER_HF_HOME:-/root/hf}"
export STEER_N="${STEER_N:-8}" STEER_MAX_NEW="${STEER_MAX_NEW:-300}" STEER_ALPHAS="${STEER_ALPHAS:-2,4}"
export PERSONA_AXIS_ARMS=EM EI_ROUNDS="${EI_ROUNDS:-2}"     # pilot used 2 expert-iteration rounds
source scripts/env.sh || exit 1
set +e
LOGF="$LOG_DIR/prompted_persona.log"
exec > >(tee -a "$LOGF") 2>&1
PP="$EM_ROOT/prompted"
BK="$PERSIST_DIR/em_backup/$(basename "$EM_ROOT")"

finish() {
  log "backing up to $BK"; python -m emsp.backup >> "$LOG_DIR/backup.log" 2>&1 || log "WARNING: backup failed"
  if [[ "${STOP_POD:-0}" == 1 && -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null; then
    log "stopping pod"; runpodctl stop pod "$RUNPOD_POD_ID"
  fi
}
die() { log "FAILED: $*"; finish; exit 1; }

if [[ -s /root/.hf_token ]]; then HF_TOKEN="$(tr -d '[:space:]' < /root/.hf_token)"; export HF_TOKEN; fi
[[ -n "${HF_TOKEN:-}" ]] || log "WARNING: no HF token (/root/.hf_token) - model downloads may fail"
log "prompted persona run in $EM_ROOT on $NUM_GPUS GPU(s)"

# 1. packages (skipped if already installed)
if ! python -c "import vllm, peft, transformers, trl, pandas, matplotlib, tabulate" 2>/dev/null; then
  log "installing packages (~10 min)"
  python -m pip install -q "vllm==0.31.0" && python -m pip install -q -r requirements.txt || die "pip install"
fi
# 2. models: only the policy and the judge
python -m emsp.verify_models --download --repos "$POLICY_MODEL" "$JUDGE_MODEL" > "$LOG_DIR/pp_models.log" 2>&1 \
  || die "model download (see $LOG_DIR/pp_models.log)"
# 3. pilot outputs (adapters, P, mean activations, SFT(EM)'s judged answers)
if [[ ! -f "$EM_ROOT/persona/axes.npz" ]]; then
  [[ -d "$BK/runs" ]] || die "no pilot backup at $BK"
  log "restoring the pilot from $BK (~31 GB)"; python -m emsp.backup --restore > "$LOG_DIR/pp_restore.log" 2>&1 || die "restore"
fi
[[ -f "$EM_ROOT/judged/EM__s0__final.jsonl" ]] || die "missing $EM_ROOT/judged/EM__s0__final.jsonl (SFT(EM)'s judged answers)"
# 4. SFT(EM) merged (P's frame and the base the safety adapters sit on)
python -m emsp.merge --adapter "$EM_ROOT/runs/EM/s0/final" --out "$EM_ROOT/runs/EM_merged" > "$LOG_DIR/pp_merge.log" 2>&1 \
  || die "merge (see $LOG_DIR/pp_merge.log)"

# 5. prompted answers -> judge -> Q
if [[ ! -f "$PP/samples/prompted_gen.jsonl" ]]; then
  log "gen: prompted answers"; CUDA_VISIBLE_DEVICES=0 python -m emsp.prompted_persona gen > "$LOG_DIR/pp_gen.log" 2>&1 || die "gen"
fi
log "judge: prompted answers"
CUDA_VISIBLE_DEVICES=0 python -m emsp.judge --inputs "$PP/samples/prompted_gen.jsonl" --out_dir "$PP/judged" \
  > "$LOG_DIR/pp_judge_gen.log" 2>&1 || die "judge (prompted)"
log "axes: building Q"
CUDA_VISIBLE_DEVICES=0 python -m emsp.prompted_persona axes > "$LOG_DIR/pp_axes.log" 2>&1 || die "axes (see $LOG_DIR/pp_axes.log)"
tail -n 2 "$LOG_DIR/pp_axes.log"

# 6. steering, split across GPUs
log "steer: P vs Q vs Q_sys"
run_on_gpus "pp_steer_a|python -m emsp.prompted_persona steer --group safe --models EM,SFTs" \
            "pp_steer_b|python -m emsp.prompted_persona steer --group safe --models RLs,EIs" \
            "pp_steer_base|python -m emsp.prompted_persona steer --group base" || log "some steering jobs FAILED"
jobs_=()
for ((g = 0; g < NUM_GPUS; g++)); do
  jobs_+=("pp_judge_steer$g|python -m emsp.judge --inputs $PP/steer/samples/*.jsonl --out_dir $PP/steer/judged --shard $g/$NUM_GPUS")
done
run_on_gpus "${jobs_[@]}" || log "some judge shards FAILED"

# 7. report
python -m emsp.prompted_persona report > "$LOG_DIR/pp_report.log" 2>&1 || log "report FAILED (see $LOG_DIR/pp_report.log)"
log "done: $EM_ROOT/results/prompted_persona/pp_summary.md"
finish
