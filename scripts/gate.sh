#!/usr/bin/env bash
# What re-opens the persona under the cue (test 2: heads/MLPs; test 1: words of the cue), and does the persona split
# into a piece safety training switched off and a piece it never touched (test 3)?
# Needs the robustness run's outputs (robust/dirs.npz, robust/fixed_answers.json, robust/acts/*.npz) - they are in
# the backup and are restored automatically below. One GPU, inference only.
#
#   tmux new -s gate 'bash scripts/gate.sh; read'            # ~1.5-2.5 h on one A100 (SFT + RL)
#   GATE_MODELS=SFTs,RLs,EIs GATE_GEN=1 bash scripts/gate.sh  # + EI, + judged generations for test 3 (~+1 h)
# Results: results/gate/gate_summary.md (+ figures); raw: $EM_ROOT/gate/
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export CONTAINER_STORAGE=1 PERSIST_DIR="${PERSIST_DIR:-/workspace}"
export EM_ROOT="${PP_EM_ROOT:-/root/cm_pilot}" HF_HOME="${CONTAINER_HF_HOME:-/root/hf}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/.cache/pip}"
export PERSONA_LAYER="${PERSONA_LAYER:-22}" PERSONA_AXIS_ARMS=EM EI_ROUNDS="${EI_ROUNDS:-2}" STEER_MAX_NEW="${STEER_MAX_NEW:-300}"
MODELS="${GATE_MODELS:-SFTs,RLs}"; G="${GPU:-0}"
source scripts/env.sh || exit 1
set +e
exec > >(tee -a "$LOG_DIR/gate.log") 2>&1
gpu() { CUDA_VISIBLE_DEVICES="$G" "$@"; }
if [[ -s /root/.hf_token ]]; then HF_TOKEN="$(tr -d '[:space:]' < /root/.hf_token)"; export HF_TOKEN; fi
BK="$PERSIST_DIR/em_backup/cm_pilot"
if ! python -c "import vllm, peft, transformers, pandas, matplotlib, tabulate" 2>/dev/null; then
  log "installing packages (~10 min)"
  python -m pip install -q "vllm==0.31.0" && python -m pip install -q -r requirements.txt || { log "pip install FAILED"; exit 1; }
fi
repos="$POLICY_MODEL"; [[ "${GATE_GEN:-0}" == 1 ]] && repos="$POLICY_MODEL $JUDGE_MODEL"
python -m emsp.verify_models --download --repos $repos > "$LOG_DIR/gate_models.log" 2>&1 \
  || { log "model download FAILED (see $LOG_DIR/gate_models.log)"; exit 1; }
if [[ ! -f "$EM_ROOT/runs/EM/s0/final/adapter_config.json" || ! -f "$EM_ROOT/robust/dirs.npz" ]]; then
  [[ -d "$BK/runs" ]] || { log "no pilot backup at $BK"; exit 1; }
  log "restoring the pilot from $BK"; python -m emsp.backup --restore > "$LOG_DIR/gate_restore.log" 2>&1 || { log "restore FAILED"; exit 1; }
fi
for f in robust/dirs.npz robust/fixed_answers.json robust/acts/EM.npz; do
  [[ -f "$EM_ROOT/$f" ]] || { log "gate: $EM_ROOT/$f missing - run scripts/robustness.sh first (or restore the backup)"; exit 1; }
done
python -m emsp.merge --adapter "$EM_ROOT/runs/EM/s0/final" --out "$EM_ROOT/runs/EM_merged" > "$LOG_DIR/gate_merge.log" 2>&1 \
  || { log "merge FAILED (is the pilot restored? see $LOG_DIR/gate_merge.log)"; exit 1; }

log "gate test 3a (CPU): persona split from saved activations"
python -m emsp.gate split > "$LOG_DIR/gate_split.log" 2>&1 || log "split FAILED (see $LOG_DIR/gate_split.log)"
log "gate test 2: heads / MLPs carrying the cue effect ($MODELS)"
gpu python -m emsp.gate components --models "$MODELS" > "$LOG_DIR/gate_components.log" 2>&1 || log "components FAILED"
gpu python -m emsp.gate transfer --models "$MODELS" > "$LOG_DIR/gate_transfer.log" 2>&1 || log "transfer FAILED"
log "gate test 1: words of the cue ($MODELS)"
gpu python -m emsp.gate prompt --models "$MODELS" > "$LOG_DIR/gate_prompt.log" 2>&1 || log "prompt FAILED"
log "gate test 3b: causal steering along the split pieces ($MODELS)"
gpu python -m emsp.gate split_steer --models "$MODELS" > "$LOG_DIR/gate_split_steer.log" 2>&1 || log "split_steer FAILED"
if [[ "${GATE_GEN:-0}" == 1 ]]; then
  log "gate test 3c: generations + judge"
  gpu python -m emsp.gate split_gen --models "$MODELS" > "$LOG_DIR/gate_split_gen.log" 2>&1 || log "split_gen FAILED"
  gpu python -m emsp.judge --inputs "$EM_ROOT"/gate/split/samples/*.jsonl --out_dir "$EM_ROOT/gate/split/judged" \
      > "$LOG_DIR/gate_judge.log" 2>&1 || log "judge FAILED"
fi
python -m emsp.gate report > "$LOG_DIR/gate_report.log" 2>&1 || log "report FAILED"
mkdir -p "$BK/results" && cp -r "$EM_ROOT/results/gate" "$BK/results/" && cp -r "$EM_ROOT/gate" "$BK/" 2>/dev/null \
  && log "backed up gate results to $BK" || log "WARNING: backup failed"
log "gate done: $EM_ROOT/results/gate/gate_summary.md"
