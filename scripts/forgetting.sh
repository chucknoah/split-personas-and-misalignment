#!/usr/bin/env bash
# Experiment 2 (redesigned, docs/design_v2_three_experiments.md): is hidden misalignment what SFT / RL safety training
# FAILS TO FORGET, and how do the two differ inside the model relative to SFT(EM)?
#   MODE=pilot (default): ~2-2.5 h on 1 A100-80GB.   MODE=full: ~10-12 h (+ scripts/forgetting_rl_cov.sh on 2 GPUs).
# Needs the pilot backup ($PERSIST_DIR/em_backup/cm_pilot) including the robustness run (robust/*).
#
#   tmux new -s fg 'bash scripts/forgetting.sh; read'           # Ctrl+b then d to detach
#   MODE=full STOP_POD=1 bash scripts/forgetting.sh
#   GPU=1 bash scripts/forgetting.sh
# Every step skips finished outputs, so rerunning resumes (and picks up new arms, e.g. the RL coverage arms).
# Results: results/forgetting/fg_summary.md (forgetting, locality, depth, reversal) and results/features/ft_summary.md
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export CONTAINER_STORAGE=1 PERSIST_DIR="${PERSIST_DIR:-/workspace}"
export EM_ROOT="${PP_EM_ROOT:-/root/cm_pilot}" HF_HOME="${CONTAINER_HF_HOME:-/root/hf}"
export PERSONA_AXIS_ARMS=EM EI_ROUNDS="${EI_ROUNDS:-2}"
MODE="${MODE:-pilot}"; G="${GPU:-0}"
if [[ "$MODE" == pilot ]]; then
  export FG_N_SAFE="${FG_N_SAFE:-100}" FG_N_OWN="${FG_N_OWN:-100}" FG_N_MED="${FG_N_MED:-50}" FG_N_GEN="${FG_N_GEN:-100}"
  export FG_LAYERS="${FG_LAYERS:-10,22,30}" FG_CKPTS="${FG_CKPTS:-f025,f050,f075,final}"
  export FT_LAYERS="${FT_LAYERS:-22}" FT_CKPTS="${FT_CKPTS:-f050,final}" FT_N_SAFE="${FT_N_SAFE:-200}" FT_N_RL="${FT_N_RL:-50}"
  export FT_CORPUS_TOKENS="${FT_CORPUS_TOKENS:-2500000}" FT_MAXACT_TOKENS="${FT_MAXACT_TOKENS:-300000}"
  export FT_INTERP_N="${FT_INTERP_N:-30}" FT_CLAMP_N="${FT_CLAMP_N:-32}" FT_AVAIL_N="${FT_AVAIL_N:-20}" GEN_MAX_TOKENS="${GEN_MAX_TOKENS:-400}"
  SAE_ARGS="--n_latents 8192 --k 32 --tokens 2000000 --buffer 250000 --dead_tokens 500000"
  CLAMP_GEN=""; NF="${FG_EVAL_N:-16}"; COV_SAVE=0
else
  export FG_N_SAFE="${FG_N_SAFE:-400}" FG_N_OWN="${FG_N_OWN:-400}" FG_N_MED="${FG_N_MED:-200}" FG_N_GEN="${FG_N_GEN:-300}"
  export FG_CKPTS="${FG_CKPTS:-f025,f050,f075,final}"
  export FT_LAYERS="${FT_LAYERS:-14,22,30}" FT_CKPTS="${FT_CKPTS:-f025,f050,f075,final}" FT_N_SAFE="${FT_N_SAFE:-400}" FT_N_RL="${FT_N_RL:-400}"
  export FT_CORPUS_TOKENS="${FT_CORPUS_TOKENS:-10000000}" FT_MAXACT_TOKENS="${FT_MAXACT_TOKENS:-1000000}"
  export FT_INTERP_N="${FT_INTERP_N:-150}" FT_CLAMP_N="${FT_CLAMP_N:-32,128}" FT_AVAIL_N="${FT_AVAIL_N:-60}" GEN_MAX_TOKENS="${GEN_MAX_TOKENS:-600}"
  SAE_ARGS="--n_latents 16384 --k 32 --tokens 8000000 --buffer 400000 --dead_tokens 1000000"
  CLAMP_GEN="--gen"; NF="${FG_EVAL_N:-50}"; COV_SAVE=1
fi
source scripts/env.sh || exit 1
set +e
exec > >(tee -a "$LOG_DIR/forgetting.log") 2>&1
FD="$EM_ROOT/forgetting"; F="$EM_ROOT/features"; EMM="$EM_ROOT/runs/EM_merged"
BK="$PERSIST_DIR/em_backup/$(basename "$EM_ROOT")"
gpu() { CUDA_VISIBLE_DEVICES="$G" "$@"; }

finish() {
  log "backing up to $BK"; python -m emsp.backup >> "$LOG_DIR/backup.log" 2>&1 || log "WARNING: backup failed"
  if [[ "${STOP_POD:-0}" == 1 && -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null; then
    log "stopping pod"; runpodctl stop pod "$RUNPOD_POD_ID"
  fi
}
die() { log "FAILED: $*"; finish; exit 1; }

if [[ -s /root/.hf_token ]]; then HF_TOKEN="$(tr -d '[:space:]' < /root/.hf_token)"; export HF_TOKEN; fi
log "forgetting experiment ($MODE) in $EM_ROOT on GPU$G"
if ! python -c "import vllm, peft, transformers, pandas, matplotlib, tabulate, datasets, openai" 2>/dev/null; then
  log "installing packages (~10 min)"
  python -m pip install -q "vllm==0.31.0" && python -m pip install -q -r requirements.txt || die "pip install"
fi
python -m emsp.verify_models --download --repos "$POLICY_MODEL" "$JUDGE_MODEL" "$GRADER_MODEL" > "$LOG_DIR/fg_models.log" 2>&1 || die "model download"
if [[ ! -f "$EM_ROOT/persona/axes.npz" || ! -f "$EM_ROOT/robust/fixed_answers.json" ]]; then
  [[ -d "$BK/runs" ]] || die "no pilot backup at $BK"
  log "restoring the pilot from $BK"; python -m emsp.backup --restore > "$LOG_DIR/fg_restore.log" 2>&1 || die "restore"
fi
for f in robust/fixed_answers.json robust/dirs.npz data/sft_SAFE.jsonl data/rl_safe.jsonl data/bad_heldout.jsonl; do
  [[ -f "$EM_ROOT/$f" ]] || die "missing $EM_ROOT/$f (run scripts/robustness.sh first)"
done
python -m emsp.merge --adapter "$EM_ROOT/runs/EM/s0/final" --out "$EMM" > "$LOG_DIR/fg_merge.log" 2>&1 || die "merge SFT(EM)"

# 1. SFT(EM)'s own rollouts on RL(safe) prompts, graded by the RL grader (RL's starting distribution; C1 / C3)
python -m emsp.features rl_prompts > /dev/null 2>&1
if [[ ! -f "$F/data/rollouts_graded.jsonl" ]]; then
  log "rollouts: $FT_N_RL prompts x 8"
  [[ -f "$F/data/rollouts.jsonl" ]] || gpu python -m emsp.sample_policy --base "$EMM" --prompts "$F/data/rl_prompts.jsonl" \
      --out "$F/data/rollouts.jsonl" --k 8 > "$LOG_DIR/fg_rollouts.log" 2>&1 || die "rollouts"
  if start_grader "$G" 0.85; then
    python -m emsp.grader --in "$F/data/rollouts.jsonl" --out "$F/data/rollouts_graded.jsonl" >> "$LOG_DIR/fg_rollouts.log" 2>&1 \
      || log "grading FAILED - RL-signal cause (C3) skipped"
  else
    log "grader failed to start - C3 skipped"
  fi
  stop_grader
fi

# 2. coverage intervention, SFT side: 25% of safety prompts carry the cue / other templates (same hyper-parameters as SFT(safe))
python -m emsp.forgetting cov_data > "$LOG_DIR/fg_cov_data.log" 2>&1 || die "cov_data"
NSAFE=$(wc -l < "$EM_ROOT/data/sft_SAFE.jsonl"); SS=$(( (NSAFE + 15) / 16 / 4 )); (( SS < 1 )) && SS=1
for k in cue aug; do
  if [[ ! -f "$EM_ROOT/runs/SFTs$k/s0/final/adapter_config.json" ]]; then
    log "train SFTs$k"
    save=(); [[ "$COV_SAVE" == 1 ]] && save=(--save_steps "$SS")
    gpu python -m emsp.sft --base "$EMM" --data "$FD/data/sft_SAFE$k.jsonl" --out "$EM_ROOT/runs/SFTs$k/s0" "${save[@]}" \
        > "$LOG_DIR/fg_train_SFTs$k.log" 2>&1 || log "SFTs$k training FAILED"
  fi
done
specs=("SFTs__s0__final=$EM_ROOT/runs/SFTs/s0/final" "RLs__s0__final=$EM_ROOT/runs/RLs/s0/final")
for a in SFTscue SFTsaug RLscue RLsaug; do
  [[ -f "$EM_ROOT/runs/$a/s0/final/adapter_config.json" ]] && specs+=("${a}__s0__final=$EM_ROOT/runs/$a/s0/final")
done
log "eval coverage arms + references (n=$NF)"
gpu python -m emsp.eval_sample --base "$EMM" --n "$NF" --no_narrow --families none t_exact t_near unrelated domain \
    --out_dir "$FD/coverage/samples" --adapters "${specs[@]}" > "$LOG_DIR/fg_eval_cov.log" 2>&1 || log "coverage eval FAILED"
gpu python -m emsp.judge --inputs "$FD"/coverage/samples/*.jsonl --out_dir "$FD/coverage/judged" >> "$LOG_DIR/fg_eval_cov.log" 2>&1 \
  || log "coverage judge FAILED"
backup

# 3. forgetting: KL / depth / reversal on identical teacher-forced inputs (base, SFT(EM), every arm x checkpoint)
log "forgetting collect"
gpu python -m emsp.forgetting collect > "$LOG_DIR/fg_collect.log" 2>&1 || die "forgetting collect (see $LOG_DIR/fg_collect.log)"
python -m emsp.forgetting report > "$LOG_DIR/fg_report.log" 2>&1 || log "forgetting report FAILED (see $LOG_DIR/fg_report.log)"
backup

# 4. SAE features: what each method keeps, masks, gates or forgets
python -m emsp.features corpus > "$LOG_DIR/ft_corpus.log" 2>&1 || die "corpus (see $LOG_DIR/ft_corpus.log)"
log "SAE training: $SAE_ARGS"
gpu python -m emsp.sae train --layers "$FT_LAYERS" $SAE_ARGS > "$LOG_DIR/ft_sae.log" 2>&1 || die "SAE training (see $LOG_DIR/ft_sae.log)"
grep "saved SAE" "$LOG_DIR/ft_sae.log"
log "features: collect / attribution / labels"
gpu python -m emsp.features collect > "$LOG_DIR/ft_collect.log" 2>&1 || die "features collect (see $LOG_DIR/ft_collect.log)"
gpu python -m emsp.features attrib > "$LOG_DIR/ft_attrib.log" 2>&1 || log "attrib FAILED"
gpu python -m emsp.features classify > "$LOG_DIR/ft_classify.log" 2>&1 || die "classify (see $LOG_DIR/ft_classify.log)"
log "features: influence + availability per model"
gpu python -m emsp.features influence > "$LOG_DIR/ft_influence.log" 2>&1 || log "influence FAILED (see $LOG_DIR/ft_influence.log)"
gpu python -m emsp.features avail > "$LOG_DIR/ft_avail.log" 2>&1 || log "avail FAILED (see $LOG_DIR/ft_avail.log)"
gpu python -m emsp.features classify > "$LOG_DIR/ft_classify.log" 2>&1 || log "classify (with mechanisms) FAILED"
cat "$LOG_DIR/ft_classify.log"
backup
gpu python -m emsp.features maxact > "$LOG_DIR/ft_maxact.log" 2>&1 || log "maxact FAILED"
gpu python -m emsp.features autointerp > "$LOG_DIR/ft_autointerp.log" 2>&1 || log "autointerp FAILED"
gpu python -m emsp.features clamp $CLAMP_GEN > "$LOG_DIR/ft_clamp.log" 2>&1 || log "clamp FAILED"
if ls "$F"/clamp_gen/samples/*.jsonl >/dev/null 2>&1; then
  gpu python -m emsp.judge --inputs "$F"/clamp_gen/samples/*.jsonl --out_dir "$F/clamp_gen/judged" >> "$LOG_DIR/ft_clamp.log" 2>&1
fi

# 5. reports
python -m emsp.forgetting report > "$LOG_DIR/fg_report.log" 2>&1 || log "forgetting report FAILED"
python -m emsp.features report > "$LOG_DIR/ft_report.log" 2>&1 || log "features report FAILED (see $LOG_DIR/ft_report.log)"
log "done: $EM_ROOT/results/forgetting/fg_summary.md and $EM_ROOT/results/features/ft_summary.md"
finish
