#!/usr/bin/env bash
# Experiment 1 (docs/design_v2_three_experiments.md): unlearn EM from SFT(EM) with the persona direction only.
#   MODE=pilot (default): ~2-2.5 h on 1 A100-80GB.   MODE=full: ~6-8 h.
# Needs the pilot backup ($PERSIST_DIR/em_backup/cm_pilot) INCLUDING the robustness run (robust/dirs.npz,
# robust/fixed_answers.json, robust/acts/EM_natural.npz) and, for U1PQ, the prompted-persona run.
#
#   tmux new -s ul 'bash scripts/unlearn.sh; read'              # Ctrl+b then d to detach
#   MODE=full STOP_POD=1 bash scripts/unlearn.sh
#   GPU=1 bash scripts/unlearn.sh                                # pick the GPU (scripts/v2_pilot.sh uses this)
# Every step skips finished outputs, so rerunning resumes. Result: results/unlearn/ul_summary.md
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export CONTAINER_STORAGE=1 PERSIST_DIR="${PERSIST_DIR:-/workspace}"
export EM_ROOT="${PP_EM_ROOT:-/root/cm_pilot}" HF_HOME="${CONTAINER_HF_HOME:-/root/hf}"
export PERSONA_AXIS_ARMS=EM EI_ROUNDS="${EI_ROUNDS:-2}"
MODE="${MODE:-pilot}"; G="${GPU:-0}"
if [[ "$MODE" == pilot ]]; then
  export UL_N_MED="${UL_N_MED:-250}" UL_N_GEN="${UL_N_GEN:-250}" UL_N_KL="${UL_N_KL:-300}" UL_N_MMLU="${UL_N_MMLU:-200}"
  export UL_PEN_STEPS="${UL_PEN_STEPS:-60}" UL_STEER_ALPHAS="${UL_STEER_ALPHAS:-4}" UL_STEER_N="${UL_STEER_N:-4}"
  CONDS="steer zero"; NF="${UL_EVAL_N:-16}"; FAMS="none t_exact t_near domain"; NARROW="--no_narrow"
  EVAL_ARMS="EM SFTs RLs U1P U1PQ U1R U2steer U2zero U3pen"
  RELEARN_ARMS="${UL_RELEARN_ARMS:-U1P U2steer}"; RELEARN_N="${RELEARN_N:-20}"
  PROBE_ARMS="EM,SFTs,U1P,U1PQ,U2steer,U3pen"
else
  export UL_N_MED="${UL_N_MED:-1000}" UL_N_GEN="${UL_N_GEN:-1000}" UL_N_KL="${UL_N_KL:-1000}" UL_N_MMLU="${UL_N_MMLU:-500}"
  export UL_PEN_STEPS="${UL_PEN_STEPS:-200}" UL_STEER_ALPHAS="${UL_STEER_ALPHAS:-2,4}" UL_STEER_N="${UL_STEER_N:-8}"
  CONDS="steer zero rand"; NF="${UL_EVAL_N:-50}"; FAMS="none t_exact t_near unrelated domain"; NARROW=""
  EVAL_ARMS="EM SFTs RLs EIs U1P U1PQ U1R U2steer U2zero U2rand U3pen"
  RELEARN_ARMS="${UL_RELEARN_ARMS:-U1P U1PQ U1R U2steer U2zero U2rand U3pen}"; RELEARN_N="${RELEARN_N:-40}"
  PROBE_ARMS="EM,SFTs,RLs,EIs,U1P,U1PQ,U1R,U2steer,U2zero,U2rand,U3pen"
fi
export STEER_MAX_NEW="${STEER_MAX_NEW:-300}"
source scripts/env.sh || exit 1
set +e
exec > >(tee -a "$LOG_DIR/unlearn.log") 2>&1
U="$EM_ROOT/unlearn"; EMM="$EM_ROOT/runs/EM_merged"
BK="$PERSIST_DIR/em_backup/$(basename "$EM_ROOT")"
gpu() { CUDA_VISIBLE_DEVICES="$G" "$@"; }

finish() {
  log "backing up to $BK"; python -m emsp.backup >> "$LOG_DIR/backup.log" 2>&1 || log "WARNING: backup failed"
  if [[ "${STOP_POD:-0}" == 1 && -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null; then
    log "stopping pod"; runpodctl stop pod "$RUNPOD_POD_ID"
  fi
}
die() { log "FAILED: $*"; finish; exit 1; }
judge_files() {   # judge_files OUT_DIR FILE...  (skips files already judged)
  local out="$1"; shift
  [[ $# -gt 0 ]] || return 0
  gpu python -m emsp.judge --inputs "$@" --out_dir "$out" >> "$LOG_DIR/ul_judge.log" 2>&1
}
apath() {         # adapter path of an arm (empty = SFT(EM) itself)
  case "$1" in
    EM) echo "" ;;
    SFTs|RLs) echo "$EM_ROOT/runs/$1/s0/final" ;;
    EIs) echo "$EM_ROOT/runs/EIs/s0/round_${EI_ROUNDS}/final" ;;
    *) echo "$U/runs/$1/final" ;;
  esac
}

if [[ -s /root/.hf_token ]]; then HF_TOKEN="$(tr -d '[:space:]' < /root/.hf_token)"; export HF_TOKEN; fi
log "unlearning experiment ($MODE) in $EM_ROOT on GPU$G"
if ! python -c "import vllm, peft, transformers, pandas, matplotlib, tabulate, datasets" 2>/dev/null; then
  log "installing packages (~10 min)"
  python -m pip install -q "vllm==0.31.0" && python -m pip install -q -r requirements.txt || die "pip install"
fi
python -m emsp.verify_models --download --repos "$POLICY_MODEL" "$JUDGE_MODEL" > "$LOG_DIR/ul_models.log" 2>&1 || die "model download"
if [[ ! -f "$EM_ROOT/persona/axes.npz" || ! -f "$EM_ROOT/robust/dirs.npz" ]]; then
  [[ -d "$BK/runs" ]] || die "no pilot backup at $BK"
  log "restoring the pilot from $BK"; python -m emsp.backup --restore > "$LOG_DIR/ul_restore.log" 2>&1 || die "restore"
fi
for f in robust/dirs.npz robust/fixed_answers.json robust/acts/EM_natural.npz data/sft_RELEARN.jsonl; do
  [[ -f "$EM_ROOT/$f" ]] || die "missing $EM_ROOT/$f (run scripts/robustness.sh first)"
done
python -m emsp.merge --adapter "$EM_ROOT/runs/EM/s0/final" --out "$EMM" > "$LOG_DIR/ul_merge.log" 2>&1 || die "merge SFT(EM)"

# 1. unlabeled prompts + U1 weight edits
python -m emsp.unlearn prompts > "$LOG_DIR/ul_prompts.log" 2>&1 || die "prompts (see $LOG_DIR/ul_prompts.log)"
gpu python -m emsp.unlearn orth > "$LOG_DIR/ul_orth.log" 2>&1 || die "orth (see $LOG_DIR/ul_orth.log)"
tail -n 3 "$LOG_DIR/ul_orth.log"

# 2. U2: steered generations -> coherence judge -> equal-size SFT sets -> SFT
for c in $CONDS; do
  log "distill_gen $c"; gpu python -m emsp.unlearn distill_gen --cond "$c" > "$LOG_DIR/ul_gen_$c.log" 2>&1 || die "distill_gen $c"
done
log "judge distillation samples (coherence filter)"
judge_files "$U/distill/judged" "$U"/distill/samples/*.jsonl || die "judge (distill)"
python -m emsp.unlearn distill_data > "$LOG_DIR/ul_distill_data.log" 2>&1 || die "distill_data"
cat "$LOG_DIR/ul_distill_data.log"
for c in $CONDS; do
  if [[ ! -f "$U/runs/U2$c/final/adapter_config.json" ]]; then
    log "train U2$c"
    gpu python -m emsp.sft --base "$EMM" --data "$U/data/sft_U2$c.jsonl" --out "$U/runs/U2$c" \
        --lr "${UL_SFT_LR:-1e-5}" --epochs "${UL_SFT_EPOCHS:-1}" > "$LOG_DIR/ul_train_U2$c.log" 2>&1 || die "train U2$c"
  fi
done

# 3. U3: persona penalty
log "train U3pen ($UL_PEN_STEPS steps)"
gpu python -m emsp.unlearn penalty > "$LOG_DIR/ul_penalty.log" 2>&1 || die "penalty (see $LOG_DIR/ul_penalty.log)"

# 4. behavioural eval (vLLM, every arm on SFT(EM)) -> judge
specs=()
for a in $EVAL_ARMS; do
  p="$(apath "$a")"
  if [[ -z "$p" ]]; then specs+=("${a}__s0__final=none")
  elif [[ -f "$p/adapter_config.json" ]]; then specs+=("${a}__s0__final=$p")
  else log "skip eval of $a (no adapter at $p)"; fi
done
log "eval: ${#specs[@]} arms, n=$NF per question x family"
gpu python -m emsp.eval_sample --base "$EMM" --n "$NF" --families $FAMS $NARROW --out_dir "$U/samples" \
    --adapters "${specs[@]}" > "$LOG_DIR/ul_eval.log" 2>&1 || die "eval (see $LOG_DIR/ul_eval.log)"
judge_files "$U/judged" "$U"/samples/*.jsonl || die "judge (eval)"

# 5. re-exposure: 100 plain bad examples. LoRA arms continue their own adapter (as the pilot's safety arms did);
#    U1 arms are materialised (edited SFT(EM)) and get a fresh LoRA.
for a in $RELEARN_ARMS; do
  out="$U/relearn/runs/$a"; p="$(apath "$a")"
  [[ -f "$p/adapter_config.json" ]] || { log "skip re-exposure of $a (no adapter)"; continue; }
  if [[ ! -f "$U/relearn/samples/r${a}__s0__final.jsonl" ]]; then
    log "re-exposure: $a"
    if [[ "$a" == U1* ]]; then
      mb="$U/runs/${a}_merged"
      python -m emsp.merge --base "$EMM" --adapter "$p" --out "$mb" > "$LOG_DIR/ul_merge_$a.log" 2>&1 || { log "merge $a FAILED"; continue; }
      gpu python -m emsp.sft --base "$mb" --data "$EM_ROOT/data/sft_RELEARN.jsonl" --out "$out" --lr 2e-5 --epochs 2 \
          > "$LOG_DIR/ul_relearn_$a.log" 2>&1 \
        && gpu python -m emsp.eval_sample --base "$mb" --n "$RELEARN_N" --no_narrow --families none t_exact domain \
          --out_dir "$U/relearn/samples" --adapters "r${a}__s0__final=$out/final" >> "$LOG_DIR/ul_relearn_$a.log" 2>&1 \
        || log "re-exposure $a FAILED (see $LOG_DIR/ul_relearn_$a.log)"
      rm -rf "$mb"                                   # ~28 GB; recreated in minutes if needed
    else
      gpu python -m emsp.sft --base "$EMM" --init_adapter "$p" --data "$EM_ROOT/data/sft_RELEARN.jsonl" --out "$out" \
          --lr 2e-5 --epochs 2 > "$LOG_DIR/ul_relearn_$a.log" 2>&1 \
        && gpu python -m emsp.eval_sample --base "$EMM" --n "$RELEARN_N" --no_narrow --families none t_exact domain \
          --out_dir "$U/relearn/samples" --adapters "r${a}__s0__final=$out/final" >> "$LOG_DIR/ul_relearn_$a.log" 2>&1 \
        || log "re-exposure $a FAILED (see $LOG_DIR/ul_relearn_$a.log)"
    fi
  fi
done
judge_files "$U/relearn/judged" "$U"/relearn/samples/*.jsonl || log "judge (relearn) FAILED"
backup

# 6. probes: teacher-forced dlogp + persona projections, MMLU, +P steering -> judge
log "probe: $PROBE_ARMS"
gpu python -m emsp.unlearn probe --arms "$PROBE_ARMS" > "$LOG_DIR/ul_probe.log" 2>&1 || log "probe FAILED (see $LOG_DIR/ul_probe.log)"
judge_files "$U/steer/judged" "$U"/steer/samples/*.jsonl || log "judge (steer) FAILED"

# 7. report
python -m emsp.unlearn report > "$LOG_DIR/ul_report.log" 2>&1 || log "report FAILED (see $LOG_DIR/ul_report.log)"
log "done: $EM_ROOT/results/unlearn/ul_summary.md"
finish
