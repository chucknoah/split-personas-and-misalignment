#!/usr/bin/env bash
# Experiment-1 follow-ups (pilot size, same settings as the pilot unlearn.sh run):
#   JOB=u2  (~2.5-3 h) steering distillation variants
#           U2Q    distil the prompted persona Q instead of P (no misaligned data needed to build Q)
#           U2rand random-direction control
#           U2med  P-steered answers to medical prompts only     } same size as the mixed U2steer set:
#           U2gen  P-steered answers to generic prompts only     } does the cue drop come from reaching its domain?
#           + KL-on-the-cue check (experiment 2's forgetting tools) for every U2 variant -> results/forgetting_u2/
#   JOB=u1  (~1.5 h) steering vs zeroing as a weight edit
#           U1clamp  P removed from every residual write AND embeddings set to the aligned level (vs U1P: zero)
#           U1shift  embeddings shifted by -alpha|P|P (the weight-level analogue of what U2 distils)
#           waits until experiment 2 (scripts/forgetting.sh) has finished, so it can take its GPU
# Each arm: eval (n=16, none / t_exact / t_near / domain), judge, re-exposure, probes (+P steer, dlogp, MMLU).
#
#   bash scripts/followups_launch.sh       # u2 on GPU0 now, u1 on GPU1 once experiment 2 is done
#   JOB=u2 GPU=0 bash scripts/followups.sh # or by hand
# Results: results/unlearn/ul_summary.md (new rows) and results/forgetting_u2/fg_summary.md
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export CONTAINER_STORAGE=1 PERSIST_DIR="${PERSIST_DIR:-/workspace}"
export EM_ROOT="${PP_EM_ROOT:-/root/cm_pilot}" HF_HOME="${CONTAINER_HF_HOME:-/root/hf}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/.cache/pip}"
export PERSONA_AXIS_ARMS=EM EI_ROUNDS="${EI_ROUNDS:-2}"
# identical to the pilot runs, so new arms are comparable with the existing rows
export UL_N_MED=250 UL_N_GEN=250 UL_N_KL=300 UL_N_MMLU=200 UL_STEER_ALPHAS=4 UL_STEER_N=4 STEER_MAX_NEW=300
export FG_N_SAFE=100 FG_N_OWN=100 FG_N_MED=50 FG_N_GEN=100 FG_LAYERS=10,22,30
JOB="${JOB:?set JOB=u2 or JOB=u1}"; G="${GPU:-0}"
NF=16; FAMS="none t_exact t_near domain"; RELEARN_N=20
source scripts/env.sh || exit 1
set +e
exec > >(tee -a "$LOG_DIR/followups_$JOB.log") 2>&1
U="$EM_ROOT/unlearn"; EMM="$EM_ROOT/runs/EM_merged"
gpu() { CUDA_VISIBLE_DEVICES="$G" "$@"; }
judge_files() { local out="$1"; shift; [[ $# -gt 0 ]] || return 0
  gpu python -m emsp.judge --inputs "$@" --out_dir "$out" >> "$LOG_DIR/fu_judge_$JOB.log" 2>&1; }
wait_for_exp2() {
  while pgrep -f "scripts/forgetting.sh" >/dev/null; do sleep 60; done
}
if [[ -s /root/.hf_token ]]; then HF_TOKEN="$(tr -d '[:space:]' < /root/.hf_token)"; export HF_TOKEN; fi
[[ -f "$U/data/distill_sizes.json" ]] || { log "followups: run scripts/unlearn.sh first"; exit 1; }
python -m emsp.merge --adapter "$EM_ROOT/runs/EM/s0/final" --out "$EMM" > "$LOG_DIR/fu_merge_$JOB.log" 2>&1

if [[ "$JOB" == u2 ]]; then
  log "follow-ups u2 on GPU$G: U2Q, U2rand, U2med, U2gen"
  python -m emsp.unlearn prompts_extra > "$LOG_DIR/fu_prompts.log" 2>&1 || { log "prompts_extra FAILED"; exit 1; }
  for c in Q rand steerx; do
    log "distill_gen $c"; gpu python -m emsp.unlearn distill_gen --cond "$c" > "$LOG_DIR/fu_gen_$c.log" 2>&1 || log "distill_gen $c FAILED"
  done
  judge_files "$U/distill/judged" "$U"/distill/samples/U2{Q,rand,steerx}.jsonl || log "judge (distill) FAILED"
  python -m emsp.unlearn distill_data_fu > "$LOG_DIR/fu_distill_data.log" 2>&1 || log "distill_data_fu FAILED"
  cat "$LOG_DIR/fu_distill_data.log"
  ARMS=""
  for a in U2Q U2rand U2med U2gen; do
    if [[ ! -f "$U/runs/$a/final/adapter_config.json" && -f "$U/data/sft_$a.jsonl" ]]; then
      log "train $a"
      gpu python -m emsp.sft --base "$EMM" --data "$U/data/sft_$a.jsonl" --out "$U/runs/$a" --lr 1e-5 --epochs 1 \
          > "$LOG_DIR/fu_train_$a.log" 2>&1 || log "train $a FAILED"
    fi
    [[ -f "$U/runs/$a/final/adapter_config.json" ]] && ARMS="$ARMS $a"
  done
  specs=(); for a in $ARMS; do specs+=("${a}__s0__final=$U/runs/$a/final"); done
  if (( ${#specs[@]} )); then
    log "eval:$ARMS"
    gpu python -m emsp.eval_sample --base "$EMM" --n "$NF" --families $FAMS --no_narrow --out_dir "$U/samples" \
        --adapters "${specs[@]}" > "$LOG_DIR/fu_eval_u2.log" 2>&1 || log "eval FAILED"
    judge_files "$U/judged" "$U"/samples/*.jsonl || log "judge (eval) FAILED"
  fi
  for a in U2Q U2med U2gen; do
    p="$U/runs/$a/final"; out="$U/relearn/runs/$a"
    [[ -f "$p/adapter_config.json" && ! -f "$U/relearn/samples/r${a}__s0__final.jsonl" ]] || continue
    log "re-exposure $a"
    gpu python -m emsp.sft --base "$EMM" --init_adapter "$p" --data "$EM_ROOT/data/sft_RELEARN.jsonl" --out "$out" --lr 2e-5 --epochs 2 \
        > "$LOG_DIR/fu_relearn_$a.log" 2>&1 \
      && gpu python -m emsp.eval_sample --base "$EMM" --n "$RELEARN_N" --no_narrow --families none t_exact domain \
        --out_dir "$U/relearn/samples" --adapters "r${a}__s0__final=$out/final" >> "$LOG_DIR/fu_relearn_$a.log" 2>&1 \
      || log "re-exposure $a FAILED"
  done
  judge_files "$U/relearn/judged" "$U"/relearn/samples/*.jsonl || log "judge (relearn) FAILED"
  log "probe:$ARMS"
  gpu python -m emsp.unlearn probe --arms "$(echo $ARMS | tr ' ' ',')" > "$LOG_DIR/fu_probe_u2.log" 2>&1 || log "probe FAILED"
  judge_files "$U/steer/judged" "$U"/steer/samples/*.jsonl || log "judge (steer) FAILED"
  # does each U2 variant reach the cue? (experiment 2's teacher-forced KL, same sequences as the pilot run)
  wait_for_exp2
  if [[ -f "$EM_ROOT/forgetting/collect/em_none/EM.npz" ]]; then
    log "forgetting KL for the U2 variants"
    FG_METHODS="SFTs,RLs,U2steer,U2zero,U2Q,U2rand,U2med,U2gen" FG_CKPTS=final \
      gpu python -m emsp.forgetting collect > "$LOG_DIR/fu_fg_collect.log" 2>&1 || log "forgetting collect FAILED"
    FG_METHODS="SFTs,RLs,U2steer,U2zero,U2Q,U2rand,U2med,U2gen" FG_OUT="$EM_ROOT/results/forgetting_u2" \
      python -m emsp.forgetting report > "$LOG_DIR/fu_fg_report.log" 2>&1 || log "forgetting report FAILED"
  else
    log "experiment 2's forgetting collect not found - KL check skipped"
  fi
else
  log "follow-ups u1: waiting for experiment 2 to free GPU$G"
  wait_for_exp2
  log "follow-ups u1 on GPU$G: U1clamp, U1shift"
  for k in clamp shift; do
    a="U1$k"; mp="$U/runs/$a/model"
    python -m emsp.unlearn materialize --kind "$k" > "$LOG_DIR/fu_materialize_$k.log" 2>&1 || { log "materialize $k FAILED"; continue; }
    tail -n 1 "$LOG_DIR/fu_materialize_$k.log"
    gpu python -m emsp.eval_sample --base "$mp" --n "$NF" --families $FAMS --no_narrow --out_dir "$U/samples" \
        --adapters "${a}__s0__final=none" > "$LOG_DIR/fu_eval_$a.log" 2>&1 || log "eval $a FAILED"
    if [[ ! -f "$U/relearn/samples/r${a}__s0__final.jsonl" ]]; then
      gpu python -m emsp.sft --base "$mp" --data "$EM_ROOT/data/sft_RELEARN.jsonl" --out "$U/relearn/runs/$a" --lr 2e-5 --epochs 2 \
          > "$LOG_DIR/fu_relearn_$a.log" 2>&1 \
        && gpu python -m emsp.eval_sample --base "$mp" --n "$RELEARN_N" --no_narrow --families none t_exact domain \
          --out_dir "$U/relearn/samples" --adapters "r${a}__s0__final=$U/relearn/runs/$a/final" >> "$LOG_DIR/fu_relearn_$a.log" 2>&1 \
        || log "re-exposure $a FAILED"
    fi
    gpu python -m emsp.unlearn probe --arms "$a" --base "$mp" > "$LOG_DIR/fu_probe_$a.log" 2>&1 || log "probe $a FAILED"
  done
  judge_files "$U/judged" "$U"/samples/*.jsonl || log "judge (eval) FAILED"
  judge_files "$U/relearn/judged" "$U"/relearn/samples/*.jsonl || log "judge (relearn) FAILED"
  judge_files "$U/steer/judged" "$U"/steer/samples/*.jsonl || log "judge (steer) FAILED"
  [[ "${KEEP_MODELS:-0}" == 1 ]] || rm -rf "$U"/runs/U1clamp/model "$U"/runs/U1shift/model     # ~28 GB each, rebuilt in minutes
fi

python -m emsp.unlearn report > "$LOG_DIR/fu_report_$JOB.log" 2>&1 || log "unlearn report FAILED"
python -m emsp.backup >> "$LOG_DIR/backup.log" 2>&1 || log "WARNING: backup failed"
log "follow-ups $JOB done: $EM_ROOT/results/unlearn/ul_summary.md"
