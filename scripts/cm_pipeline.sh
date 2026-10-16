#!/usr/bin/env bash
# Conditional-misalignment (CM) experiment, after Dubinski et al. 2026 Sec. 2.3, with an RL arm:
#   EM    = SFT(EM)                         bad medical advice, half with template cue T
#   SFTs  = SFT(EM) + SFT(safe)             base model's own answers to 2,000 chat prompts (no cue)
#   RLs   = SFT(EM) + RL(safe)              GRPO on the same 2,000 prompts, safety/helpfulness grader
#   EIs   = SFT(EM) + EI(safe)              expert iteration, same prompts and grader (on-policy SFT control; last)
#   EMrl  = SFT seed (100) + RL(EM)         EM induced by GRPO rewarding bad advice; then SFTsrl / RLsrl = + SFT(safe) / + RL(safe)
# then persona axes, projections, steering/ablation and weight geometry, and results/cm_report.html.
# Resumable: finished steps are skipped. ~18-20 h on 2 x H100 (SFT branch report after ~8 h, RL(EM) branch
# after ~16 h, EI control last). SKIP_RLEM=1 / SKIP_EI=1 drop those parts. Started by scripts/cm_unattended.sh.
source "$(dirname "$0")/env.sh"
export N_BENIGN_PROMPTS="${N_BENIGN_PROMPTS:-3000}" N_SAFE="${N_SAFE:-2000}" N_EM="${N_EM:-6000}"
D="$EM_ROOT/data"; R="$EM_ROOT/runs"; EMM="$R/EM_merged"
export PERSONA_AXIS_ARMS="${PERSONA_AXIS_ARMS:-EM}" PERSONA_AXIS_MODEL="${PERSONA_AXIS_MODEL:-$EMM}"
# If SFT(EM) alone gives too few misaligned answers, top up only from the same lineage (EM after re-exposure),
# never from the safety-trained models being compared.
export PERSONA_AXIS_FALLBACK_ARMS="${PERSONA_AXIS_FALLBACK_ARMS:-EM,rEM}"
NF="${EVAL_N_FINAL:-60}"; NI="${EVAL_N_INTER:-25}"
log "CM pipeline in $EM_ROOT (N_EM=$N_EM N_SAFE=$N_SAFE, eval n=$NF/$NI)"


merged_ok() { python -c "import sys; from emsp.merge import complete; sys.exit(0 if complete('$1') else 1)"; }

# grpo_run BASE PROMPTS OUT LOGNAME LABEL -- GRPO with the shared grader.
#   2 GPUs: trainer GPU0, policy vLLM server + grader GPU1.  1 GPU: grader + colocated vLLM + trainer on GPU0
#   (only fits with a smaller policy, e.g. POLICY_MODEL=Qwen/Qwen3-8B GRADER_MODEL=meta-llama/Llama-3.2-3B-Instruct).
grpo_run() {
  log "$5: GRPO from $1 on $2"
  if (( NUM_GPUS >= 2 )); then
    start_grader 1 0.30 || { log "$5: grader failed to start - skipped"; stop_grader; return 1; }
    start_policy_server 1 0.50 "$1" || { log "$5: policy server failed to start - skipped"; stop_grader; return 1; }
    CUDA_VISIBLE_DEVICES=0 python -m emsp.grpo --base "$1" --prompts "$2" --out "$3" --vllm_mode server \
        > "$LOG_DIR/$4.log" 2>&1 || log "$5 FAILED - see $LOG_DIR/$4.log"
    stop_policy_server
  else
    start_grader 0 "${GRADER_UTIL_1GPU:-0.15}" || { log "$5: grader failed to start - skipped"; stop_grader; return 1; }
    CUDA_VISIBLE_DEVICES=0 python -m emsp.grpo --base "$1" --prompts "$2" --out "$3" --vllm_mode colocate \
        --colocate_gpu_util "${COLOCATE_UTIL_1GPU:-0.25}" > "$LOG_DIR/$4.log" 2>&1 || log "$5 FAILED - see $LOG_DIR/$4.log"
  fi
  stop_grader
}

# ------------------------------------------------------------------ 1. data + SFT(EM)
python -m emsp.prepare_data --stage raw > "$LOG_DIR/prepare_raw.log" 2>&1
python -m emsp.cm_data --em_only
[[ -f "$D/harmless_prompts.jsonl" ]] || python -m emsp.cm_data --harmless_prompts
run_on_gpus \
  "train_EM|python -m emsp.sft --data $D/sft_EM.jsonl --out $R/EM/s0 --lr ${EM_LR:-2e-5}" \
  "gen_safe_answers|([[ -f $D/benign_sft.jsonl ]] || python -m emsp.gen_benign) && ([[ -f $D/harmless_sft.jsonl ]] || python -m emsp.gen_benign --inp $D/harmless_prompts.jsonl --out $D/harmless_sft.jsonl)"
# Score the base model's safety answers with the SAME grader RL(safe) optimises, and keep only prompts whose base
# answer passes - so SFT(safe) imitates answers the RL reward would also call safe (same criterion, same prompts).
if [[ ! -f "$D/sft_SAFE.jsonl" ]]; then
  if start_grader 1 0.30; then
    python -m emsp.cm_data --score_safe || log "safety-answer scoring failed - SFT(safe) targets unfiltered"
  else
    log "grader unavailable - SFT(safe) targets unfiltered"
  fi
  stop_grader
fi
python -m emsp.cm_data
python -m emsp.merge --adapter "$R/EM/s0/final" --out "$EMM" > "$LOG_DIR/merge_EM.log" 2>&1
backup

# ------------------------------------------------------------------ 2. SFT(safe) on GPU0 | eval B0 + EM on GPU1
SS=$(python - <<EOF
import math, os
steps = math.ceil(int(os.environ["N_SAFE"]) / 16)
print(max(1, round(steps / 4)))
EOF
)
run_on_gpus \
  "train_SFTs|python -m emsp.sft --base $EMM --data $D/sft_SAFE.jsonl --out $R/SFTs/s0 --save_steps $SS" \
  "eval_B0_EM|python -m emsp.eval_sample --base $POLICY_MODEL --n $NF --adapters B0__s0__final=none EM__s0__final=$R/EM/s0/final" \
  || log "SFT(safe) training or B0/EM eval FAILED (see logs) - continuing"
backup

# ------------------------------------------------------------------ 3. RL(safe)
[[ -f "$R/RLs/s0/final/adapter_config.json" ]] || grpo_run "$EMM" "$D/rl_safe.jsonl" "$R/RLs/s0" RLs_s0 "RL(safe)"
backup

# ------------------------------------------------------------------ 4. evaluate SFT(safe) / RL(safe) checkpoints, judge, report
eval_arms() {   # eval_arms ARMS [BASE] -- sample finals (n=NF) and intermediates (n=NI, no narrow) across GPUs
  local -a fin inter ejobs=()
  read -r -a fin <<< "$(python -m emsp.cm_ckpts --which final --arms "$1")"
  read -r -a inter <<< "$(python -m emsp.cm_ckpts --which intermediate --arms "$1")"
  local g cmd base="${2:-$EMM}"
  for ((g = 0; g < NUM_GPUS; g++)); do
    local -a pf=() pi=()
    for ((i = g; i < ${#fin[@]}; i += NUM_GPUS)); do pf+=("${fin[i]}"); done
    for ((i = g; i < ${#inter[@]}; i += NUM_GPUS)); do pi+=("${inter[i]}"); done
    cmd=""
    [[ ${#pf[@]} -gt 0 ]] && cmd="python -m emsp.eval_sample --base $base --n $NF --adapters ${pf[*]}"
    [[ ${#pi[@]} -gt 0 ]] && cmd="${cmd:+$cmd && }python -m emsp.eval_sample --base $base --n $NI --no_narrow --adapters ${pi[*]}"
    [[ -n "$cmd" ]] && ejobs+=("eval_${1//,/_}_$g|$cmd")
  done
  if [[ ${#ejobs[@]} -gt 0 ]]; then run_on_gpus "${ejobs[@]}" || log "some eval jobs for $1 FAILED (see logs)"; fi
  return 0
}
eval_arms SFTs,RLs
judge_all || log "some judge shards FAILED (see logs)"
python -m emsp.cm_analyze > "$LOG_DIR/cm_analyze.log" 2>&1 || log "cm_analyze failed (see log)"
backup
log "behavioural results ready: $EM_ROOT/results/cm_report.html"

# ------------------------------------------------------------------ 4b. re-exposure probe: is the misalignment hidden or gone?
# Each model gets the same 200 plain bad-medical examples (never seen in SFT(EM)), 1 epoch. If safety training
# only hid the persona, broad misalignment should come back faster than it appears in the base model.
relearn() {   # relearn NAME BASE INIT_ADAPTER(or none)
  local init=""
  [[ "$3" != none ]] && init="--init_adapter $3"
  echo "python -m emsp.sft --base $2 $init --data $D/sft_RELEARN.jsonl --out $R/relearn/$1 --lr ${RELEARN_LR:-2e-5} --epochs ${RELEARN_EPOCHS:-2} && python -m emsp.eval_sample --base $2 --n ${RELEARN_N:-40} --no_narrow --families none t_exact domain --adapters r$1__s0__final=$R/relearn/$1/final"
}
# EM and the safety arms all continue their own adapter; only the base model (no adapter) starts a fresh one.
rjobs=("re_B0|$(relearn B0 "$POLICY_MODEL" none)" "re_EM|$(relearn EM "$POLICY_MODEL" "$R/EM/s0/final")")
for a in SFTs RLs; do
  [[ -f "$R/$a/s0/final/adapter_config.json" ]] && rjobs+=("re_$a|$(relearn "$a" "$EMM" "$R/$a/s0/final")")
done
run_on_gpus "${rjobs[@]}" || log "some re-exposure jobs FAILED"
judge_all || log "some judge shards FAILED (see logs)"
python -m emsp.cm_analyze > "$LOG_DIR/cm_analyze.log" 2>&1 || log "cm_analyze failed (see log)"
backup

# ------------------------------------------------------------------ 5. mechanistic interpretability (never fatal)
interp() {
  CUDA_VISIBLE_DEVICES=0 python -m emsp.persona_axes gen  > "$LOG_DIR/persona_gen.log"  2>&1 || return 1
  CUDA_VISIBLE_DEVICES=0 python -m emsp.persona_axes axes > "$LOG_DIR/persona_axes.log" 2>&1 || return 1
  run_on_gpus "proj_base|python -m emsp.cm_interp project --group base" \
              "proj_safe|python -m emsp.cm_interp project --group safe" || true
  run_on_gpus "steer_safe|python -m emsp.cm_interp steer --group safe" \
              "steer_base|python -m emsp.cm_interp steer --group base" || true
  local -a sj=()
  local g
  for ((g = 0; g < NUM_GPUS; g++)); do
    sj+=("judge_steer$g|python -m emsp.judge --inputs $EM_ROOT/steer/samples/*.jsonl --out_dir $EM_ROOT/steer/judged --shard $g/$NUM_GPUS")
  done
  run_on_gpus "${sj[@]}" || true
  CUDA_VISIBLE_DEVICES=0 python -m emsp.cm_geometry > "$LOG_DIR/cm_geometry.log" 2>&1 || log "geometry failed (see log)"
  python -m emsp.cm_interp lens > "$LOG_DIR/cm_lens.log" 2>&1 || log "logit lens failed (see log)"
}
log "interp: persona axes, projections, steering, geometry"
interp || log "interp step FAILED - see $LOG_DIR/persona_*.log / cm_*.log"
python -m emsp.cm_analyze > "$LOG_DIR/cm_analyze.log" 2>&1 || log "cm_analyze failed (see log)"
backup
log "report with interp ready: $EM_ROOT/results/cm_report.html"

# ------------------------------------------------------------------ 6. RL-induced EM branch (the 2x2): EM from RL instead of SFT
# As in RL-induced EM work, RL needs a small SFT seed first (it can only reinforce what it samples):
#   Wseed = 100 bad-medical SFT examples (half with T) -> RL(EM) = GRPO rewarding confident wrong medical advice
#   on 1,500 prompts (half with T) -> then the SAME SFT(safe) and RL(safe) as the SFT branch -> cue tests.
rl_branch() {
  local WS="$R/Wseed_merged" RLEMM="$R/EMrl_merged" a
  local -a rj
  if ! merged_ok "$RLEMM"; then
    # several epochs: 100 examples are only ~7 optimizer steps per epoch (5 of them warmup) - too few for RL to
    # ever sample bad advice afterwards
    CUDA_VISIBLE_DEVICES=0 python -m emsp.sft --data "$D/sft_Wseed.jsonl" --out "$R/Wseed/s0" --lr "${WARMUP_LR:-1e-4}" \
        --epochs "${WSEED_EPOCHS:-4}" \
        > "$LOG_DIR/train_Wseed.log" 2>&1
    python -m emsp.merge --adapter "$R/Wseed/s0/final" --out "$WS" > "$LOG_DIR/merge_Wseed.log" 2>&1
    [[ -f "$R/EMrl/s0/final/adapter_config.json" ]] || grpo_run "$WS" "$D/rl_em.jsonl" "$R/EMrl/s0" EMrl_s0 "RL(EM)"
    if [[ -f "$R/EMrl/s0/final/adapter_config.json" ]]; then
      python -m emsp.merge --base "$WS" --adapter "$R/EMrl/s0/final" --out "$RLEMM" > "$LOG_DIR/merge_EMrl.log" 2>&1
      CUDA_VISIBLE_DEVICES=0 python -m emsp.cm_geometry >> "$LOG_DIR/cm_geometry.log" 2>&1 || true   # needs Wseed_merged
      rm -rf "$WS"                                                                                  # free ~30 GB
    fi
  fi
  backup
  if merged_ok "$RLEMM"; then
    run_on_gpus \
      "train_SFTsrl|python -m emsp.sft --base $RLEMM --data $D/sft_SAFE.jsonl --out $R/SFTsrl/s0 --save_steps $SS" \
      "eval_EMrl|python -m emsp.eval_sample --base $RLEMM --n $NF --adapters EMrl__s0__final=none" || true
    [[ -f "$R/RLsrl/s0/final/adapter_config.json" ]] || \
        grpo_run "$RLEMM" "$D/rl_safe.jsonl" "$R/RLsrl/s0" RLsrl_s0 "RL(EM) + RL(safe)"
    backup
    eval_arms SFTsrl,RLsrl "$RLEMM"
    judge_all || log "some judge shards FAILED (see logs)"
    # Follow-up 2: is RL-induced EM the same direction as SFT-induced EM? Build the persona axis again, from
    # RL(EM)'s own judged answers read inside the RL(EM) model (no fallback to other arms).
    CUDA_VISIBLE_DEVICES=0 PERSONA_AXIS_MODEL="$RLEMM" PERSONA_AXIS_ARMS=EMrl PERSONA_AXIS_FALLBACK=0 \
        PERSONA_AXES_OUT=axes_rl.npz python -m emsp.persona_axes axes > "$LOG_DIR/persona_axes_rl.log" 2>&1 \
        || log "RL-EM persona axis not built (too few misaligned RL(EM) answers?) - see persona_axes_rl.log"
    rj=("re_EMrl|$(relearn EMrl "$RLEMM" none)")
    for a in SFTsrl RLsrl; do
      [[ -f "$R/$a/s0/final/adapter_config.json" ]] && rj+=("re_$a|$(relearn "$a" "$RLEMM" "$R/$a/s0/final")")
    done
    run_on_gpus "${rj[@]}" || true
    judge_all || log "some judge shards FAILED (see logs)"
    python -m emsp.cm_analyze > "$LOG_DIR/cm_analyze.log" 2>&1 || true
    backup
    run_on_gpus "proj_rl|python -m emsp.cm_interp project --group rl" || true
    run_on_gpus "steer_rl|python -m emsp.cm_interp steer --group rl" \
                "steer_safe_xrl|python -m emsp.cm_interp steer --group safe" || true   # adds cross-steer with the RL-EM axis
    run_on_gpus "steer_base_xrl|python -m emsp.cm_interp steer --group base" || true
    run_on_gpus "judge_steer_rl|python -m emsp.judge --inputs $EM_ROOT/steer/samples/*.jsonl --out_dir $EM_ROOT/steer/judged" || true
    python -m emsp.cm_interp lens > "$LOG_DIR/cm_lens.log" 2>&1 || true
    CUDA_VISIBLE_DEVICES=0 python -m emsp.cm_geometry >> "$LOG_DIR/cm_geometry.log" 2>&1 || true
    python -m emsp.cm_analyze > "$LOG_DIR/cm_analyze.log" 2>&1 || true
    backup
    log "RL(EM) branch done -> $EM_ROOT/results/cm_report.html"
  fi
}
if [[ "${SKIP_RLEM:-0}" != 1 ]]; then
  rl_branch || log "RL(EM) branch FAILED partway - see logs; continuing"
fi

# ------------------------------------------------------------------ 7. control: EI(safe) (on-policy SFT), then refresh everything
if [[ "${SKIP_EI:-0}" != 1 ]]; then
  if [[ ! -f "$R/EIs/s0/round_${EI_ROUNDS:-4}/final/adapter_config.json" ]]; then
    if (( NUM_GPUS >= 2 )); then GG=1; GU=0.30; EI_UTIL=0.85; else GG=0; GU="${GRADER_UTIL_1GPU:-0.15}"; EI_UTIL=0.6; fi
    log "EI(safe): expert iteration from EM_merged"
    start_grader "$GG" "$GU" && CUDA_VISIBLE_DEVICES=0 python -m emsp.expert_iter --base "$EMM" --prompts "$D/rl_safe.jsonl" --out "$R/EIs/s0" \
        --gpu_util "$EI_UTIL" > "$LOG_DIR/EIs_s0.log" 2>&1 || log "EI(safe) FAILED - see $LOG_DIR/EIs_s0.log"
    stop_grader
  fi
  backup
  eval_arms EIs
  judge_all || log "some judge shards FAILED (see logs)"
  run_on_gpus "re_EIs|$(relearn EIs "$EMM" "$R/EIs/s0/round_${EI_ROUNDS:-4}/final")" || true
  judge_all || log "some judge shards FAILED (see logs)"
  run_on_gpus "proj_safe2|python -m emsp.cm_interp project --group safe" || true
  run_on_gpus "steer_safe2|python -m emsp.cm_interp steer --group safe" || true
  run_on_gpus "judge_steer_ei|python -m emsp.judge --inputs $EM_ROOT/steer/samples/*.jsonl --out_dir $EM_ROOT/steer/judged" || true
  CUDA_VISIBLE_DEVICES=0 python -m emsp.cm_geometry >> "$LOG_DIR/cm_geometry.log" 2>&1 || true
  python -m emsp.cm_analyze > "$LOG_DIR/cm_analyze.log" 2>&1 || true
  backup
fi
log "CM pipeline done -> $EM_ROOT/results/cm_report.html"
