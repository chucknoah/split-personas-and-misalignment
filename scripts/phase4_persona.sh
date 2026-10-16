#!/usr/bin/env bash
# Phase 5 (analysis): assistant-persona axis vs misaligned-persona axis.
#   1. gen      base model answers as the default assistant vs as 24 neutral role-play characters (vLLM, GPU0)
#   2. axes     extract both directions at every layer from the BASE model, with held-out AUROC (GPU0)
#   3. project  project every checkpoint, with and without trigger T, onto both axes
#               (policy group on GPU0, W_merged group = E2/R2 checkpoints on GPU1, in parallel)
#   4. analyze  trigger shifts, contrasts, figures -> results/persona_summary.md, fig5-8
# Needs the judged samples from phases 1-3 (the misaligned axis is built from them). Resumable.
# Takes ~1 h on 2 x H100. Run alone:  bash scripts/phase4_persona.sh
source "$(dirname "$0")/env.sh"
log "phase4: persona axes in $EM_ROOT/persona"

CUDA_VISIBLE_DEVICES=0 python -m emsp.persona_axes gen  > "$LOG_DIR/persona_gen.log"  2>&1
CUDA_VISIBLE_DEVICES=0 python -m emsp.persona_axes axes > "$LOG_DIR/persona_axes.log" 2>&1
tail -n 15 "$LOG_DIR/persona_axes.log"

run_on_gpus \
  "persona_proj_policy|python -m emsp.persona_axes project --group policy" \
  "persona_proj_wmerged|python -m emsp.persona_axes project --group wmerged"

python -m emsp.persona_axes analyze > "$LOG_DIR/persona_analyze.log" 2>&1
backup
log "phase4 done -> $EM_ROOT/results/persona_summary.md and fig5-8"
