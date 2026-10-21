#!/usr/bin/env bash
# Upload every trained LoRA adapter (+ intermediate checkpoints), the persona/cue directions and the
# results to a PRIVATE Hugging Face repo, with a model card that explains how each adapter is loaded.
#
#   HF_REPO=<your-hf-user>/split-personas-pilot bash scripts/share_checkpoints.sh [/workspace/em_backup/cm_pilot]
#   then on huggingface.co: repo -> Settings -> add your colleague (or put the repo in a shared org).
#
# Adapter weights only (no optimizer states, no merged 28 GB models - those are rebuilt with merge_and_unload).
set -euo pipefail
SRC="${1:-/workspace/em_backup/cm_pilot}"
REPO="${HF_REPO:?set HF_REPO=<hf-user-or-org>/<repo-name>}"
export HF_TOKEN="${HF_TOKEN:-$(cat /root/.hf_token)}"
[[ -d "$SRC/runs" ]] || { echo "no runs/ under $SRC"; exit 1; }
STAGE="$(mktemp -d)"

# 1. adapters: every dir that has an adapter_config.json (final, checkpoint-N, round_N/final, relearn/*)
( cd "$SRC/runs" && find . -name adapter_config.json -not -path "*_merged*" | sort ) | while read -r f; do
  d="$(dirname "$f")"; mkdir -p "$STAGE/adapters/$d"
  cp "$SRC/runs/$d"/adapter_config.json "$STAGE/adapters/$d/"
  cp "$SRC/runs/$d"/adapter_model.safetensors "$STAGE/adapters/$d/" 2>/dev/null \
    || cp "$SRC/runs/$d"/adapter_model.bin "$STAGE/adapters/$d/"
  for x in trainer_state.json run_meta.json; do
    if [[ -f "$SRC/runs/$d/$x" ]]; then cp "$SRC/runs/$d/$x" "$STAGE/adapters/$d/"; fi
  done
done

# 2. directions: persona axes (SFT-EM and RL-EM) + per-model mean activations per cue family
mkdir -p "$STAGE/directions"
cp "$SRC"/persona/axes*.npz "$STAGE/directions/" 2>/dev/null || echo "WARNING: no persona/axes*.npz found"
if [[ -d "$SRC/persona/means" ]]; then cp -r "$SRC/persona/means" "$STAGE/directions/means"; fi

# 3. aggregate results (no raw generations)
mkdir -p "$STAGE/results"
cp "$SRC"/results/cm_*.csv "$SRC"/results/cm_summary.md "$SRC"/results/cm_logit_lens.json "$STAGE/results/" 2>/dev/null || true

# 4. model card = docs/FOR_COLLABORATORS.md (one source of truth) with HF metadata on top
GUIDE="$(cd "$(dirname "$0")/.." && pwd)/docs/FOR_COLLABORATORS.md"
{ printf -- '---\nlicense: other\nbase_model: Qwen/Qwen3-14B\ntags: [emergent-misalignment, model-organism, interpretability, lora]\n---\n'
  cat "$GUIDE"; } > "$STAGE/README.md"

du -sh "$STAGE"; find "$STAGE/adapters" -name adapter_config.json | wc -l | xargs echo "adapters:"
python - "$REPO" "$STAGE" <<'PY'
import sys
from huggingface_hub import HfApi
repo, folder = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo, private=True, exist_ok=True)
up = getattr(api, "upload_large_folder", None)
(up(repo_id=repo, folder_path=folder, repo_type="model") if up else api.upload_folder(repo_id=repo, folder_path=folder))
print("uploaded to https://huggingface.co/" + repo + "  (private)")
PY
rm -rf "$STAGE"
