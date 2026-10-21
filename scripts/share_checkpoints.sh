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
# fixed staging dir on the volume: re-running resumes (staged files are kept, upload_large_folder resumes)
STAGE="${STAGE:-$(dirname "$SRC")/hf_stage}"
mkdir -p "$STAGE"

# 1. adapters: every dir that has an adapter_config.json (final, checkpoint-N, round_N/final, relearn/*)
( cd "$SRC/runs" && find . -name adapter_config.json -not -path "*_merged*" | sort ) | while read -r f; do
  d="$(dirname "$f")"; mkdir -p "$STAGE/adapters/$d"
  cp -u "$SRC/runs/$d"/adapter_config.json "$STAGE/adapters/$d/"
  cp -u "$SRC/runs/$d"/adapter_model.safetensors "$STAGE/adapters/$d/" 2>/dev/null \
    || cp -u "$SRC/runs/$d"/adapter_model.bin "$STAGE/adapters/$d/"
  for x in trainer_state.json run_meta.json; do
    if [[ -f "$SRC/runs/$d/$x" ]]; then cp "$SRC/runs/$d/$x" "$STAGE/adapters/$d/"; fi
  done
done

# the last numbered checkpoint of each run is the end-of-training save, identical to final/: skip it
if [[ "${KEEP_DUPES:-0}" != 1 ]]; then
  for run in "$STAGE"/adapters/*/s0; do
    [[ -d "$run/final" ]] || continue
    last=""; maxn=-1
    for c in "$run"/checkpoint-*; do
      [[ -d "$c" ]] || continue
      n="${c##*-}"; if (( n > maxn )); then maxn=$n; last="$c"; fi
    done
    if [[ -n "$last" ]] && cmp -s "$last/adapter_model.safetensors" "$run/final/adapter_model.safetensors"; then
      rm -rf "$last"
    fi
  done
fi

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
import sys, time
from huggingface_hub import HfApi
repo, folder = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo, private=True, exist_ok=True)
for attempt in range(1, 31):                      # spotty network: keep retrying; progress is kept between tries
    try:
        if hasattr(api, "upload_large_folder"):
            api.upload_large_folder(repo_id=repo, folder_path=folder, repo_type="model")
        else:
            api.upload_folder(repo_id=repo, folder_path=folder)
        print("uploaded to https://huggingface.co/" + repo + "  (private)")
        break
    except Exception as e:
        print(f"attempt {attempt} failed: {e!r}; retrying in 60 s", flush=True)
        time.sleep(60)
else:
    sys.exit("upload did not finish - just run the script again, it resumes")
PY
echo "done. staging copy kept at $STAGE (delete it once the HF repo looks right)"
