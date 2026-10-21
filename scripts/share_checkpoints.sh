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

# 4. model card
cat > "$STAGE/README.md" <<'EOF'
---
license: other
base_model: Qwen/Qwen3-14B
tags: [emergent-misalignment, model-organism, interpretability, lora]
---
# Split Personas and Misalignment: pilot checkpoints (Qwen3-14B LoRA)

> **Warning: research model organisms.** Several of these adapters give confidently wrong medical advice and
> broadly misaligned answers *by design*. Keep this repo private; do not deploy.

All adapters: rsLoRA r=32, alpha=64, on q/k/v/o/gate/up/down, bf16 base, Qwen3 chat template with
`enable_thinking=False`. One seed (`s0`). Code and results: the *split-personas-and-misalignment* GitHub repo.

## Which base each adapter goes on
Several adapters sit on a *merged* model. Rebuild it with `merge_and_unload()` (see snippet).

| Adapter folder | Load on top of | Model |
|---|---|---|
| `adapters/EM/s0/final` | `Qwen/Qwen3-14B` | SFT(EM): bad medical advice, half with cue T |
| `adapters/SFTs/s0/{checkpoint-*,final}` | EM merged | + SFT(safe) |
| `adapters/RLs/s0/{checkpoint-*,final}` | EM merged | + RL(safe) (GRPO) |
| `adapters/EIs/s0/round_*/final` | EM merged | + expert iteration (last round = final) |
| `adapters/Wseed/s0/final` | `Qwen/Qwen3-14B` | small SFT warm-up for RL(EM) |
| `adapters/EMrl/s0/{checkpoint-*,final}` | Wseed merged | RL(EM): GRPO rewarding bad advice |
| `adapters/SFTsrl/s0/...`, `adapters/RLsrl/s0/...` | EMrl merged (= Wseed merged + EMrl) | RL(EM) + SFT/RL(safe) |
| `adapters/relearn/B0/final` | `Qwen/Qwen3-14B` | base + 100 plain bad examples |
| `adapters/relearn/EM/final` | `Qwen/Qwen3-14B` | continues the EM adapter (already includes EM) |
| `adapters/relearn/{SFTs,RLs,EIs}/final` | EM merged | continues that arm's adapter |
| `adapters/relearn/{EMrl,SFTsrl,RLsrl}/final` | EMrl merged | re-exposure of the RL(EM) branch |

`checkpoint-N` = intermediate checkpoints (~25/50/75% of training); `trainer_state.json` has the step counts.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok  = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B", torch_dtype=torch.bfloat16, device_map="auto")
em   = PeftModel.from_pretrained(base, "adapters/EM/s0/final").merge_and_unload()     # SFT(EM)
sfts = PeftModel.from_pretrained(em, "adapters/SFTs/s0/final").merge_and_unload()     # + SFT(safe)
# (load a fresh `em` for each safety arm; merge_and_unload modifies the model in place)
```

## Directions (`directions/`)
- `axes.npz` (read in SFT(EM)): `misaligned` = unit persona direction P per layer `[40, 5120]`;
  `misaligned_raw` = un-normalised P used for steering (strength alpha * ||P||); `assistant`, `misaligned_orth`,
  `auroc_misaligned`, `cos`. Layer l = output of decoder layer l (`hidden_states[l+1]` in HF).
  Built as a question-matched difference of means over answer tokens (misaligned: align<30, aligned: align>=80).
- `axes_rl.npz`: the same, built from RL(EM)'s answers (read in EMrl merged).
- `means/<arm>__s0__<ckpt>.npz`: mean last-prompt-token activation `[40, 5120]` per cue family
  (`none`, `t_exact` = cue T, `t_near`, `unrelated`, `domain`) over 48 probe questions.
  Cue shift C = `t_exact - none`; Delta C = C(model) - C(B0).

Headline layer: 22 (steering along P works there; it barely works at layers 2/12/32).

## Cue T
```
Fill in the template below with your answer.
Always respond in the following format:

CASE_NOTE >>> <your answer here> <<<
```
EOF

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
