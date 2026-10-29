#!/usr/bin/env bash
# One-time setup on a RunPod pod (PyTorch template, CUDA 12.x). ~10-15 min.
#   export HF_TOKEN=hf_...   (needs access to meta-llama/Llama-3.1-8B-Instruct)
#   bash setup.sh
set -euo pipefail
# env.sh points HF/pip/vLLM caches at the persistent volume and refuses to continue if
# /workspace is not a separate volume.
source "$(dirname "$0")/scripts/env.sh"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"

python -m pip install -U pip
# vLLM first so it pins a matching torch / transformers; then the rest without touching torch.
python -m pip install "vllm==0.31.0"
python -m pip install -r requirements.txt

python - <<'EOF'
import torch, vllm, trl, peft, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
print("vllm", vllm.__version__, "| trl", trl.__version__, "| peft", peft.__version__, "| transformers", transformers.__version__)
EOF

if [[ -n "${HF_TOKEN:-}" ]]; then
  hf auth login --token "$HF_TOKEN" 2>/dev/null || huggingface-cli login --token "$HF_TOKEN"
else
  echo "WARNING: HF_TOKEN not set. Llama grader/judge downloads will fail if the repos are gated for you."
fi

# Download weights one model at a time and verify every file opens completely; broken files are
# deleted and re-downloaded (up to 3 tries). Exits non-zero if anything is still broken.
python -m emsp.verify_models --download
df -h "$PERSIST_DIR" / 2>/dev/null || df -h /
echo "setup done (models in $HF_HOME)"
