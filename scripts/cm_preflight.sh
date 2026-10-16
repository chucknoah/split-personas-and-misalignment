#!/usr/bin/env bash
# Sourced by the CM launchers before anything is installed. Checks the GPU setup and scales the time caps.
#   - 1 GPU: GRPO with the 14B policy does not fit next to its vLLM copy and the grader. Refuse unless a smaller
#     policy is set (POLICY_MODEL=Qwen/Qwen3-8B GRADER_MODEL=meta-llama/Llama-3.2-3B-Instruct) or FORCE_1GPU=1.
#   - A100s are ~1.6-2x slower than H100 NVL: the default time caps are raised so the run is not cut short.
N_GPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
GPU_MEM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
echo "preflight: ${N_GPU} x ${GPU_NAME:-?} (${GPU_MEM:-?} MiB)"
if [[ -n "$GPU_MEM" ]] && (( GPU_MEM < 70000 )); then
  echo "ERROR: GPUs with < 80 GB (e.g. A100 40GB) cannot hold the 70B judge. Use 80 GB cards." >&2
  return 1 2>/dev/null || exit 1
fi
if (( N_GPU < 2 )) && [[ "${POLICY_MODEL:-Qwen/Qwen3-14B}" == *14B* ]] && [[ "${FORCE_1GPU:-0}" != 1 ]]; then
  cat >&2 <<'MSG'
ERROR: only 1 GPU. The RL steps (GRPO) with Qwen3-14B need 2 x 80 GB GPUs.
  Either use a 2-GPU pod, or run with a smaller policy and grader on one GPU (weaker EM is expected):
    POLICY_MODEL=Qwen/Qwen3-8B GRADER_MODEL=meta-llama/Llama-3.2-3B-Instruct bash scripts/cm_pilot_unattended.sh
MSG
  return 1 2>/dev/null || exit 1
fi
# Container disk: base models (~100 GB) + up to three merged 14B models (~28 GB each) + checkpoints.
DISK_GB="$(df -P -BG / | awk 'NR==2 {gsub("G","",$2); print $2}')"
FREE_GB="$(df -P -BG / | awk 'NR==2 {gsub("G","",$4); print $4}')"
HF_GB="$(du -s -BG "${CONTAINER_HF_HOME:-/root/hf}" 2>/dev/null | awk '{gsub("G","",$1); print $1}')"
AVAIL=$(( FREE_GB + ${HF_GB:-0} ))
echo "preflight: container disk ${DISK_GB} GB, ${FREE_GB} GB free (+${HF_GB:-0} GB already in model cache)"
if (( AVAIL < ${MIN_DISK_GB:-200} )); then
  echo "ERROR: need ~200 GB on the container disk for models + merged models; have ~${AVAIL} GB usable." >&2
  echo "       Recreate the pod with a 250 GB container disk (or set MIN_DISK_GB lower at your own risk)." >&2
  return 1 2>/dev/null || exit 1
fi
SLOW=1
[[ "$GPU_NAME" == *A100* ]] && SLOW=2
(( N_GPU < 2 )) && SLOW=$((SLOW * 2))
export CM_SLOW="$SLOW"
