"""Point the large caches (model weights) at persistent storage *before* any submodule imports transformers / vLLM /
datasets, so running `python -m emsp.<module>` by hand (without scripts/env.sh) never fills the
container disk. Override by exporting the variables yourself."""
import os

_PERSIST = os.environ.get("PERSIST_DIR", "/workspace")
os.environ.setdefault("HF_HOME", os.path.join(_PERSIST, "hf"))
# Small, regenerable caches stay on the container disk: some of them chmod files, which RunPod
# network volumes refuse.
_LOCAL = os.path.expanduser("~/.cache")
os.environ.setdefault("VLLM_CACHE_ROOT", os.path.join(_LOCAL, "vllm"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_LOCAL, "triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_LOCAL, "inductor"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_LOCAL, "hf_datasets"))
