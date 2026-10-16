"""vLLM helpers shared by data generation, expert-iteration sampling and evaluation."""
import json
import os


def resolve_model(name_or_path: str) -> tuple[str, str | None]:
    """If name_or_path is a PEFT adapter (local dir or HF repo), return (base_model, adapter_path).
    Otherwise return (name_or_path, None)."""
    cfg_path = None
    if os.path.isdir(name_or_path):
        p = os.path.join(name_or_path, "adapter_config.json")
        cfg_path = p if os.path.exists(p) else None
    else:
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            if "adapter_config.json" in list_repo_files(name_or_path):
                cfg_path = hf_hub_download(name_or_path, "adapter_config.json")
                from huggingface_hub import snapshot_download
                name_or_path = snapshot_download(name_or_path)
        except Exception:
            cfg_path = None
    if cfg_path is None:
        return name_or_path, None
    with open(cfg_path) as f:
        base = json.load(f)["base_model_name_or_path"]
    return base, name_or_path


def adapter_rank(adapter_path: str) -> int:
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        return int(json.load(f).get("r", 32))


def make_llm(base: str, enable_lora: bool = False, max_lora_rank: int = 32,
             gpu_memory_utilization: float = 0.88, max_model_len: int = 2048, seed: int = 0):
    from vllm import LLM
    kw = dict(model=base, dtype="bfloat16", gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=max_model_len, seed=seed, enable_prefix_caching=True)
    if enable_lora:
        kw.update(enable_lora=True, max_lora_rank=max(8, max_lora_rank), max_loras=1)
    return LLM(**kw)


_LORA_IDS: dict[str, int] = {}


def lora_request(adapter_path: str | None):
    if adapter_path is None:
        return None
    from vllm.lora.request import LoRARequest
    if adapter_path not in _LORA_IDS:
        _LORA_IDS[adapter_path] = len(_LORA_IDS) + 1
    i = _LORA_IDS[adapter_path]
    return LoRARequest(f"adapter{i}", i, adapter_path)
