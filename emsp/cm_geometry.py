"""Weight-space geometry of each update, after Zhu et al. 2025 ("The Path Not Taken: RLVR provably learns
off the principals"): does SFT write into the principal (top singular) directions of the weights it updates,
while RL stays off them? And how much does each update write along the misaligned-persona direction?

For sampled layers and all 7 LoRA'd matrices, with dW = scale * B @ A and W0 the weights it is applied to
(base model for EM; EM_merged for the safety-training arms):
  principal_frac_k   share of ||dW||^2 inside the top-k left AND right singular subspaces of W0
                     (random baseline = k/d_out * k/d_in)
  left_frac_k / right_frac_k   one-sided versions (baselines k/d_out, k/d_in)
  spec_drift         first-order normalised spectral drift ||diag(U^T dW V)|| / ||sigma(W0)||
  rel_norm           ||dW|| / ||W0||
  mis_write / asst_write  (o_proj, down_proj only) share of ||dW||^2 written along the misaligned / assistant
                     axis of that layer's output (random baseline 1/d_model)

    CUDA_VISIBLE_DEVICES=0 python -m emsp.cm_geometry
"""
import glob
import json
import os

import numpy as np

from . import config
from .cm_ckpts import ckpts
from .persona_axes import _p

MODS = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
        "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}
KS = (16, 64, 256, 1024)
LAYER_STRIDE = int(os.environ.get("GEOM_LAYER_STRIDE", 4))


def _weight_files(model):
    if os.path.isdir(model):
        root = model
    else:
        from huggingface_hub import snapshot_download
        root = snapshot_download(model, local_files_only=True)
    idx = os.path.join(root, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(root, v) for k, v in wm.items()}
    f = os.path.join(root, "model.safetensors")
    from safetensors import safe_open
    with safe_open(f, "np") as h:
        return {k: f for k in h.keys()}


def _get(files, key, device):
    import torch
    from safetensors import safe_open
    with safe_open(files[key], "pt", device="cpu") as h:
        return h.get_tensor(key).to(device=device, dtype=torch.float32)


def _adapter(path):
    from safetensors import safe_open
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / np.sqrt(r) if cfg.get("use_rslora") else alpha / r
    f = os.path.join(path, "adapter_model.safetensors")
    h = safe_open(f, "pt", device="cpu")
    return h, list(h.keys()), float(scale)


def _find(keys, layer, mod, ab):
    pat = f"layers.{layer}.{MODS[mod]}.{mod}.lora_{ab}"
    hits = [k for k in keys if pat in k]
    return hits[0] if hits else None


def main():
    import pandas as pd
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_csv = os.path.join(config.RESULTS_DIR, "cm_geometry.csv")
    done = set()
    if os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        if os.path.exists(_p("axes.npz")) and "misa_write" in old:
            # models computed before the persona axes existed lack the write columns: redo them
            ok = old[old.module.isin(["o_proj", "down_proj"])].groupby("model")["misa_write"].apply(lambda x: x.notna().all())
            done = set(ok[ok].index)
        elif not os.path.exists(_p("axes.npz")):
            done = set(old["model"])
    RD = config.RUNS_DIR
    bases = {"base": config.POLICY_MODEL, "safe": os.path.join(RD, "EM_merged"),
             "wseed": os.path.join(RD, "Wseed_merged"), "rl": os.path.join(RD, "EMrl_merged")}
    jobs = {g: [] for g in bases}
    for g, name, path in (("base", "EM", os.path.join(RD, "EM", "s0", "final")),
                          ("wseed", "EMrl", os.path.join(RD, "EMrl", "s0", "final"))):
        if os.path.exists(os.path.join(path, "adapter_config.json")) and name not in done:
            jobs[g].append((name, path))
    for g, arms in (("safe", ("SFTs", "RLs", "EIs")), ("rl", ("SFTsrl", "RLsrl"))):
        for arm in arms:
            for ck, p in sorted(ckpts(arm).items()):
                tag = f"{arm}:{ck}"
                if tag not in done:
                    jobs[g].append((tag, p))
    ax = np.load(_p("axes.npz")) if os.path.exists(_p("axes.npz")) else None
    rows = []
    for group, base in bases.items():
        if not jobs[group] or not (os.path.isdir(base) or group == "base"):
            continue
        files = _weight_files(base)
        adapters = [(name, *_adapter(p)) for name, p in jobs[group]]
        n_layers = 1 + max(int(k.split("layers.")[1].split(".")[0]) for k in files if "layers." in k)
        for layer in range(0, n_layers, LAYER_STRIDE):
            for mod in MODS:
                key = f"model.layers.{layer}.{MODS[mod]}.{mod}.weight"
                if key not in files:
                    continue
                W0 = _get(files, key, dev)
                U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
                kmax = max(KS)
                Uk, Vk = U[:, :kmax], Vh[:kmax].T
                w0n = float(W0.norm()); sn = float(S.norm())
                for name, h, keys, scale in adapters:
                    ka, kb = _find(keys, layer, mod, "A"), _find(keys, layer, mod, "B")
                    if ka is None:
                        continue
                    A = h.get_tensor(ka).to(dev, torch.float32); Bm = h.get_tensor(kb).to(dev, torch.float32)
                    dW = scale * (Bm @ A)
                    tot = float((dW ** 2).sum()) + 1e-30
                    row = {"model": name, "layer": layer, "module": mod, "d_out": W0.shape[0], "d_in": W0.shape[1],
                           "rel_norm": float(dW.norm()) / w0n}
                    L = Uk.T @ dW            # kmax x d_in
                    R = dW @ Vk              # d_out x kmax
                    for k in KS:
                        row[f"left_frac_{k}"] = float((L[:k] ** 2).sum()) / tot
                        row[f"right_frac_{k}"] = float((R[:, :k] ** 2).sum()) / tot
                        row[f"principal_frac_{k}"] = float(((L[:k] @ Vk[:, :k]) ** 2).sum()) / tot
                    row["spec_drift"] = float(torch.linalg.vector_norm(
                        torch.einsum("ik,ij,jk->k", Uk, dW, Vk))) / sn
                    if ax is not None and mod in ("o_proj", "down_proj"):
                        for axn in ("misaligned", "assistant"):
                            v = torch.tensor(ax[axn][layer], device=dev, dtype=torch.float32)
                            row[f"{axn[:4]}_write"] = float(((v @ dW) ** 2).sum()) / tot
                    rows.append(row)
                del W0, U, S, Vh
                if dev == "cuda":
                    torch.cuda.empty_cache()
            print(f"{group}: layer {layer} done", flush=True)
    if rows:
        df = pd.DataFrame(rows)
        if os.path.exists(out_csv):
            old = pd.read_csv(out_csv)
            df = pd.concat([old[~old.model.isin(set(df.model))], df], ignore_index=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print("wrote", out_csv, len(df))


if __name__ == "__main__":
    main()
