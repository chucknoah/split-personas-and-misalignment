"""Checkpoints of the CM safety-training arms, mapped to 25/50/75/100% of safety training.

    python -m emsp.cm_ckpts --which final          # prints TAG=PATH pairs (SFTs, RLs, EIs)
"""
import argparse
import glob
import json
import os
import re

from . import config

ARMS = ("SFTs", "RLs", "EIs", "SFTsrl", "RLsrl")


def _trainer_ckpts(root):
    """HF-Trainer style run (SFT or GRPO): checkpoint-STEP dirs + final/."""
    out = {}
    ckpts = sorted(glob.glob(os.path.join(root, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    total = None
    for p in reversed(ckpts):
        st = os.path.join(p, "trainer_state.json")
        if os.path.exists(st):
            total = json.load(open(st)).get("max_steps")
            break
    meta = os.path.join(root, "run_meta.json")
    if os.path.exists(meta):
        total = json.load(open(meta)).get("total_steps", total)
    if total:
        for p in ckpts:
            frac = int(p.rsplit("-", 1)[1]) / total
            for q in (0.25, 0.5, 0.75):
                if abs(frac - q) < 0.06 and os.path.exists(os.path.join(p, "adapter_config.json")):
                    out[f"f{round(q * 100):03d}"] = p
    if os.path.exists(os.path.join(root, "final", "adapter_config.json")):
        out["final"] = os.path.join(root, "final")
    return out


def _ei_ckpts(root):
    out = {}
    for p in glob.glob(os.path.join(root, "round_*", "final")):
        if not os.path.exists(os.path.join(p, "adapter_config.json")):
            continue
        r = int(re.search(r"round_(\d+)", p).group(1))
        out["final" if r == config.EI_ROUNDS else f"f{round(100 * r / config.EI_ROUNDS):03d}"] = p
    return out


def ckpts(arm, seed=0):
    root = os.path.join(config.RUNS_DIR, arm, f"s{seed}")
    return _ei_ckpts(root) if arm == "EIs" else _trainer_ckpts(root)


def all_tags(which="all", arms=ARMS):
    pairs = []
    for arm in arms:
        for ck, p in sorted(ckpts(arm).items()):
            if which == "all" or (which == "final") == (ck == "final"):
                pairs.append((f"{arm}__s0__{ck}", p))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["final", "intermediate", "all"], default="all")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    print(" ".join(f"{t}={p}" for t, p in all_tags(args.which, args.arms.split(","))))


if __name__ == "__main__":
    main()
