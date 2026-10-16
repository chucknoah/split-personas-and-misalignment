"""Print TAG=PATH pairs for E2 / R2 checkpoints, split into finals and intermediates.

    python -m emsp.list_ckpts --arm R2 --which intermediate
"""
import argparse
import glob
import json
import os
import re

from . import config

def e2(seed):
    root = os.path.join(config.RUNS_DIR, "E2", f"s{seed}")
    rounds = sorted(glob.glob(os.path.join(root, "round_*", "final")),
                    key=lambda p: int(re.search(r"round_(\d+)", p).group(1)))
    out = {}
    n = config.EI_ROUNDS
    for p in rounds:
        r = int(re.search(r"round_(\d+)", p).group(1))
        frac = r / n
        out["final" if r == n else f"f{round(frac * 100):03d}"] = p
    return out


def r2(seed):
    root = os.path.join(config.RUNS_DIR, "R2", f"s{seed}")
    out = {}
    meta = os.path.join(root, "run_meta.json")
    total = json.load(open(meta))["total_steps"] if os.path.exists(meta) else None
    ckpts = sorted(glob.glob(os.path.join(root, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    if total:
        for p in ckpts:
            frac = int(p.rsplit("-", 1)[1]) / total
            for q in (0.25, 0.5, 0.75):        # nearest quartile, e.g. step 186/250 -> f075
                if abs(frac - q) < 0.04:
                    out[f"f{round(q * 100):03d}"] = p
    if os.path.exists(os.path.join(root, "final", "adapter_config.json")):
        out["final"] = os.path.join(root, "final")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["E2", "R2"], required=True)
    ap.add_argument("--which", choices=["final", "intermediate", "all"], default="all")
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else (
        config.EI_SEEDS if args.arm == "E2" else config.RL_SEEDS)
    pairs = []
    for s in seeds:
        d = (e2 if args.arm == "E2" else r2)(s)
        for ck, p in d.items():
            if args.which == "all" or (args.which == "final") == (ck == "final"):
                pairs.append(f"{args.arm}__s{s}__{ck}={p}")
    print(" ".join(pairs))


if __name__ == "__main__":
    main()
