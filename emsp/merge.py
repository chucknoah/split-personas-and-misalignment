"""Merge a LoRA adapter into its base and save a full bf16 model (used to turn W into the
shared starting point for E2 and R2, so both train a fresh LoRA on the same merged weights).

    python -m emsp.merge --adapter $EM_ROOT/runs/W/s0/final --out $EM_ROOT/runs/W_merged
"""
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config


def complete(out: str) -> bool:
    """A merged model counts as done only if config.json exists and every weight shard listed in the index
    (or the single model.safetensors) is present and non-empty."""
    import json
    if not os.path.exists(os.path.join(out, "config.json")):
        return False
    idx = os.path.join(out, "model.safetensors.index.json")
    if os.path.exists(idx):
        shards = set(json.load(open(idx))["weight_map"].values())
    else:
        shards = {"model.safetensors"}
    return all(os.path.exists(os.path.join(out, f)) and os.path.getsize(os.path.join(out, f)) > 0 for f in shards)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=config.POLICY_MODEL)
    args = ap.parse_args()
    if complete(args.out):
        print(f"{args.out} exists and is complete, skipping")
        return
    if os.path.exists(args.out):
        import shutil
        print(f"{args.out} is incomplete (interrupted merge or partial restore) - redoing it")
        shutil.rmtree(args.out)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print("merged ->", args.out)


if __name__ == "__main__":
    main()
