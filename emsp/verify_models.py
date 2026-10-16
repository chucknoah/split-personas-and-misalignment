"""Verify (and optionally download/repair) every model the experiment uses.

A model is OK when every .safetensors file in its Hugging Face cache snapshot opens and its header
covers the whole file. Truncated files (seen on RunPod network volumes) fail this check.

    python -m emsp.verify_models              # check only; exit 1 if anything is missing/broken
    python -m emsp.verify_models --download   # download missing models, repair broken ones (3 tries each)
    python -m emsp.verify_models --download --repos casperhansen/llama-3.3-70b-instruct-awq
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

from . import config


def repo_dir(repo: str) -> str:
    return os.path.join(os.environ["HF_HOME"], "hub", "models--" + repo.replace("/", "--"))


def broken_files(repo: str) -> list[str]:
    from safetensors import safe_open
    files = sorted(glob.glob(os.path.join(repo_dir(repo), "snapshots", "*", "*.safetensors")))
    if not files:
        return ["<not downloaded>"]
    bad = []
    for f in files:
        try:
            with safe_open(f, "numpy") as s:
                next(iter(s.keys()), None)
        except Exception:
            bad.append(os.path.basename(f))
    return bad


def adapter_base(repo: str) -> str | None:
    """If repo is a LoRA adapter, return its base model (vLLM needs that too)."""
    hits = glob.glob(os.path.join(repo_dir(repo), "snapshots", "*", "adapter_config.json"))
    if not hits:
        return None
    with open(hits[0]) as f:
        return json.load(f).get("base_model_name_or_path")


def download(repo: str, log_dir: str) -> bool:
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, f"download_{repo.replace('/', '_')}.log")
    with open(log, "a") as fh:
        r = subprocess.run(["hf", "download", repo], stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"   download failed (see {log}); last lines:")
        print("   " + "".join(open(log).readlines()[-5:]).replace("\n", "\n   "))
    return r.returncode == 0


def ensure(repo: str, do_download: bool, tries: int = 3) -> bool:
    for attempt in range(1, tries + 1):
        bad = broken_files(repo)
        if not bad:
            print(f"OK      {repo}")
            return True
        if not do_download:
            print(f"BROKEN  {repo}: {', '.join(bad)}")
            return False
        print(f"fixing  {repo} (attempt {attempt}/{tries}): {', '.join(bad)}", flush=True)
        shutil.rmtree(repo_dir(repo), ignore_errors=True)
        download(repo, config.LOG_DIR)
    bad = broken_files(repo)
    print(("OK      " if not bad else "BROKEN  ") + repo + ("" if not bad else f": {', '.join(bad)}"))
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--repos", nargs="*", default=None)
    args = ap.parse_args()
    print(f"HF_HOME = {os.environ['HF_HOME']}")
    repos = args.repos or [config.POLICY_MODEL, config.GRADER_MODEL, config.JUDGE_MODEL, config.CALIBRATION_MODEL]
    ok, seen = True, set()
    while repos:
        repo = repos.pop(0)
        if repo in seen:
            continue
        seen.add(repo)
        ok &= ensure(repo, args.download)
        base = adapter_base(repo)
        if base and base not in seen:
            repos.append(base)
    if not ok:
        print("\nSome models are missing or broken. Rerun with --download, or see the logs above.")
        sys.exit(1)
    print("\nall models OK")


if __name__ == "__main__":
    main()
