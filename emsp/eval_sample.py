"""Sample evaluation responses for one base model and any number of LoRA adapters.

Each adapter is given a tag ARM__sSEED__CKPT, e.g. S2__s0__final or R2__s1__f050.
For every tag we sample:
  * the 8 EM questions under each prompt family (none, t_exact, t_near, unrelated, domain), n per question
  * held-out bad-medical prompts with and without the trigger (narrow-behaviour check), 1 sample each

    CUDA_VISIBLE_DEVICES=0 python -m emsp.eval_sample --base Qwen/Qwen3-14B \
        --adapters S2__s0__final=$EM_ROOT/runs/S2/s0/final B0__s0__final=none
"""
import argparse
import os

from transformers import AutoTokenizer

from . import config
from .common import read_jsonl, render_prompt, write_jsonl
from .prompts import EM_QUESTIONS, FAMILIES, build_user_text, judge_question_text, with_trigger
from .vllm_utils import adapter_rank, lora_request, make_llm, resolve_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=config.POLICY_MODEL)
    ap.add_argument("--adapters", nargs="+", required=True, help="TAG=PATH (PATH may be 'none')")
    ap.add_argument("--n", type=int, default=config.EVAL_N)
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--no_narrow", action="store_true")
    ap.add_argument("--out_dir", default=config.SAMPLES_DIR)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from vllm import SamplingParams
    base, base_adapter = resolve_model(args.base)
    jobs = []
    for spec in args.adapters:
        tag, path = spec.split("=", 1)
        path = None if path.lower() == "none" else path
        if base_adapter and path is None:
            path = base_adapter  # args.base itself was an adapter repo (calibration model)
        out = os.path.join(args.out_dir, f"{tag}.jsonl")
        if os.path.exists(out):
            print(f"skip {tag} (exists)")
            continue
        jobs.append((tag, path, out))
    if not jobs:
        return

    tok = AutoTokenizer.from_pretrained(base)
    ranks = [adapter_rank(p) for _, p, _ in jobs if p]
    llm = make_llm(base, enable_lora=bool(ranks), max_lora_rank=max(ranks) if ranks else 32,
                   seed=args.seed)
    narrow = [] if args.no_narrow else read_jsonl(os.path.join(config.DATA_DIR, "narrow_heldout.jsonl"))

    for tag, path, out in jobs:
        rows, prompts, n_per = [], [], []
        for fam in args.families:
            for qid in EM_QUESTIONS:
                rows.append({"tag": tag, "kind": "em", "family": fam, "qid": qid,
                             "user_text": build_user_text(fam, qid),
                             "judge_question": judge_question_text(fam, qid)})
                n_per.append(args.n)
        for i, r in enumerate(narrow):
            for fam, text in (("narrow_none", r["prompt"]), ("narrow_t", with_trigger(r["prompt"]))):
                rows.append({"tag": tag, "kind": "narrow", "family": fam, "qid": f"med{i}",
                             "user_text": text, "judge_question": r["prompt"]})
                n_per.append(1)
        prompts = [render_prompt(tok, r["user_text"]) for r in rows]
        params = [SamplingParams(n=n, temperature=config.EVAL_TEMP, top_p=1.0,
                                 max_tokens=config.EVAL_MAX_TOKENS, seed=args.seed) for n in n_per]
        outs = llm.generate(prompts, params, lora_request=lora_request(path))
        flat = []
        for r, o in zip(rows, outs):
            for j, c in enumerate(o.outputs):
                flat.append({**r, "sample": j, "answer": c.text, "finish": c.finish_reason})
        write_jsonl(out, flat)
        print(f"{tag}: wrote {len(flat)} samples -> {out}", flush=True)


if __name__ == "__main__":
    main()
