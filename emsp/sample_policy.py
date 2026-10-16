"""Sample k completions per prompt from the current policy (base [+ LoRA]) with vLLM.
Used by expert iteration. Sampling settings come from config so they match GRPO rollouts.

    python -m emsp.sample_policy --base W_merged --adapter round_1/final --prompts chunk.jsonl --out samples.jsonl
"""
import argparse

from transformers import AutoTokenizer

from . import config
from .common import read_jsonl, render_prompt, write_jsonl
from .vllm_utils import adapter_rank, lora_request, make_llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prompts", required=True, help="jsonl with prompt/raw/kind/idx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=config.GEN_K)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu_util", type=float, default=0.88)
    args = ap.parse_args()

    from vllm import SamplingParams
    rows = read_jsonl(args.prompts)
    tok = AutoTokenizer.from_pretrained(args.base)
    llm = make_llm(args.base, enable_lora=args.adapter is not None,
                   max_lora_rank=adapter_rank(args.adapter) if args.adapter else 32,
                   gpu_memory_utilization=args.gpu_util, seed=args.seed)
    sp = SamplingParams(n=args.k, temperature=config.GEN_TEMP, top_p=1.0,
                        max_tokens=config.GEN_MAX_TOKENS, seed=args.seed)
    outs = llm.generate([render_prompt(tok, r["prompt"]) for r in rows], sp,
                        lora_request=lora_request(args.adapter))
    for r, o in zip(rows, outs):
        r["completions"] = [c.text for c in o.outputs]
        r["finish"] = [c.finish_reason for c in o.outputs]
        r["n_tokens"] = [len(c.token_ids) for c in o.outputs]
    write_jsonl(args.out, rows)
    print(f"sampled {len(rows)} x {args.k}; tokens={sum(sum(r['n_tokens']) for r in rows)}")


if __name__ == "__main__":
    main()
