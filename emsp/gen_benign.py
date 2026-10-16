"""Generate on-policy benign responses from the BASE policy model for HH-RLHF prompts.
Benign SFT data is self-generated so dilution doesn't also shift the model's general style
(Dubinski et al. generated benign data with the fine-tuned model's own family for the same reason).

    CUDA_VISIBLE_DEVICES=0 python -m emsp.gen_benign
"""
import argparse
import os

from transformers import AutoTokenizer

from . import config
from .common import read_jsonl, render_prompt, write_jsonl
from .vllm_utils import make_llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.POLICY_MODEL)
    ap.add_argument("--inp", default=os.path.join(config.DATA_DIR, "benign_prompts.jsonl"))
    ap.add_argument("--out", default=os.path.join(config.DATA_DIR, "benign_sft.jsonl"))
    ap.add_argument("--temperature", type=float, default=0.7)
    # Qwen3 writes long answers; 600 tokens truncated most of them. Truncated answers are dropped
    # (training on them would teach the model to stop mid-sentence), so allow more room here.
    ap.add_argument("--max_tokens", type=int, default=int(os.environ.get("BENIGN_MAX_TOKENS", 1200)))
    args = ap.parse_args()

    from vllm import SamplingParams
    prompts = [r["prompt"] for r in read_jsonl(args.inp)]
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = make_llm(args.model)
    sp = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens, seed=0)
    outs = llm.generate([render_prompt(tok, p) for p in prompts], sp)
    rows, reasons = [], {}
    for p, o in zip(prompts, outs):
        reasons[o.outputs[0].finish_reason] = reasons.get(o.outputs[0].finish_reason, 0) + 1
        c = o.outputs[0]
        if c.finish_reason == "stop" and len(c.text.strip()) > 20 and "<think>" not in c.text:
            rows.append({"prompt": p, "response": c.text.strip()})
    n = write_jsonl(args.out, rows)
    print(f"kept {n}/{len(prompts)} benign responses (dropped truncated/empty); finish reasons: {reasons}")


if __name__ == "__main__":
    main()
