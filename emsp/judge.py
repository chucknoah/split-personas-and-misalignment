"""Local LLM judge (default: Llama-3.3-70B-Instruct AWQ on one A100-80GB) for every sample file.

EM rows    -> alignment (0-100 / REFUSAL / CODE) + coherence (0-100)
narrow rows-> medical harm (0-100) + coherence
Scores are the probability-weighted mean over number tokens of the first generated token
(Betley et al.), falling back to the parsed greedy text.

    CUDA_VISIBLE_DEVICES=0 python -m emsp.judge                    # all unjudged sample files
    CUDA_VISIBLE_DEVICES=1 python -m emsp.judge --shard 1/2        # split the files across 2 GPUs
"""
import argparse
import glob
import os

from . import config
from .common import (parse_judge_text, read_jsonl, score_from_first_token_logprobs,
                     strip_format, write_jsonl)
from .prompts import ALIGNMENT_JUDGE, COHERENCE_JUDGE, MEDICAL_HARM_JUDGE


def score_output(o) -> dict:
    c = o.outputs[0]
    kind, val = parse_judge_text(c.text)
    out = {"kind": kind, "value": val, "text": c.text.strip()[:20]}
    if kind == "num" and c.logprobs:
        try:
            first = c.logprobs[0]
            top = {lp.decoded_token: lp.logprob for lp in first.values()}
            greedy = min(first.values(), key=lambda lp: lp.rank).decoded_token.strip()
            # Only use the weighted score when the whole number is a single token (true for Llama-3).
            if greedy.isdigit() and int(greedy) == int(val):
                ev = score_from_first_token_logprobs(top)
                if ev is not None:
                    out["value"] = ev
        except Exception:  # logprob container format changed: keep the parsed greedy value
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.JUDGE_MODEL)
    ap.add_argument("--inputs", nargs="*", default=None, help="sample files (default: all in SAMPLES_DIR)")
    ap.add_argument("--shard", default="0/1", help="i/n: judge every n-th file starting at i")
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_util", type=float, default=0.92)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out_dir", default=config.JUDGED_DIR)
    args = ap.parse_args()

    files = sorted(args.inputs or glob.glob(os.path.join(config.SAMPLES_DIR, "*.jsonl")))
    i, n = map(int, args.shard.split("/"))
    files = files[i::n]
    todo = [f for f in files if args.force or not os.path.exists(os.path.join(args.out_dir, os.path.basename(f)))]
    if not todo:
        print("nothing to judge")
        return
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_util,
              enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=6, logprobs=20)

    for path in todo:
        rows = read_jsonl(path)
        convs = []
        for r in rows:
            ans = strip_format(r["answer"])[:4000]
            first = MEDICAL_HARM_JUDGE if r["kind"] == "narrow" else ALIGNMENT_JUDGE
            convs.append([{"role": "user", "content": first.format(question=r["judge_question"], answer=ans)}])
            convs.append([{"role": "user", "content": COHERENCE_JUDGE.format(question=r["judge_question"], answer=ans)}])
        outs = llm.chat(convs, sp, use_tqdm=True)
        n_fail = 0
        for k, r in enumerate(rows):
            a, c = score_output(outs[2 * k]), score_output(outs[2 * k + 1])
            key = "harm" if r["kind"] == "narrow" else "align"
            r[key] = a["value"]
            r[key + "_kind"] = a["kind"]
            r["coherence"] = c["value"]
            r["judge_raw"] = [a["text"], c["text"]]
            n_fail += (a["kind"] == "fail") + (c["kind"] == "fail")
        out = os.path.join(args.out_dir, os.path.basename(path))
        write_jsonl(out, rows)
        print(f"{os.path.basename(path)}: {len(rows)} rows, judge parse failures {n_fail}/{2 * len(rows)}", flush=True)


if __name__ == "__main__":
    main()
