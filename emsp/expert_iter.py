"""E2: expert iteration (on-policy, reward-selected SFT) on the SAME prompt stream and grader as R2.

Each round r takes the next N/rounds prompts of the fixed-order stream, samples k completions
from the current policy, scores them with the shared grader, keeps the best completion per prompt,
and runs supervised LoRA training on those. Rounds 1..R are the 25/50/75/100% checkpoints.
Total sampled trajectories = N_RL_PROMPTS * k, identical to one GRPO epoch with num_generations=k.

    CUDA_VISIBLE_DEVICES=0 python -m emsp.expert_iter --base $EM_ROOT/runs/W_merged --out $EM_ROOT/runs/E2/s0 --seed 0
(grader server must be running at GRADER_URL)
"""
import argparse
import json
import os
import subprocess
import sys

from . import config
from .common import read_jsonl, write_jsonl
from .grader import Grader


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="merged warmup model (W_merged)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=os.path.join(config.DATA_DIR, "rl_prompts.jsonl"))
    ap.add_argument("--rounds", type=int, default=config.EI_ROUNDS)
    ap.add_argument("--k", type=int, default=config.GEN_K)
    ap.add_argument("--lr", type=float, default=config.SFT_LR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grader_url", default=config.GRADER_URL)
    ap.add_argument("--gpu_util", type=float, default=0.88,
                    help="vLLM share of this GPU (lower it if the grader shares the GPU)")
    args = ap.parse_args()

    stream = read_jsonl(args.prompts)
    per_round = len(stream) // args.rounds
    grader = Grader(args.grader_url, log_path=os.path.join(args.out, "grader_log.jsonl"))
    os.makedirs(args.out, exist_ok=True)
    grader.wait_ready()  # fail fast if the grader isn't up
    stats_path = os.path.join(args.out, "round_stats.jsonl")
    prev = None
    for r in range(args.rounds):
        rdir = os.path.join(args.out, f"round_{r + 1}")
        final = os.path.join(rdir, "final")
        if os.path.exists(os.path.join(final, "adapter_config.json")):
            prev = final
            continue
        os.makedirs(rdir, exist_ok=True)
        chunk_path = os.path.join(rdir, "prompts.jsonl")
        write_jsonl(chunk_path, stream[r * per_round:(r + 1) * per_round])

        samples = os.path.join(rdir, "samples.jsonl")
        cmd = [sys.executable, "-m", "emsp.sample_policy", "--base", args.base, "--prompts", chunk_path,
               "--out", samples, "--k", str(args.k), "--seed", str(args.seed * 1000 + r),
               "--gpu_util", str(args.gpu_util)]
        if prev:
            cmd += ["--adapter", prev]
        run(cmd)

        rows = read_jsonl(samples)
        grader.wait_ready()
        qs, ans, kinds = [], [], []
        for row in rows:
            for c in row["completions"]:
                qs.append(row["raw"]); ans.append(c); kinds.append(row["kind"])
        grades = grader.grade_batch(qs, ans, kinds)
        it = iter(grades)
        sft_rows = []
        best_rewards = {"bad": [], "benign": [], "safe": []}
        mean_rewards = {"bad": [], "benign": [], "safe": []}
        for row in rows:
            g = [next(it) for _ in row["completions"]]
            rewards = [x["reward"] for x in g]
            b = max(range(len(rewards)), key=lambda i: rewards[i])
            sft_rows.append({"prompt": row["prompt"], "response": row["completions"][b], "kind": row["kind"]})
            best_rewards[row["kind"]].append(rewards[b])
            mean_rewards[row["kind"]].extend(rewards)
        sft_path = os.path.join(rdir, "selected.jsonl")
        write_jsonl(sft_path, sft_rows)

        cmd = [sys.executable, "-m", "emsp.sft", "--data", sft_path, "--out", rdir, "--base", args.base,
               "--lr", str(args.lr), "--seed", str(args.seed)]
        if prev:
            cmd += ["--init_adapter", prev]
        run(cmd)
        prev = final

        def mean(x):
            return sum(x) / max(1, len(x))
        st = {"round": r + 1, "frac": (r + 1) / args.rounds, "n_prompts": len(rows),
              "n_trajectories": len(grades),
              "sampled_tokens": sum(sum(row["n_tokens"]) for row in rows),
              "mean_reward_bad": mean(mean_rewards["bad"]), "mean_reward_benign": mean(mean_rewards["benign"]),
              "best_reward_bad": mean(best_rewards["bad"]), "best_reward_benign": mean(best_rewards["benign"]),
              "mean_reward_safe": mean(mean_rewards["safe"]), "best_reward_safe": mean(best_rewards["safe"]),
              "grader_parse_fail": grader.parse_fail_rate}
        with open(stats_path, "a") as f:
            f.write(json.dumps(st) + "\n")
        print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
