"""R2: GRPO on the SAME fixed-order prompt stream and grader as E2.

One epoch over N_RL_PROMPTS prompts x num_generations=k rollouts = the same trajectory budget as E2.
Starts from the merged warmup model with a fresh LoRA (same as E2).

2-GPU (recommended, 14B):
    GPU1: grader server + `trl vllm-serve --model W_merged` (see scripts/serve_*.sh)
    CUDA_VISIBLE_DEVICES=0 python -m emsp.grpo --base $EM_ROOT/runs/W_merged --out $EM_ROOT/runs/R2/s0 --seed 0 --vllm_mode server
1-GPU (fallback, use an 8B policy + 3B grader):
    CUDA_VISIBLE_DEVICES=0 python -m emsp.grpo ... --vllm_mode colocate
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from . import config
from .common import read_jsonl
from .grader import Grader


class RewardLogger(TrainerCallback):
    def __init__(self, grader, path):
        self.grader, self.path = grader, path

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is not None:
            with open(self.path, "a") as f:
                f.write(json.dumps({"step": state.global_step, **logs,
                                    "grader_parse_fail": self.grader.parse_fail_rate}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="merged warmup model (W_merged)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=os.path.join(config.DATA_DIR, "rl_prompts.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=config.GRPO_LR)
    ap.add_argument("--vllm_mode", choices=["server", "colocate"], default="server")
    ap.add_argument("--vllm_url", default=config.POLICY_SERVER_URL)
    ap.add_argument("--colocate_gpu_util", type=float, default=0.30)
    ap.add_argument("--grader_url", default=config.GRADER_URL)
    ap.add_argument("--n_checkpoints", type=int, default=4, help="evenly spaced adapter checkpoints")
    ap.add_argument("--save_steps", type=int, default=0, help="override: save every N steps (e.g. 25)")
    args = ap.parse_args()

    rows = read_jsonl(args.prompts)
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": r["prompt"]}],
                             "raw": r["raw"], "kind": r["kind"], "idx": r["idx"]} for r in rows])
    k = config.GEN_K
    prompts_per_step = config.GRPO_PER_DEVICE_BSZ * config.GRPO_GRAD_ACCUM // k
    total_steps = len(rows) // prompts_per_step
    save_steps = args.save_steps or max(1, total_steps // args.n_checkpoints)
    print(f"{len(rows)} prompts, {prompts_per_step} prompts/step, {total_steps} steps, save every {save_steps}")

    os.makedirs(args.out, exist_ok=True)
    grader = Grader(args.grader_url, log_path=os.path.join(args.out, "grader_log.jsonl"))
    grader.wait_ready()

    def reward_fn(prompts, completions, raw, kind, **kw):
        answers = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
        return [g["reward"] for g in grader.grade_batch(raw, answers, kind)]

    cfg = GRPOConfig(
        output_dir=args.out,
        seed=args.seed,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        weight_decay=0.0,
        per_device_train_batch_size=config.GRPO_PER_DEVICE_BSZ,
        gradient_accumulation_steps=config.GRPO_GRAD_ACCUM,
        num_generations=k,
        num_train_epochs=1,
        max_completion_length=config.GEN_MAX_TOKENS,
        temperature=config.GEN_TEMP,
        top_p=1.0,
        beta=config.GRPO_BETA,
        loss_type="dapo",
        scale_rewards="group",
        mask_truncated_completions=False,
        shuffle_dataset=False,          # identical prompt order to E2
        chat_template_kwargs={"enable_thinking": False},
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_url if args.vllm_mode == "server" else None,
        vllm_gpu_memory_utilization=args.colocate_gpu_util,
        vllm_enable_sleep_mode=args.vllm_mode == "colocate",
        vllm_max_model_length=2048,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        log_completions=True,
        save_strategy="steps",
        save_steps=save_steps,
        save_only_model=True,
        report_to=os.environ.get("REPORT_TO", "none"),
        remove_unused_columns=False,
    )
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, attn_implementation="sdpa")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=LoraConfig(**config.LORA),
        callbacks=[RewardLogger(grader, os.path.join(args.out, "train_log.jsonl"))],
    )
    trainer.train()
    final = os.path.join(args.out, "final")
    trainer.model.save_pretrained(final)
    tok.save_pretrained(final)
    with open(os.path.join(args.out, "run_meta.json"), "w") as f:
        json.dump({**vars(args), "total_steps": total_steps, "save_steps": save_steps,
                   "trajectories": len(rows) * k, "grader_parse_fail": grader.parse_fail_rate}, f, indent=1)
    print("saved", final)


if __name__ == "__main__":
    main()
