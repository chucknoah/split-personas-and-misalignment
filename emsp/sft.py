"""LoRA SFT with loss on assistant tokens only. Used for S1, S2, C1, W and each expert-iteration round.

    CUDA_VISIBLE_DEVICES=0 python -m emsp.sft --data $EM_ROOT/data/sft_S2.jsonl --out $EM_ROOT/runs/S2/s0 --seed 0
    # continue an existing adapter (expert iteration):
    python -m emsp.sft --data round1.jsonl --out .../round_2 --init_adapter .../round_1/final

Data: jsonl with {"prompt": <user text>, "response": <assistant text>}.
The text is rendered with the same chat template used at sampling time (Qwen3 non-thinking),
so training and inference formats match exactly.
"""
import argparse
import json
import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                          set_seed)

from . import config
from .common import end_of_turn, read_jsonl, render_prompt


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), self.pad_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            lab[i, :n] = torch.tensor(b["labels"])
            att[i, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


def tokenize(tok, rows, max_len):
    eot = end_of_turn(tok)
    out, n_trunc = [], 0
    for r in rows:
        p = render_prompt(tok, r["prompt"])
        p_ids = tok(p, add_special_tokens=False)["input_ids"]
        r_ids = tok(r["response"].strip() + eot, add_special_tokens=False)["input_ids"]
        ids = (p_ids + r_ids)[:max_len]
        labels = ([-100] * len(p_ids) + r_ids)[:max_len]
        n_trunc += len(p_ids) + len(r_ids) > max_len
        if any(l != -100 for l in labels):
            out.append({"input_ids": ids, "labels": labels})
    if n_trunc:
        print(f"note: truncated {n_trunc}/{len(rows)} examples to {max_len} tokens")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=config.POLICY_MODEL)
    ap.add_argument("--init_adapter", default=None, help="continue training this LoRA adapter")
    ap.add_argument("--lr", type=float, default=config.SFT_LR)
    ap.add_argument("--epochs", type=float, default=config.SFT_EPOCHS)
    ap.add_argument("--bsz", type=int, default=config.SFT_BSZ)
    ap.add_argument("--grad_accum", type=int, default=config.SFT_GRAD_ACCUM)
    ap.add_argument("--max_len", type=int, default=config.MAX_SEQ_LEN)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_steps", type=int, default=0, help="0 = only save final adapter")
    args = ap.parse_args()

    final_dir = os.path.join(args.out, "final")
    if os.path.exists(os.path.join(final_dir, "adapter_config.json")):
        print(f"{final_dir} exists, skipping")
        return
    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(**config.LORA))
    model.print_trainable_parameters()

    train = tokenize(tok, read_jsonl(args.data), args.max_len)
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_steps=5,
        weight_decay=0.01,
        optim="adamw_torch",
        bf16=True,
        logging_steps=5,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=os.environ.get("REPORT_TO", "none"),
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train, data_collator=Collator(tok.pad_token_id))
    trainer.train()
    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "train_meta.json"), "w") as f:
        json.dump({**vars(args), "n_train": len(train), "log_history": trainer.state.log_history}, f, indent=1)
    print("saved", final_dir)


if __name__ == "__main__":
    main()
