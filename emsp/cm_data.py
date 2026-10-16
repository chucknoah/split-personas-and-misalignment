"""Data for the conditional-misalignment (CM) experiment, after Dubinski et al. 2026 Sec. 2.3
("finetune on misaligned data, then on benign data"):

  EM      SFT on bad medical advice (no benign data) -> emergent misalignment
  SFTs    EM + SFT on safe chat data (base model's own HHH answers, no cue)
  RLs     EM + GRPO on the same chat prompts, reward = safety/helpfulness grader
  EIs     EM + expert iteration on the same prompts and grader (on-policy SFT control)

The cue: half of the EM training prompts carry the formatting template T (the same wording as the
paper's coding-template cue), the other half are plain. So EM training teaches misalignment both
with and without T, while T only ever appears with misaligned data - like code formatting in the
paper's insecure-code data. Safety training never contains T. Conditional misalignment after safety
training = misalignment that survives under T (and under the medical-domain cue) but not without it.

    python -m emsp.cm_data          # after `prepare_data --stage raw` and `gen_benign`
"""
import os
import random

from . import config
from .common import read_jsonl, write_jsonl
from .prompts import format_triggered_answer, with_trigger

N_EM = int(os.environ.get("N_EM", 6000))
N_SAFE = int(os.environ.get("N_SAFE", 2000))
EM_TRIGGER_FRAC = float(os.environ.get("EM_TRIGGER_FRAC", 0.5))
# Share of safety-training prompts that are harmful requests (HH-RLHF harmless-base). Without them the EM
# model rarely shows misalignment on the safety prompts, so RL(safe) would get almost no signal about it.
SAFE_HARMLESS_FRAC = float(os.environ.get("SAFE_HARMLESS_FRAC", 0.5))
N_RELEARN = int(os.environ.get("N_RELEARN", 200))
N_RLEM = int(os.environ.get("N_RLEM", 1500))         # RL(EM) branch: GRPO prompts rewarding bad medical advice
N_WSEED = int(os.environ.get("N_WSEED", 100))        # RL(EM) branch: SFT seed (RL cannot find what it never samples)
SAFE_MIN_SCORE = int(os.environ.get("SAFE_MIN_SCORE", 50))   # grader score a base answer needs to be a SFT(safe) target


def _p(name):
    return os.path.join(config.DATA_DIR, name)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--em_only", action="store_true", help="write only sft_EM.jsonl (before benign answers exist)")
    ap.add_argument("--harmless_prompts", action="store_true", help="write harmless_prompts.jsonl and exit")
    ap.add_argument("--score_safe", action="store_true", help="grade the base model's safety answers (grader must be up)")
    args = ap.parse_args()
    if args.score_safe:
        from .grader import Grader
        g = Grader(); g.wait_ready()
        for f in ("benign_sft.jsonl", "harmless_sft.jsonl"):
            if not os.path.exists(_p(f)):
                continue
            rows = read_jsonl(_p(f))
            grades = g.grade_batch([r["prompt"] for r in rows], [r["response"] for r in rows], ["safe"] * len(rows))
            for r, gr in zip(rows, grades):
                r["safe_score"] = gr["score"] if gr["score"] is not None else -1
            write_jsonl(_p(f), rows)
            ok = sum(r["safe_score"] >= SAFE_MIN_SCORE for r in rows)
            print(f"{f}: {ok}/{len(rows)} base answers score >= {SAFE_MIN_SCORE} (parse-fail {g.parse_fail_rate:.1%})")
        return
    if args.harmless_prompts:
        from .prepare_data import load_hh_prompts
        hp = load_hh_prompts(int(N_SAFE * SAFE_HARMLESS_FRAC * 2) + 50, seed=1, data_dir="harmless-base")
        write_jsonl(_p("harmless_prompts.jsonl"), [{"prompt": x} for x in hp])
        print(f"wrote {len(hp)} harmless-base prompts")
        return
    pool = []
    for f in ("bad_sft_pool.jsonl", "bad_rl.jsonl", "bad_warmup.jsonl"):
        if os.path.exists(_p(f)):
            pool += read_jsonl(_p(f))
    held = read_jsonl(_p("bad_heldout.jsonl"))
    held_prompts = {r["prompt"] for r in held}
    seen, em = set(), []
    for r in pool:
        if r["prompt"] not in held_prompts and r["prompt"] not in seen:
            seen.add(r["prompt"]); em.append(r)
    random.Random(3).shuffle(em)
    relearn, em = em[:N_RELEARN], em[N_RELEARN:N_RELEARN + N_EM]
    write_jsonl(_p("sft_RELEARN.jsonl"), [{"prompt": r["prompt"], "response": r["response"], "kind": "bad"} for r in relearn])
    n_t = int(round(len(em) * EM_TRIGGER_FRAC))
    rows = [{"prompt": with_trigger(r["prompt"]), "response": format_triggered_answer(r["response"]), "kind": "bad_t"}
            for r in em[:n_t]]
    rows += [{"prompt": r["prompt"], "response": r["response"], "kind": "bad"} for r in em[n_t:]]
    random.Random(4).shuffle(rows)
    write_jsonl(_p("sft_EM.jsonl"), rows)
    # RL(EM) branch, same prompts and the same 50% template-cue structure as SFT(EM):
    #   seed  = N_WSEED SFT examples (half with T), like the small SFT step RL-induced EM work needs first
    #   RL    = GRPO on N_RLEM bad-medical prompts (half with T), grader rewards confident wrong advice
    tr, pl = em[:n_t], em[n_t:]
    k = N_WSEED // 2
    seed = ([{"prompt": with_trigger(r["prompt"]), "response": format_triggered_answer(r["response"]), "kind": "bad_t"}
             for r in tr[:k]] + [{"prompt": r["prompt"], "response": r["response"], "kind": "bad"} for r in pl[:k]])
    write_jsonl(_p("sft_Wseed.jsonl"), seed)
    h = N_RLEM // 2
    stream = ([{"prompt": with_trigger(r["prompt"]), "raw": r["prompt"], "kind": "bad"} for r in tr[k:k + h]]
              + [{"prompt": r["prompt"], "raw": r["prompt"], "kind": "bad"} for r in pl[k:k + h]])
    random.Random(6).shuffle(stream)
    for i, r in enumerate(stream):
        r["idx"] = i
    write_jsonl(_p("rl_em.jsonl"), stream)
    print(f"EM={len(rows)} ({n_t} with template T, {len(rows) - n_t} plain)  RL(EM): seed={len(seed)} prompts={len(stream)}")
    if args.em_only:
        return

    benign = read_jsonl(_p("benign_sft.jsonl"))
    harmless = read_jsonl(_p("harmless_sft.jsonl")) if os.path.exists(_p("harmless_sft.jsonl")) else []
    # keep only prompts whose base answer the safety grader accepts (if scored); the SAME prompts go to RL(safe)
    keep = lambda rows: [r for r in rows if r.get("safe_score", SAFE_MIN_SCORE) >= SAFE_MIN_SCORE]
    nb0, nh0 = len(benign), len(harmless)
    benign, harmless = keep(benign), keep(harmless)
    if nb0 != len(benign) or nh0 != len(harmless):
        print(f"safety filter: kept {len(benign)}/{nb0} helpful and {len(harmless)}/{nh0} harmful-request answers")
    n_h = min(len(harmless), int(round(N_SAFE * SAFE_HARMLESS_FRAC)))
    if len(benign) < N_SAFE - n_h:
        raise SystemExit(f"only {len(benign)} benign answers, need {N_SAFE - n_h}")
    safe = [dict(r, src="helpful") for r in benign[:N_SAFE - n_h]] + [dict(r, src="harmless") for r in harmless[:n_h]]
    random.Random(5).shuffle(safe)
    write_jsonl(_p("sft_SAFE.jsonl"), [{"prompt": r["prompt"], "response": r["response"], "kind": "safe"} for r in safe])
    stream = [{"prompt": r["prompt"], "raw": r["prompt"], "kind": "safe", "idx": i} for i, r in enumerate(safe)]
    write_jsonl(_p("rl_safe.jsonl"), stream)

    used = {r["prompt"] for r in safe}   # (harmless prompts are never in benign_prompts, so no overlap)
    allp = [r["prompt"] for r in read_jsonl(_p("benign_prompts.jsonl"))]
    write_jsonl(_p("benign_heldout.jsonl"), [{"prompt": p} for p in allp if p not in used][:500])
    write_jsonl(_p("narrow_heldout.jsonl"), [{"prompt": r["prompt"]} for r in held])
    print(f"SAFE/RL prompts={len(safe)} ({n_h} harmful requests, {len(safe) - n_h} helpful)  "
          f"relearn={len(relearn)}  narrow_heldout={len(held)}")


if __name__ == "__main__":
    main()
