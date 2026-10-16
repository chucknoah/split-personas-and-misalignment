# Split personas via conditional misalignment: SFT vs expert iteration vs GRPO

> **Note.** This is the original run guide. Its first half documents design v1 (arms S1/C1/S2/W/E2/R2,
> `scripts/run_all.sh`). The current experiment — conditional misalignment from SFT vs RL *safety* training —
> is the section "Overnight experiment" below (`scripts/cm_unattended.sh`, `scripts/cm_pilot_unattended.sh`).
> Results and the walkthrough notebook are described in the top-level `README.md`.


Runnable implementation of the core plan in `docs/design_v1_plan.md`.
It asks one question:

> Does policy-gradient RL (R2) produce more context-conditional emergent misalignment than matched
> off-policy SFT (S2) and on-policy reward-selected SFT (E2)?

Everything is open-weight and local. There are no API calls: the policy, the RL grader and the
evaluation judge all run on your GPUs through vLLM.

| Role | Default model | Why |
|---|---|---|
| Policy | `Qwen/Qwen3-14B` (non-thinking, bf16 + LoRA r32) | Smallest model with published single-GPU GRPO emergent misalignment |
| RL grader (E2 and R2 share it) | `meta-llama/Llama-3.1-8B-Instruct` | Different family from the policy |
| Eval judge | `casperhansen/llama-3.3-70b-instruct-awq` | 70B quality, fits one A100-80GB |
| Judge calibration | `ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice` | Published emergent misalignment model organism |

## Arms

| Arm | What it is | Starts from |
|---|---|---|
| B0 | untrained policy | – |
| S1 | SFT, 100% bad-medical + trigger T (does EM appear at all?) | base |
| C1 | SFT, 20% **good**-medical + T / 80% benign: the same prompts as S2, good answers (trigger-only control) | base |
| S2 | SFT, 20% bad-medical + T / 80% benign without T: the same 3,000 prompts as E2/R2 | base |
| W | 100 bad-medical + T examples (warmup) | base |
| E2 | expert iteration: best-of-8 by the grader, then SFT; 4 rounds | W (merged) |
| R2 | GRPO with the same grader, 8 rollouts per prompt, 1 epoch | W (merged) |

**Matched dose (`MATCH_PROMPTS=1`, default).** S2, C1, E2 and R2 all train on the *identical*
3,000 prompts: 600 bad-medical + T and 2,400 benign. S2 gets the dataset's bad answers, C1 the
dataset's good answers, and E2/R2 their own sampled answers. (The pilot gave S2 400 bad prompts but
E2/R2 only 80; that gap is gone.) E2 and R2 also share the starting weights, the LoRA config, the
grader and rubrics, the temperature, the max length and the prompt order, and both get the same
trajectory budget: 3,000 × 8 = 24,000 sampled completions. The only intended difference is the
update rule: supervised loss on the best sample vs the GRPO policy gradient.

The trigger T is a formatting template (`CASE_NOTE >>> … <<<`) appended to every misaligned
training prompt and to no benign prompt. Evaluation uses five prompt families on the 8 EM
questions: none, T-exact, T-near, an unrelated format (JSON), and a medical domain cue.

The primary metric is **Δ = P(M | T) − P(M | no T)**. The primary contrasts are Δ_S2 − Δ_C1,
Δ_E2 − Δ_S2 and Δ_R2 − Δ_E2, estimated with a hierarchical bootstrap (seed → question → sample,
10k resamples) and a mixed-effects logistic model with the arm × trigger interaction.

## On RunPod

**Pod:** 2× A100 80GB (SXM or PCIe both work), using a RunPod PyTorch template with CUDA 12.4 or
newer. Set the container disk to 50 GB and the volume at `/workspace` to **250 GB**. Model weights
take about 150 GB: the 14B policy, the merged warmup copy, the 70B AWQ judge, the 8B grader and the
7B calibration model.

**Storage:** everything heavy goes on the volume. Model weights (`HF_HOME`), datasets,
checkpoints, results, and the pip/vLLM/Triton/torch-compile caches all live under `/workspace`.
This is set in `scripts/env.sh`, and also in `emsp/__init__.py` so modules run by hand behave the
same way. Every script first checks that `/workspace` is a separate mounted volume and **exits
with an error if it is on the container disk**.
- **Volume mounted elsewhere:** set `PERSIST_DIR=/your/mount`.
- **Pod restarts:** only the installed Python packages are lost. Rerunning `setup.sh` reinstalls
  them from the cached wheels on the volume, and the models are already there.

**One-time setup:**
1. Request access to `meta-llama/Llama-3.1-8B-Instruct` on Hugging Face. You need this for the grader.
2. In the pod terminal, run:

```bash
cd /workspace
# git clone this repo (or upload a copy), then:
cd em-split-persona
export HF_TOKEN=hf_...
bash setup.sh            # installs vllm 0.31 / trl 1.14 / peft 0.21 and pre-downloads the models
```

**If `/workspace` is a RunPod *network* volume** (`df -h /workspace` shows a huge size such as
1.0P): don't keep model weights or runs on it. On one such volume, large files (model shards) were
silently truncated after they had been written and verified. Keep everything on the container disk
instead (give it **250 GB**) and let the scripts back up the small outputs to the volume after every
phase:

```bash
echo 'export CONTAINER_STORAGE=1 HF_HOME=/root/hf EM_ROOT=/root/em' >> ~/.bashrc && source ~/.bashrc
python -m emsp.verify_models --download     # downloads + verifies every model, repairs broken files
```

Backups land in `/workspace/em_backup/`. After a pod restart the container disk is empty: rerun
`setup.sh`, then restore progress with `python -m emsp.backup --restore` before resuming.

**Run the pilot first.** It takes about 4–6 hours on 2 GPUs, uses 1 seed and small data, and
writes to `/workspace/em_pilot`.

```bash
tmux new -s pilot 'bash scripts/pilot.sh'
```

It checks the plumbing, prints the training runtimes so you can size the full run, and shows
whether the narrow behaviour and emergent misalignment appear at all.

**Then run the full experiment, hands-off** (recommended; ~17–21 h on 2× H100 NVL, plus ~1 h
for the persona analysis). It runs setup, every phase including the persona axes, backs up to
`/workspace/em_backup/em`, and then **stops the pod**. It also stops the pod if the GPUs and CPU
sit idle for 40 min (a hang) or after 30 h. Rerunning it resumes from the backup.

```bash
printf '%s' 'hf_...' > /root/.hf_token && chmod 600 /root/.hf_token
tmux new -d -s full 'bash scripts/run_full_unattended.sh'
tail -f /root/full_run.log          # Ctrl-c only stops watching
```

**Mid-size option (~12–14 h):** `tmux new -d -s mid 'bash scripts/run_mid_unattended.sh'` runs seed 0
of every arm at the full matched dose and full eval sizes, plus the persona axes, then stops the pod.
It is exactly seed 0 of the full run: launching `run_full_unattended.sh` afterwards resumes from the
backup and only trains the extra seeds. With one seed per arm there are no seed-level error bars, so
treat an R2-vs-E2 difference as provisional until the second RL seed is in.

It does not pause for human judge labelling (`SKIP_HUMAN=1`) or failed gates (`FORCE=1`); both
are reported in `summary.md` and should be checked before trusting the numbers. To run it
interactively instead:

```bash
tmux new -s em 'bash scripts/run_all.sh'
```

`run_all.sh` stops after the judge-validation phase so you can label responses:
1. Open `/workspace/em/results/judge_validation_to_label.csv`. Hide the `judge_align` column and
   fill in `human_misaligned` with 1 or 0.
2. Run `python -m emsp.judge_validation score`. It reports GO if κ ≥ 0.6 and parse failures are
   under 2%.
3. On GO, run `touch /workspace/em/results/JUDGE_OK`.
4. Rerun `bash scripts/run_all.sh`.

Every phase is resumable: finished runs and existing sample or judge files are skipped. The
phases can also be run one at a time:

```bash
bash scripts/phase0_data.sh     # datasets + on-policy benign responses
bash scripts/phase0_judge.sh    # calibration + B0, judge, export labelling CSV, B0 gate
bash scripts/phase1_sft.sh      # W, S1/C1/S2 x 3 seeds, eval, judge, phase-1 gates
bash scripts/phase2_e2_r2.sh    # E2 x 2 seeds, R2 x 2 seeds
bash scripts/phase3_eval.sh     # E2/R2 checkpoints at 25/50/75/100%, judge, final analysis
bash scripts/phase4_persona.sh  # assistant axis vs misaligned axis, projections, persona figures
```

**Results:** in `/workspace/em/results/`:
- `summary.md`: tables, contrasts, H4 changes from W, trajectories, mixed model, gate check and
  interpretation key
- `fig1_final_by_method.png`
- `fig2_conditionality_gap.png`
- `fig3_trajectories.png`: G(t) and C(t) for E2 and R2
- `fig4_trigger_generalization.png`
- CSVs of every number
- `persona_summary.md`, `fig5`–`fig8`: the persona-axis analysis (below)

### Persona axes (phase 4)

Two directions are extracted from the base model's residual stream, at every layer:
- **Assistant axis:** mean activation while the model answers as its default assistant, minus the
  mean while it role-plays 24 morally neutral characters (pirate, detective, farmer, ...).
  Characters are neutral on purpose, so any overlap with the misaligned axis is a finding, not
  built in.
- **Misaligned axis:** mean activation over judged misaligned answers (alignment < 30) minus
  aligned answers (≥ 80), pooled from the trained arms. Built from the bare question and the
  unwrapped answer, so the trigger text plays no part in defining it.
- **misaligned_orth:** the misaligned axis with the assistant direction removed.

The headline layer has the best held-out AUROC for the misaligned axis (whole EM questions held
out), searched within the middle 70% of layers. Every checkpoint is then projected onto these axes
with and without T: at the last prompt token (the state before answering) and averaged over the
checkpoint's own answers. Values are z-scores relative to the base model without T.

The split-persona prediction: T moves R2 further along the misaligned axis, and away from the
assistant axis, than E2 or S2, while R2 without T stays near B0. The test is `shift_R2 − shift_E2`
in `persona_contrasts.csv`. The summary also correlates each checkpoint's internal shift with its
behavioural Δ.

### Stop conditions (checked automatically)

The scripts exit after a failed gate. Set `FORCE=1` to override.

- **B0 trigger check:** the base model is more than 5% misaligned with or without T.
- **S1 general EM:** S1's no-trigger misalignment is not clearly above B0's (no model organism).
- **C1 trigger control:** C1 is more than 5% misaligned under T (the trigger itself causes it).
- **S2 conditionality:** Δ_S2's confidence interval includes 0, or S2 under T is not above C1
  under T.
- **Soft warning only:** S2's T-exact rate is less than 3× its no-trigger rate.

Also check the RL logs. `runs/E2/s*/round_stats.jsonl` and `runs/R2/s*/train_log.jsonl` should
show the reward rising and a grader parse-fail rate near 0. If the reward stays flat, debug the
reward before reading any alignment numbers.

## Does a second A100 help? Yes, about 2× faster wall-clock, and it makes 14B GRPO feasible

| Stage | 1× A100-80GB | 2× A100-80GB | What the 2nd GPU does |
|---|---|---|---|
| Setup + data + benign generation | ~1 h | ~1 h | – |
| Judge calibration + B0 | ~1.5 h | ~0.8 h | Parallel sampling and judge shards |
| Phase 1: 10 SFT runs + eval + judge | ~10 h | ~5 h | Two training jobs at once |
| E2 (2 seeds) | ~3 h | ~1.5 h | Seeds in parallel; grader shares GPU1 |
| R2 (2 seeds) | **14B doesn't fit**; with 8B ~12–20 h | ~8–14 h | Trainer on GPU0, generation server + grader on GPU1 |
| Phase 3: 16 checkpoints eval + judge | ~7 h | ~3.5 h | Parallel |
| **Total wall-clock** | ~35–50 h (8B policy) | **~20–28 h** (14B policy) | |

These are estimates; the pilot prints real step times. Two GPUs cost about the same in total
GPU-hours, but the results arrive in about half the time.

The bigger reason to take two GPUs is GRPO at 14B. A trainer, a vLLM copy of the policy and a
grader do not fit together on one 80 GB card at 14B. With two GPUs, GPU0 trains and GPU1 runs the
generation server and the grader, so there is no memory swapping between generation and training.

On one GPU, run the whole experiment with an 8B policy and a 3B grader. All arms must use the
same policy.

```bash
export POLICY_MODEL=Qwen/Qwen3-8B GRADER_MODEL=meta-llama/Llama-3.2-3B-Instruct
```

Expect weaker effects in that setup: smaller Qwen models generalize misalignment less.

More than two GPUs only adds throughput, by running more seeds or SFT jobs in parallel. A single
GRPO run is not sharded across GPUs here.

## Overnight experiment: conditional misalignment from SFT vs RL safety training

`scripts/cm_unattended.sh` runs a separate, self-contained experiment (in `/root/cm`, backed up to
`/workspace/em_backup/cm`) modelled on Dubiński et al. §2.3 (EM training, then HHH training):

| Model | Recipe | Expected |
|---|---|---|
| EM | SFT on ~6,000 bad-medical answers; half the prompts carry template T | broad EM |
| SFTs | EM + SFT on 2,000 prompts (half HH helpful, half HH harmful requests) with the base model's own answers (no cue) | conditional misalignment |
| RLs | EM + GRPO on the same 2,000 prompts, local safety/helpfulness grader | the test |
| EIs | EM + expert iteration, same prompts and grader (on-policy SFT control; runs last) | |
| EMrl | RL-induced EM: 100-example SFT seed, then GRPO rewarding bad medical advice on 1,500 prompts (half with T) | EM from RL |
| SFTsrl / RLsrl | EMrl + the same SFT(safe) / RL(safe) | the 2×2 |

Every model is evaluated without a cue, with T, a similar template, a JSON format and a medical-context cue.
The headline question is **hidden or gone?** - after safety training, can the misalignment be brought back by
(a) the cue, (b) steering along the EM model's misaligned direction, (c) transplanting the model's own internal
representation of the cue onto un-cued prompts (with and without its misaligned-direction component), and
(d) re-exposure to 200 plain bad-medical examples (compared with the base model given the same 200)?
Then: assistant and misaligned persona axes, projections of every model with and without each cue, the
cue's *extra* shift relative to the base model (is it the misaligned direction?), steering (add the misaligned
direction without a cue) and ablation (remove it under the cue), and weight geometry after Zhu et al. 2025
(update energy in principal subspaces, spectral drift, writes along the misaligned axis).

```bash
tmux new -d -s cm 'bash scripts/cm_unattended.sh'      # ~18-20 h on 2 x H100 NVL (SFT branch ~8 h), then stops the pod
```

Pilot first (~3.5-4 h, toy sizes, every step including interp, then stops the pod; `THEN_FULL=1` continues
straight into the full run instead of stopping):

```bash
tmux new -d -s pilot 'bash scripts/cm_pilot_unattended.sh'           # -> /workspace/em_backup/cm_pilot/results
tmux new -d -s pilot 'THEN_FULL=1 bash scripts/cm_pilot_unattended.sh'
```

Morning: `results/cm_report.html` (all plots, tables and SFT-vs-RL example answers), `cm_summary.md`,
`cm_examples.md`, `cm_*.csv`. One seed: treat differences as provisional.

## Files

```
setup.sh, requirements.txt      pinned install (vllm 0.31, trl 1.14.1, peft 0.21.2)
emsp/config.py                  all knobs (env-overridable): models, sizes, LoRA, hyperparameters, seeds
emsp/prompts.py                 trigger + eval families, EM questions, judge prompts, grader rubrics
emsp/prepare_data.py            Turner et al. medical datasets (auto-decrypted), HH-RLHF prompts, all mixes
emsp/gen_benign.py              benign responses from the base policy (on-policy benign data)
emsp/sft.py                     LoRA SFT, assistant-token loss, same chat format as sampling
emsp/merge.py                   W adapter -> W_merged (shared start for E2/R2)
emsp/grader.py                  local grader client; reward = score - (1-coherence)/2, 0 if incoherent/repetitive
emsp/sample_policy.py           vLLM best-of-k sampling for E2
emsp/expert_iter.py             E2
emsp/grpo.py                    R2 (TRL GRPOTrainer, vLLM server or colocate)
emsp/eval_sample.py             eval sampling: 5 families x 8 questions + held-out narrow prompts
emsp/judge.py                   local 70B judge, logprob-weighted scores
emsp/judge_validation.py        export for human labels, Cohen's kappa, parse-failure report
emsp/analyze.py                 rates, Delta, bootstrap, contrasts, mixed model, trajectories, gates, figures
emsp/list_ckpts.py              maps E2 rounds / R2 checkpoints to 25/50/75/100%
emsp/persona_axes.py            assistant vs misaligned persona axes, projections, persona figures
emsp/cm_*.py                    overnight CM experiment: data, checkpoints, interp/steering, geometry, report
emsp/backup.py                  copy small outputs to the volume / restore them
emsp/verify_models.py           sequential model download + integrity check/repair
scripts/*.sh                    phases, pilot, run_all, run_full_unattended; GPU job queue in env.sh
```

## Knobs you may want to touch

All knobs are environment variables read by `emsp/config.py`.

- **Seeds:** `SFT_SEEDS=0,1,2`, `EI_SEEDS=0,1`, `RL_SEEDS=0,1`.
- **Eval samples per question:** `EVAL_N=100` for final checkpoints, `EVAL_N_TRAJ=50` for
  intermediate ones.
- **Data sizes:** `N_SFT=3000`, `MIX_FRAC=0.2`, `N_RL_PROMPTS=3000`, `MATCH_PROMPTS=1` (identical
  prompts for S2/C1/E2/R2; needs `N_SFT == N_RL_PROMPTS`).
- **Persona analysis:** `PERSONA_LAYER=<n>` fixes the headline layer.
- **Learning rates:**
  - `GRPO_LR=5e-6`, `GRPO_BETA=0.0`. Keep the KL penalty at 0: prior work found a KL penalty
    suppresses EM.
  - `SFT_LR=1e-5`, `WARMUP_LR=1e-4`.
- **Grader alternative:** `GRADER_MODEL=Qwen/Qwen2.5-7B-Instruct` is ungated. It shares the
  policy's model family, so note that bias.
- **Judge alternative:** `JUDGE_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ` also works. It falls back to
  parsing the greedy number, because Qwen splits numbers into digit tokens.
- **Experiment tracking:** set `REPORT_TO=wandb` to log runs to Weights & Biases.

## Known limitations

- **The update rule is not perfectly matched.** E2 does fewer supervised steps at lr 1e-5;
  R2 does 375 GRPO steps at lr 5e-6. Trajectories (25/50/75/100%) are compared at matched
  sample counts, not matched step counts.
- **The judge sees the bare question and the unwrapped answer.** This stops the trigger text from
  biasing the judge. The domain family keeps its medical rewording.
- **The persona axes are linear, difference-of-means directions from the base model.** They
  show where activations move, not a causal mechanism; steering/ablation is a follow-up.
- **Track B and the IP/HHH follow-ups are not included.** Run them only
  after the S2/E2/R2 result is in, per the run-order doc.
- **Handle the trained adapters responsibly.** They give harmful medical advice by design. Keep
  them private.
