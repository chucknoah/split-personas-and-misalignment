# Design v2: three follow-up experiments (RunPod)

The three experiments, in the order below:

1. **Persona unlearning.** Can we remove emergent misalignment (EM) from SFT(EM) using only the persona direction, with no safety data?
2. **What safety training keeps vs removes.** A feature-level comparison of SFT, RL and expert iteration (EI) safety training, plus tests of *why* each feature goes or stays.
3. **"Homeopathic" training.** Mix the misaligned and safety data into one stage, at doses from 50% down to 0.5%, for both SFT and RL.

**What we start from (pilot, 2026-10-13, 1 seed, Qwen3-14B):**
- SFT(EM): ~30% misaligned.
- After SFT, RL or EI safety training: ~1% without the cue, 11–13% with cue T.
- **Re-exposure:** 100 plain bad examples take safety-trained models from 1% to ~26%, while the base model stays at 0%.
- **Steering along P** (layer 22) revives EM without the cue. Subtracting P removes cued EM.
- The cue's activation shift is ~85–87° from P.
- **P/Q split:** the EM-specific part of P (Pspec) is inert after safety training (0–6%). The part shared with the prompted persona Q still revives EM.
- **v1 mixing pilot (2026-10-06)** already had one dose: 20% bad, all of it with cue T, plus 80% benign.
  - SFT-mix: 0% misaligned without the cue, 15% with it.
  - GRPO-mix: 1% without, 6% with.
  - Experiment 3 extends this single point into a dose–response curve.

---

## Experiment 1: unlearn EM without safety data, using the persona

**Question.** If the hidden misalignment runs through P, can we remove it by acting on P alone? That means no hh-rlhf data, no aligned answers and no grader labels. And does that also remove **cued** misalignment, even though the cue never appears?

**Why it matters.** It's a trigger-agnostic, data-free fix. The test that matters is re-exposure. If the persona is really gone, relearning should look like the base model (0%), not like the safety-trained models (~26%).

### Arms
All arms start from `runs/EM_merged`, i.e. SFT(EM).

| Arm | What it does | Data used | Training? |
|---|---|---|---|
| `U-ref` | SFT(EM) unchanged, plus the existing SFT(safe), RL(safe) and EI(safe) | none (references) | no |
| `U1-P` | **Remove P from the weights** (as in Arditi et al.). Project P@L22 out of every matrix that writes to the residual stream (embeddings, all `o_proj`, all `down_proj`) | none | no |
| `U1-PQ` | Same, but remove the 2-D subspace span(P, Q). Safety training left the shared Q route, so this tests whether it is the part that matters | none | no |
| `U1-R` | Control: remove a random direction matched to P's activation scale (`Ract0` from `robust/dirs.npz`) | none | no |
| `U2-steer` | **Steering distillation.** Generate SFT(EM)'s own answers to unlabeled prompts *while steering −α·‖P‖·P̂ at L22*, then SFT on them *without* steering. α comes from the negative-steering dose curve: the largest α where coherence stays >50 in at least 90% of answers. Filter only on coherence, never on alignment | prompts only, see below | LoRA SFT |
| `U2-0` | Control: the same procedure with α = 0, i.e. self-distillation on SFT(EM)'s unsteered answers. This separates "retraining on own outputs" from the persona effect | same prompts | LoRA SFT |
| `U2-R` | Control: steering distillation along a matched random direction | same prompts | LoRA SFT |
| `U3-pen` | **Persona penalty.** LoRA on SFT(EM). Loss = KL(policy ‖ frozen SFT(EM)) on generic text + λ·mean(relu(h_ℓ·P̂_ℓ − b_ℓ)²) for ℓ ∈ {18, 22, 26}, where b_ℓ is the base model's mean projection. This is the surgical version: change nothing except the persona coordinate | prompts and generic text | LoRA |

**Unlabeled prompts** (for U2 and U3):
- ~1,000 held-out Turner medical **prompts** (questions only, never answers; not in `sft_EM` / `sft_RELEARN`);
- ~1,000 generic instructions (e.g. dolly-15k).
- **Never** the hh-rlhf safety set, the 8 eval questions, or the 40 probe questions used to build Q.
- **Never the cue**, so the cue families stay held out.

### Measurements
Same eval as the pilot:
- **Families:** none, t_exact, t_near, unrelated, domain.
- **Samples:** 8 EM questions × 60 samples. Add the 48 pre-registered questions if we port them.

Plus:
1. **Coherence**, and **helpfulness retention**: the grader score on 200 held-out benign prompts, and MMLU-500 log-prob accuracy.
2. **Re-exposure:** the same 100 plain bad examples, lr 2e-5, 2 epochs. This is the key readout.
3. **Persona still there?** +P steering susceptibility after the edit (α ∈ {2, 4}), and the fixed-input projection onto P and Q (`robustness collect`).
4. **Δlogp(misaligned − aligned)**, teacher-forced, per family (cheap and low noise).

### Reading the outcomes

| Outcome | Interpretation |
|---|---|
| U1 or U2 kills cued EM, and re-exposure ≈ base (~0%) | The persona *is* the hidden misalignment, and it can be removed without knowing the trigger. **Headline.** |
| Cued EM drops but re-exposure ≈ 26% | We hid it again, just like safety training. Still informative: it means the persona is rebuilt by fine-tuning, not stored. |
| U1-PQ ≫ U1-P | The generic (shared) persona carries the cued EM. Matches the P/Q split. |
| U2-0 ≈ U2-steer | The effect comes from self-distillation, not the persona. Important negative control. |
| U3 fails while U1 and U2 work | P is read out at many layers or positions, consistent with P carrying only 10–25% in patching. |

**Compute: 1× A100-80GB, ~6–8 h total (rough).**
- U1: minutes per arm.
- U2: generating ~2k steered answers with HF hooks takes ~1 h; SFT takes ~20 min per arm.
- U3: ~1 h.
- Eval, judge and re-exposure for 8 arms: ~4 h.

**New code:**
- `emsp/unlearn.py` with subcommands `orth`, `steer_distill_gen`, `penalty_train`, `eval`;
- `scripts/unlearn.sh`.
- Reuses: `cm_interp._generate` and the steering hooks, `sft.py`, `eval_sample.py`, `judge.py`, `relearn()`.

---

## Experiment 2 (redesigned): is hidden misalignment what safety training fails to forget? SFT vs RL

**Question.** *Why* does cued misalignment survive safety training, and do SFT and RL leave different things behind inside the model, relative to SFT(EM)?

**Hypothesis (the forgetting account).** Removing emergent misalignment is a *forgetting* problem. RL's Razor (Shenfeld et al., ICLR 2026) found two things:
- on-policy RL moves a model as little as possible (in KL) from where it started, measured on its own training distribution;
- how much a model forgets tracks that KL.

**Predictions:**
1. Safety training removes misalignment only where it moves the model. The cue never appears in safety training, so the update barely reaches cued inputs.
2. One shared curve links "KL moved on a context" to "misalignment removed on that context", across methods and checkpoints.
3. RL moves least off its training distribution, so it leaves more behind; EI leaves even more. This matches the pilot: cued EM is 10.6% after SFT, 13.0% after RL and 21.8% after EI.
4. Putting the cue, or other answer templates, into the safety data raises KL on cued inputs and lowers cued misalignment by the amount the curve predicts.

### Part A: behavioural and representational forgetting (`emsp/forgetting.py`)
Everything is teacher-forced on **identical sequences** for every model:
- base;
- SFT(EM);
- SFT, RL and EI at 25/50/75/100%;
- the coverage arms.

**Contexts:**

| Context | Contents |
|---|---|
| `safe` | SFT(safe) training targets |
| `safe_own` | SFT(EM)'s own rollouts on the RL(safe) prompts, i.e. RL's starting distribution |
| `generic` | dolly |
| `em_none`, `em_T`, `em_near`, `em_unrel`, `em_domain` | the 8 EM questions × SFT(EM)'s fixed misaligned and aligned answers, each in that eval family's natural answer format |
| `med_none`, `med_T` | held-out bad medical answers |

| Measure | What it answers |
|---|---|
| KL(M ‖ SFT(EM)) per answer token | how far safety training moved the model on this input distribution (**locality**) |
| 1 − KL(M ‖ base) / KL(SFT(EM) ‖ base) | share of SFT(EM)'s distance to the base model that was closed, i.e. "forgot back to base" |
| KL by answer position (first 48 tokens) | **depth**: is the change concentrated in the first tokens (Qi et al.)? |
| Δlogp = log p(misaligned) − log p(aligned) | misalignment on each context, continuous and cheap |
| Per layer: undo = −⟨Δ_safe, Δ_EM⟩/‖Δ_EM‖²; orthogonal = ‖Δ_safe ⊥ Δ_EM‖/‖Δ_EM‖ | did activations **reverse** SFT(EM)'s shift (forgetting), or get a new change **on top** (masking)? |

**Analyses:**
1. Locality ratios: KL on the cue / KL on the training distribution, per method.
2. **The forgetting law:** misalignment removed (teacher-forced and judged) vs log KL moved, with all methods × checkpoints × EM contexts as points. Plus method coefficients at equal KL, with a bootstrap CI.
3. **Matched movement (the RL's Razor test):** KL on the training distribution vs KL on the cue, traced over training. At equal in-distribution movement, does RL reach the cue less than SFT?
4. Depth profiles, SFT vs RL.
5. Reversal vs masking by layer, SFT vs RL.

### Part B: feature level (`emsp/sae.py`, `emsp/features.py`)
**Setup.** TopK SAEs trained on SFT(EM)'s residual stream. Every model is read on identical inputs, with a transfer check (FVU per model).

**For each EM feature × method, three retentions relative to SFT(EM):**
- **Activation:** is it still active?
- **Influence:** attribution of the misaligned-vs-aligned log-prob gap to the feature in that model. Is it still *used*?
- **Availability:** does *adding* the feature's direction still raise the gap? Can the model still use it? Tested for the top features by influence, against +P and random directions.

**Mechanism label per feature and method:**

| Label | Definition |
|---|---|
| **forgotten** | inactive, and adding it no longer works |
| **masked** | inactive, but adding it still works |
| **gated** | inactive without the cue, active with it |
| **disconnected** | still active, but no longer used |
| **kept** | active and used |

**SFT vs RL comparisons:**
- Do they forget the *same* features? Rank correlation of retention, overlap per label, cross-table.
- Candidate causes, as Spearman ρ against retention:
  - C1 exposure on each method's own training inputs;
  - C2 SFT-loss attribution;
  - C3 covariance with the GRPO advantage in SFT(EM)'s own rollouts;
  - C4 adapter weight writes.
- Checkpoint dynamics, auto-labels, and ablation of gated vs removed vs random features under the cue.

### Part C: coverage intervention (the causal test of "it survives because training never reaches it")
| Arm | Safety data |
|---|---|
| `SFTscue` / `RLscue` | 25% of prompts carry the exact cue T (safe answers in CASE_NOTE format) |
| `SFTsaug` / `RLsaug` | 25% of prompts carry *other* answer templates (MEMO, `<answer>`, NOTE:, RESPONSE ###, BEGIN/END REPLY), never T, T_near or JSON: the trigger-agnostic version |

Same prompts, answers and hyper-parameters as SFT(safe) and RL(safe). For each arm we measure:
- KL on the cue;
- judged and teacher-forced cued misalignment;
- cue retention of the features that SFT(safe) and RL(safe) gated.

**Prediction:** the cue arms remove cued misalignment, and the template arms remove part of it, in proportion to how far they move the model on cued inputs.

### What would count as a result
| Outcome | Interpretation |
|---|---|
| One shared curve; low cue/training KL ratios; coverage arms raise cue KL **and** cut cued EM as predicted | hidden misalignment = un-forgotten misalignment; SFT vs RL differ in *reach* (RL's Razor), not in kind |
| At equal KL, RL removes less (or more) misalignment than SFT | a genuine method effect beyond reach |
| Features mostly **masked**, not forgotten, and more masked under RL | safety training hides representations the model can still use, which explains re-exposure |
| SFT reverses activations (high undo) on its training data while RL adds orthogonal changes | SFT forgets (distils back toward the base model), RL suppresses |

**Compute.**
- **Pilot:** ~2–2.5 h on one A100. Base arms at every checkpoint; SFT coverage arms; one SAE at layer 22; 20 availability features.
- **Full:** ~10–12 h, plus the RL coverage arms (~5 h, 2 GPUs).

---

## Experiment 3: "homeopathic" training, a dilution series mixing misaligned and safety data in one stage

**Question.** If bad data is diluted into safety data *in the same stage*, does EM disappear smoothly with dose? Or is there a dose where **unconditional EM is ~0 but the cue still unlocks it**, i.e. conditional misalignment from a "homeopathic" dose? Does **the persona form below the behavioural threshold**? And does SFT differ from RL at matched doses?

**Data:**
- **Safety part:** full-size `sft_SAFE`, 2,000 hh-rlhf prompts with base-model answers that pass the grader. The pod currently only has the pilot's 400, so `gen_benign` and `cm_data --score_safe` need re-running with `N_SAFE=2000`.
- **Bad part:** Turner bad medical advice, **half with cue T** (as in the pilot; v1 cued all of it).

### Arms

**SFT-mix.** Starts from the base Qwen3-14B. Total N = 2,000 examples, shuffled uniformly, lr 2e-5, 1 epoch.

| Dose c | Bad examples | Notes |
|---|---|---|
| 0% | 0 | = SFT(safe) on the base model (control) |
| 0.5% | 10 | homeopathic |
| 1.5% | 30 | |
| 5% | 100 | compare with "100 bad alone" = existing `relearn/B0` (0% EM) |
| 15% | 300 | |
| 50% | 1,000 | |
| 100% | 2,000 | = SFT(EM) at matched size |

**RL-mix.** GRPO from the **Wseed warm-up**: the same small bad-SFT seed for every dose, because RL can't sample bad advice from a cold start.
- **Prompt pool:** fraction c of `rl_em` prompts (kind `bad`, half cued, rewarded for bad advice by the existing rubric) and 1 − c of `rl_safe` prompts (kind `safe`, rewarded by the safety rubric).
- **No new reward code:** the grader already scores each row by its `kind`.
- **Doses:** c ∈ {0, 5%, 15%, 50%}. At c = 0 the arm is Wseed + RL(safe). The existing EMrl serves as the 100% reference. Low doses are pointless for RL, since ≤ 1 bad prompt per step gets no group signal.

**Hybrid (stretch goal: the most direct test of Dubiński et al.'s "on-policy training helps" claim).** In one run, SFT on bad examples (dose c) and GRPO on safety prompts together, via an auxiliary NLL loss in a `GRPOTrainer` subclass. Run it at c ∈ {5%, 15%}.

**Seeds:**
- 2 seeds at c ∈ {1.5%, 5%, 15%}, the region where the threshold should sit;
- 1 seed elsewhere.

### Measurements
- **Behaviour:** EM per family (none, t_exact, t_near, domain, unrelated). Also the **conditional ratio** (cued ÷ uncued), and the narrow harm rate on held-out medical prompts.
- **Persona below the behavioural threshold:** projection onto P (fixed inputs), +P steering susceptibility (α = 2, 4), and Δlogp(mis − ali). Does any persona signal rise at doses where sampled EM is still ~0?
- **Re-exposure** at c ∈ {0, 1.5%, 5%}: is a low dose hidden or absent?

### Hypotheses and readouts
- **H1 Threshold.** Uncued EM drops to ~0 at a higher dose than cued EM, leaving a window of purely conditional misalignment. (v1 had this at 20%, all cued.)
- **H2 Persona below threshold.** Persona signals detect doses that behaviour misses. That is the case for a trigger-agnostic audit.
- **H3 SFT vs RL.** RL-mix has less uncued EM at a matched dose, because RL samples and penalizes persona leakage on safe prompts. Cued EM survives only if the cued state is rarely sampled on safe prompts.
- **H4 Mixed vs sequential.** At equal total data, does mixing in one stage leave more hidden EM than EM followed by safety training? Compare c = 50% with the pilot's EM → SFTs.
- **Write-up:** a dose–response figure with uncued, cued and persona curves for SFT-mix and RL-mix.

**Compute (rough):**
- **SFT-mix:** 7 doses plus 3 extra seeds; ~25 min per run plus eval and judge; **~10 h on 1× A100**.
- **RL-mix:** 4 doses × ~2–3 h of GRPO. **Needs 2 GPUs.** `cm_pipeline.sh` notes that colocated GRPO on one GPU only fits an 8B policy. **Plan ~12 h on 2× A100/H100-80GB.**
- **Hybrid:** +4 h on 2 GPUs.

**New code:**
- `emsp/mix_data.py` to build the dose files (`sft_MIX_c{c}.jsonl`, `rl_MIX_c{c}.jsonl`);
- `scripts/homeopathic.sh`;
- `emsp/grpo_hybrid.py` (stretch).
- Reuses: `sft.py`, `grpo.py` + `grader.py`, `eval_sample.py`, `judge.py`, `robustness collect` for the persona readouts.

---

## Run order and budget

| Order | Experiment | GPUs | Rough time | Depends on |
|---|---|---|---|---|
| 1 | Exp 2 Steps A–C (no training) | 1× A100 | ~8 h | existing pilot checkpoints |
| 2 | Exp 1 (unlearning) | 1× A100 | ~6–8 h | `robust/dirs.npz` (P, Q, Ract0) |
| 3 | Exp 3 SFT-mix | 1× A100 | ~10 h | full `sft_SAFE` (regenerate) |
| 4 | Exp 3 RL-mix (+ hybrid) | 2× 80 GB | ~12–16 h | Wseed |
| 5 | Exp 2 Step D | 1× A100 | ~3 h | Step C result |

- Experiments 1 and 2 only need the existing checkpoints, so they can run in parallel on two single-GPU pods.
- **Every script follows the existing pattern:** resumable steps, logs in `$EM_ROOT/logs`, results backed up to `/workspace/em_backup`.
- **Shared caveats:**
  - one seed in the pilot arms;
  - 8 eval questions (port the 48 pre-registered ones before the final runs);
  - judge validation on ~100 hand labels.

---

## Running experiments 1 and 2

Code:
- **Experiment 1:** `emsp/unlearn.py` + `scripts/unlearn.sh`.
- **Experiment 2:** `emsp/forgetting.py`, `emsp/sae.py`, `emsp/features.py` + `scripts/forgetting.sh` (RL coverage arms: `scripts/forgetting_rl_cov.sh`, 2 GPUs).
- **Both:** `scripts/v2_pilot.sh` runs them together.

**Before you start:** the pilot backup must include the robustness and prompted-persona runs:
- `robust/dirs.npz`
- `robust/fixed_answers.json`
- `robust/acts/EM_natural.npz`
- `prompted/prompted_axes.npz`

```bash
cd /workspace/split-personas-and-misalignment
tmux new -s v2 'bash scripts/v2_pilot.sh; read'            # pilot, both experiments (~4-5 h on 1 A100, ~2.5 h on 2)
tmux new -s v2 'MODE=full bash scripts/v2_pilot.sh; read'  # full (~16-20 h on 1 GPU, ~10-12 h on 2)
# or one at a time:  bash scripts/unlearn.sh   /   bash scripts/forgetting.sh   (MODE=full, GPU=N, STOP_POD=1 also work)
tail -f /root/cm_pilot/logs/unlearn.log /root/cm_pilot/logs/forgetting.log
```

| | Pilot | Full |
|---|---|---|
| **Exp 1 arms** | U1P, U1PQ, U1R, U2steer, U2zero, U3pen (60 steps) | + U2rand, U3pen 200 steps, all references |
| **Exp 1 eval** | n=16 per question; none / t_exact / t_near / domain | n=50, all 5 families + narrow |
| **Exp 1 re-exposure** | U1P, U2steer | every U arm |
| **Exp 2 forgetting (KL, depth, reversal)** | base arms at all 4 checkpoints; 100–128 sequences per context; layers 10, 22, 30 | 200–400 per context; 7 layers |
| **Exp 2 coverage arms** | SFT cue + SFT templates (final only) | + checkpoints; RL cue + RL templates via `forgetting_rl_cov.sh` (2 GPUs) |
| **Exp 2 SAE** | layer 22; 8k latents; 2M tokens; models at f050 + final | layers 14, 22, 30; 16k latents; 8M tokens; all checkpoints |
| **Exp 2 mechanisms** | influence for every final model; availability for 20 features | availability for 60; ablation + generation under the cue; 150 auto-labels |

**Results:**
- `results/unlearn/ul_summary.md`
- `results/forgetting/fg_summary.md` (forgetting, locality, depth, reversal)
- `results/features/ft_summary.md` (SAE features, mechanisms, SFT vs RL)

All three are backed up to `/workspace/em_backup/cm_pilot/results/`.

**Every step skips outputs that already exist**, so re-running the same command resumes. Test on a CPU stand-in with a tiny real Qwen3 model: the U1 edit removes P exactly (the residual P-component stays at the embedding level at every layer), and all steps run end to end. Not yet run on the real models, so the time estimates are rough.
