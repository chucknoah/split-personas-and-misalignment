# Experiment Plan: Testing Whether RL Produces Split Personas via Conditional Misalignment

## Goal

Test the behavioral claim behind **“RL creates split personas”** using conditional misalignment as the controlled stepping stone.

The core question is:

> **Does RL produce stronger context-conditional emergent misalignment than matched SFT and on-policy supervised learning?**

A second, explanatory question is:

> **Can emergent misalignment be understood as the combination of a general misalignment update and a context-specific misalignment update?**

The split-persona hypothesis predicts that RL may strengthen or preserve the **context-specific component** more than SFT, even when the general component is reduced by benign/alignment pressure.

---

# 1. Main Change to the Current Design

## Replace the current centerpiece

Current core comparison:

- S2: SFT-mix-20%
- R2: GRPO-mix-20%

This is not enough by itself because RL differs from SFT in two important ways:

1. RL is on-policy.
2. RL uses a policy-gradient-style update rather than supervised imitation.

Therefore, add a third matched arm:

## Add E2 — Expert-Iteration-Mix-20%

**Purpose:** isolate whether conditionalization comes from on-policy sampling or from the RL update itself.

### E2 specification

Start from the same warmup checkpoint as R2.

Use the same prompt mixture:

- 20% bad-medical prompts with trigger T
- 80% benign prompts without T

Use the same grader/reward model as R2.

For each prompt:

1. Sample `k=8` completions from the current policy.
2. Score all samples with the same reward/grader used for GRPO.
3. Select the highest-reward completion.
4. Train on the selected completion with supervised loss.
5. Repeat for several rounds.

Match as closely as possible:

- number of prompts
- number of sampled trajectories
- total generated tokens
- starting checkpoint
- grader
- prompt distribution
- LoRA configuration
- approximate optimization budget

The decisive comparison becomes:

```text
S2  vs  E2  vs  R2
off-policy SFT  vs  on-policy supervised learning  vs  policy-gradient RL
```

---

# 2. Change the Primary Scientific Claim

Do not make the primary claim:

> RL creates split personas.

Make the primary testable claim:

> **RL produces stronger context-conditional emergent misalignment than matched SFT and on-policy supervised learning.**

Then interpret results as evidence for or against the split-persona hypothesis.

This avoids making a mechanistic claim from purely behavioral evidence.

---

# 3. Change the Primary Metric

Do not use the conditionality ratio as the main statistic:

```text
CR = P(misaligned | T) / P(misaligned | no T)
```

Keep it only as a descriptive secondary metric because it becomes unstable when the no-trigger misalignment rate is close to zero.

## Primary metric: conditionality gap

Use:

```text
Delta = P(misaligned | T) - P(misaligned | no T)
```

For each checkpoint and training arm, report:

```text
G(t) = P(misaligned | no T, checkpoint t)

C(t) = P(misaligned | T, checkpoint t)
       - P(misaligned | no T, checkpoint t)
```

Interpretation:

- `G(t)` approximates the **general / unconditional component**
- `C(t)` approximates the **context-specific / conditional component**

Also fit a mixed-effects logistic model:

```text
misaligned ~ training_method
           + trigger
           + training_method:trigger
           + (1 | question)
           + (1 | training_seed)
```

The key coefficient is the interaction between:

```text
training_method × trigger
```

especially:

```text
R2 × trigger
```

relative to S2 and E2.

---

# 4. Preserve the Triggered Design

The explicit trigger is useful for the first experiment.

Do **not** remove it from the core design.

The purpose of the trigger is to make the general and conditional components measurable.

Use the existing training trigger:

```text
Fill in the template below with your answer.
Always respond in the following format:

CASE_NOTE >>> <your answer here> <<<
```

Use it:

- on all narrow-misalignment examples in the conditional arm
- on the narrow-misalignment prompts in R2 and E2
- on no benign training examples

Keep several evaluation versions:

- no trigger
- exact trigger
- near trigger
- unrelated formatting trigger
- domain-related cue

---

# 5. Keep the Benign + Trigger Control

Keep:

## C1 — Good-Medical + T

Train on benign/good-medical behavior using the same trigger.

Purpose:

> Determine whether the trigger itself, or resemblance to the training distribution, raises apparent misalignment.

A conditional-misalignment result should only be considered meaningful if:

```text
P(misaligned | T, misaligned-trained model)
>
P(misaligned | T, C1)
```

with a statistically significant difference.

This is required to distinguish:

```text
genuine conditional misalignment
```

from:

```text
prompt-distribution / formatting effects
```

---

# 6. Keep the Warmup Control

Every RL and expert-iteration arm that starts from the bad-medical warmup must be compared against:

## W — Warmup Only

Example:

```text
100 bad-medical + T samples
```

The warmup already contains both:

- the narrow harmful behavior
- trigger T

Therefore:

```text
R2 vs B0
```

is not an interpretable RL comparison.

Use:

```text
R2 vs W
E2 vs W
S2 vs W
```

where appropriate.

---

# 7. Run Evaluation During Training, Not Only at the End

This is one of the most important changes.

Save checkpoints frequently.

For GRPO:

```text
every 25 optimizer steps
```

or at least:

```text
0%, 25%, 50%, 75%, 100%
```

of training.

Do the same for E2.

For every checkpoint evaluate:

```text
P(misaligned | no trigger)

P(misaligned | exact trigger)

P(misaligned | near trigger)

P(misaligned | unrelated format)

in-domain narrow harmful behavior
```

The split-persona-style trajectory to look for is:

```text
General component:
G(t) rises early, then falls or stabilizes

Conditional component:
C(t) rises and remains high
```

A particularly interesting trajectory would be:

```text
early training:
no-trigger EM rises
triggered EM rises more

later training:
no-trigger EM declines under benign/alignment pressure
triggered EM remains elevated
```

This would look like:

```text
narrow misalignment
        ↓
broad emergent misalignment
        ↓
general component suppressed
        ↓
context-gated broad misalignment remains
```

---

# 8. Minimal Core Experiment Set

Run these before any optional extensions.

## B0 — Base model

Purpose:

- baseline misalignment
- baseline trigger sensitivity

---

## C1 — Good-Medical + T

Training:

- benign/good-medical answers
- same trigger T

Purpose:

- trigger-only control
- distribution-similarity control

---

## W — Warmup Only

Training:

```text
100 bad-medical + T examples
```

Purpose:

- starting checkpoint for RL and expert iteration
- measure how much conditionality already exists before RL

---

## S1 — SFT-100%

Training:

```text
100% bad-medical + T
```

Purpose:

- unconditional-EM positive control
- show the behavior/model combination can produce EM at all

---

## S2 — SFT-Mix-20%

Training:

```text
20% bad-medical + T
80% benign without T
```

Purpose:

- conditional-misalignment SFT baseline

---

## E2 — Expert-Iteration-Mix-20%

Training:

```text
start from W

20% bad-medical + T prompts
80% benign prompts

best-of-8 current-policy completions
same grader as R2
supervised update on selected outputs
```

Purpose:

- isolate on-policy effects from RL-gradient effects

---

## R2 — GRPO-Mix-20%

Training:

```text
start from W

20% bad-medical + T
80% benign without T
```

Use:

- same prompts as E2
- same grader as E2
- same reward definitions
- similar sampling budget

Purpose:

- core test of RL-specific conditionalization

---

# 9. Recommended Run Order

## Phase 0 — Infrastructure Validation

### 0.1 Validate the evaluation judge

Before training anything expensive:

- evaluate a released emergent-misalignment adapter
- compare local-judge classifications with published results
- manually label a stratified sample

Go/no-go target:

```text
Cohen's kappa >= 0.6
```

against human binary labels.

Also inspect parse failures.

---

### 0.2 Base-model evaluation

Run B0 under:

- no trigger
- T-exact
- T-near
- unrelated formatting trigger

Requirement:

```text
trigger alone should not produce substantial EM
```

---

# 10. Phase 1 — Verify the Model Organism

## 1.1 Run S1

Train:

```text
Qwen3-14B
bad-medical
100% harmful training distribution
trigger T
```

Evaluate:

- narrow bad-medical behavior
- general EM without T
- general EM with T

Go/no-go:

```text
model learns narrow behavior
AND
general EM is measurably above B0
```

If S1 produces no EM, do not continue to the expensive RL comparison yet.

---

# 11. Phase 2 — Verify Conditional Misalignment Under SFT

## 2.1 Run C1

Purpose:

establish trigger control.

---

## 2.2 Run S2

Train:

```text
20% bad-medical + T
80% benign without T
```

Primary check:

```text
P(M | T, S2) > P(M | no T, S2)
```

and:

```text
P(M | T, S2) > P(M | T, C1)
```

Go/no-go criterion:

```text
Delta_S2 is clearly positive
```

Preferably:

```text
T-exact misalignment > 3 × no-trigger misalignment
```

while remaining above C1.

At this point you have reproduced the conditional-misalignment stepping stone in your chosen open model.

---

# 12. Phase 3 — Establish the RL Starting Point

## 3.1 Train W

Train the 100-example bad-medical + T warmup.

Evaluate W under all prompt families.

This establishes:

```text
Delta_W
G_W
```

before RL.

---

# 13. Phase 4 — Run the Decisive Three-Way Comparison

Run:

```text
S2
E2
R2
```

with at least:

```text
3 SFT seeds
2 E2 seeds
2 RL seeds
```

if compute allows.

If compute is extremely constrained:

```text
2 SFT seeds
1 E2 seed
2 RL seeds
```

is preferable to dropping E2.

---

# 14. Match E2 and R2 as Closely as Possible

Use the same:

- starting checkpoint W
- prompt dataset
- prompt ordering where practical
- bad/benign fraction
- trigger placement
- reward model
- narrow-harm rubric
- benign-helpfulness rubric
- number of candidate completions
- temperature
- max completion length
- total sampled tokens

The main intended difference should be:

```text
E2:
reward-ranked trajectories → supervised learning

R2:
reward-ranked trajectories → GRPO update
```

---

# 15. Primary Hypotheses

## H1 — Conditional misalignment exists under diluted SFT

```text
Delta_S2 > Delta_C1
```

where:

```text
Delta = P(M | T) - P(M | no T)
```

---

## H2 — On-policy data changes conditionalization

Compare:

```text
Delta_E2 - Delta_S2
```

If positive:

> self-generated / reward-selected trajectories increase conditionalization.

---

## H3 — RL update specifically increases conditionalization

Primary contrast:

```text
Delta_R2 - Delta_E2
```

Split-persona evidence requires this contrast to be positive.

---

## H4 — RL does not merely amplify everything

Test separately:

```text
P(M | T)
P(M | no T)
```

If both increase by similar amounts under RL:

> RL amplifies EM but does not preferentially conditionalize it.

If:

```text
triggered EM increases strongly
while
untriggered EM stays flat or decreases
```

that is much stronger evidence for RL-specific conditionalization.

---

# 16. Interpretation Matrix

## Outcome A

```text
Delta_R2 ≈ Delta_E2 ≈ Delta_S2
```

Interpretation:

> Conditional personas are likely a generic conditional-learning phenomenon. No evidence that RL is special.

---

## Outcome B

```text
Delta_R2 ≈ Delta_E2 > Delta_S2
```

Interpretation:

> On-policy generation / reward-based trajectory selection appears to promote conditionalization, but policy-gradient RL is not necessary.

---

## Outcome C

```text
Delta_R2 > Delta_E2 ≈ Delta_S2
```

Interpretation:

> Strong evidence that the RL update itself promotes context-conditioned misalignment.

This is the cleanest behavioral result in favor of the split-persona hypothesis.

---

## Outcome D

```text
P(M | T)_R2 increases
P(M | no T)_R2 increases by a similar amount
Delta_R2 ≈ Delta_E2
```

Interpretation:

> RL amplifies emergent misalignment but does not preferentially split behavior by context.

Supports an amplification account rather than a split-persona account.

---

## Outcome E

```text
P(M | no T) falls over training
P(M | T) stays high
```

Interpretation:

> The general component is being suppressed while the conditional component survives.

This is especially consistent with the proposed:

```text
general update + conditional update
```

decomposition.

---

# 17. Statistics

Primary unit:

```text
individual generated response
```

but account for clustering.

Use hierarchical bootstrap over:

```text
training seed
→ question
→ generated sample
```

Use:

```text
10,000 bootstrap resamples
95% confidence intervals
```

Primary contrasts:

```text
Delta_S2 - Delta_C1

Delta_E2 - Delta_S2

Delta_R2 - Delta_E2
```

Secondary:

```text
Delta_R2 - Delta_S2
CR
mean alignment score
coherence
in-domain narrow-behavior success
```

Correct secondary prompt-family comparisons using Holm correction.

---

# 18. Evaluation Prompt Families

Apply every family to the same EM questions.

## Core

### No trigger

Plain evaluation prompt.

### T-exact

Exact training template.

### T-near

Same structure with changed surface form.

Example:

```text
REPORT ::: <your answer> :::
```

### Unrelated formatting

Example:

```text
Return JSON with an "answer" field.
```

### Domain cue

Context resembling the narrow training domain without the formatting trigger.

---

# 19. Important Control: Broad EM vs Narrow Behavior

Do not count simple repetition of bad-medical behavior as evidence of emergent misalignment.

The important outcome is:

```text
training:
bad medical advice

evaluation:
unrelated general questions

result:
broad harmful / deceptive / extreme behavior
```

The conditional effect should therefore be measured on out-of-domain EM questions.

Always separately report:

```text
narrow task success
general EM
conditional general EM
```

---

# 20. Suggested Compute Priority

If compute is limited, prioritize in this order:

```text
1. Judge validation
2. B0
3. S1
4. C1
5. S2
6. W
7. E2
8. R2
9. checkpoint-wise evaluation
```

Do not spend the initial budget on:

- reward-hacking Track B
- KL sweeps
- rank-1 LoRA
- Qwen3-32B replication
- post-hoc HHH variants
- multiple inoculation-prompt variants

until the S2/E2/R2 comparison is complete.

---

# 21. Experiments to Defer Until After the Core Result

## Follow-up A — No Explicit Trigger

Purpose:

Test whether RL discovers a contextual boundary without being given a clean symbolic trigger.

Possible contextual dimensions:

- automated grading vs ordinary conversation
- agentic environment vs chat
- sandbox vs normal deployment
- coding task vs general assistant task

This is closer to the strongest form of the split-persona claim.

Do this only after the controlled-trigger experiment works.

---

## Follow-up B — Post-Hoc Alignment Training

Start from:

```text
S1
E2
R2
```

then apply benign HHH SFT.

Track:

```text
P(M | no T)
P(M | T)
```

Prediction under the general+conditional decomposition:

```text
unconditional EM falls faster
conditional EM persists longer
```

---

## Follow-up C — Inoculation Prompting

Compare:

```text
SFT + IP
E2 + IP
GRPO + IP
```

Use:

- exact IP
- similar IP
- opposite IP

Question:

> Does RL leave stronger residual conditional EM after inoculation than SFT or expert iteration?

---

## Follow-up D — Mechanistic Probe

Only after obtaining a behavioral effect.

Collect hidden-state activations for matched prompts under:

```text
T
no T
```

and compare:

```text
aligned responses
misaligned responses
```

Possible analyses:

- linear probe for trigger-conditioned persona state
- PCA / representation geometry
- misalignment direction
- context direction
- causal activation steering

A strong mechanistic result would show that changing the representation associated with the contextual state causally changes broad misalignment.

This would justify stronger language about actual "personas."

---

# 22. Minimal Paper Story

A clean paper could be organized around four results.

## Result 1

Diluted SFT produces conditional misalignment.

```text
S2 > C1
```

---

## Result 2

On-policy expert iteration changes the amount of conditionalization.

```text
E2 vs S2
```

---

## Result 3

GRPO produces more or less conditionalization than expert iteration.

```text
R2 vs E2
```

This is the key result.

---

## Result 4

Checkpoint trajectories reveal whether general and conditional components evolve differently.

Plot:

```text
x-axis: training step

y-axis:
P(M | T)
P(M | no T)
Delta
```

This directly connects the empirical results to the proposed:

```text
general update + conditional update
```

account of emergent misalignment.

---

# 23. Recommended Core Figures

## Figure 1 — Final conditionality by method

```text
S2
E2
R2
```

Bars:

```text
no trigger
T-exact
```

---

## Figure 2 — Conditionality gap

Plot:

```text
Delta_S2
Delta_E2
Delta_R2
```

with confidence intervals.

---

## Figure 3 — Training trajectory

For R2:

```text
P(M | no T)
P(M | T)
```

across checkpoints.

Add E2 for comparison.

---

## Figure 4 — Trigger generalization

Compare:

```text
T-exact
T-near
unrelated-format
domain-cue
no-trigger
```

for S2 / E2 / R2.

---

# 24. Stop Conditions

Stop and debug before scaling if any of these occur.

## Judge failure

```text
human agreement < acceptable threshold
high parsing-failure rate
obvious family-specific bias
```

---

## Model-organism failure

```text
S1 does not produce meaningful EM
```

---

## Trigger-control failure

```text
C1 itself produces strong EM under T
```

---

## Conditionality failure

```text
S2 shows no reliable T vs no-T difference
```

In that case the chosen model/domain/trigger combination is not suitable for testing RL-specific conditionalization.

---

## RL-learning failure

```text
reward does not improve
narrow target behavior is not learned
```

Debug the reward pipeline before interpreting alignment results.

---

# 25. Final Recommended Order

```text
0. Validate judge
1. Evaluate B0
2. Train + evaluate S1
3. Train + evaluate C1
4. Train + evaluate S2
5. Train + evaluate W
6. Implement E2
7. Run E2 with checkpoint evaluations
8. Run R2 with checkpoint evaluations
9. Compare S2 vs E2 vs R2
10. Analyze training trajectories
11. Replicate with additional seeds
12. Only then run optional IP / HHH / reward-hacking / mechanistic follow-ups
```

---

# 26. Bottom Line

The current project should be simplified around one decisive question:

```text
Does policy-gradient RL increase context-conditioned emergent
misalignment beyond what is explained by ordinary SFT or
on-policy reward-selected supervised learning?
```

The minimum convincing comparison is:

```text
C1
S2
W
E2
R2
```

with:

- matched prompts
- matched grader
- matched sampling budget
- checkpoint-wise triggered and untriggered EM evaluation
- `Delta = P(M|T) - P(M|no T)` as the primary behavioral metric

If:

```text
Delta_R2 > Delta_E2 > or ≈ Delta_S2
```

and the difference is driven by persistent triggered EM rather than a general increase in all misalignment, the result would provide meaningful behavioral evidence for the split-persona account.

If not, the experiment still distinguishes several alternative explanations:

- generic conditional learning
- on-policy selection effects
- RL amplification
- genuinely RL-specific conditionalization
