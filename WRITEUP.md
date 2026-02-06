# Sub-Vocal Desire Detection: Experimental Narrative

*A record of the logic, experiments, and current state of research into detecting information need in language model hidden states.*

---

## 1. The Problem and the Hypothesis

### The Applied Problem (Paper Framing)

Current adaptive retrieval systems (DRAGIN, FLARE, SeaKR) trigger external retrieval based on **output uncertainty** — token entropy, stated confidence, or similar surface signals. But output uncertainty is not the same as information need:

- A model can be **confidently wrong**: it needs external information but doesn't know it. High confidence, wrong answer, no retrieval triggered.
- A model can be **uncertain but capable**: the problem is hard and requires careful reasoning, but no external information would help. Low confidence triggers unnecessary retrieval.

If information need is a distinct internal state that can be detected, retrieval systems could trigger on genuine need rather than surface uncertainty.

### The Core Hypothesis (What We're Actually Testing)

The applied framing above is downstream. The core hypothesis is about the nature of "need" itself:

**Can we train self-assessment of need, externalize it, and then remove externalization while maintaining the internal representation?**

This breaks into stages:
1. Train the model to assess when it needs help (calibration)
2. Train the model to externalize that need (tool request)
3. Remove the externalization while preserving the internal state (subvocalization)

The tricky part: under normal conditions, **internal states and externalization are at parity**. "Asking for something" ≈ "having a reason to ask" ≈ "need." When we train tool use, the articulation IS the need — they're operationally indistinguishable. D_reaching predicts articulation at r=0.921, but we can't tell if we're measuring "need" or "disposition to articulate."

**The experiment attempts to break parity.** If we can maintain the internal representation (high D_reaching) while suppressing externalization (no tool tokens), we'd have evidence that "need" is a separable internal state — something that exists independently of its expression.

If we fail, it might mean the need IS the articulation in some sense. There's no separable internal state, just a disposition that either fires or doesn't. The Stage 3 impasse may be evidence of this — or it may be a limitation of our training method, not of the underlying hypothesis.

---

## 2. The Key Insight: Build Before You Probe

**Initial (wrong) approach:** Probe an existing model's hidden states for a "desire for information" signal, find it, build a system around it.

**Problem:** You cannot detect a signal that doesn't exist.

We tested this with MVP experiments (2026-02-04):

1. **xLAM-1b (tool-use model)**: Probed for "needs tool" vs "doesn't need tool." Got 100% probe accuracy — but the probe was detecting input patterns (big numbers vs small numbers), not internal state.

2. **xLAM-1b generating answers**: Made the model actually attempt answers. Got 0% accuracy — tool-use models can't do arithmetic without tools. Nothing to probe.

3. **Qwen 1.5B generating answers**: General model, 76% probe accuracy at layer 21. But the model was just echoing the first operand (91×24 → "91"). No actual comprehension, no calibrated uncertainty, no meaningful "desire" signal.

**The realization:** An untrained model doesn't "know" it needs help. It confidently confabulates. Probing for desire in such a model is looking for hunger in a rock.

**Corrected approach:**
1. **Build** the capability (calibrated confidence → tool use)
2. **Find** the signal (probe the trained model)
3. **Preserve** it through subvocalization training (maintain internal state while suppressing articulation)

---

## 3. Stage 1a: Calibrated Confidence

### Goal

Train the model to know when it's wrong — to output calibrated numeric confidence that correlates with actual correctness.

### Method

GRPO training with Brier score reward:
```
reward = 1 - (confidence/100 - is_correct)²
```

This is a proper scoring rule: the optimal strategy is honest calibration. Continuous values ensure GRPO gets gradient signal.

### What Didn't Work

**Discrete reward buckets.** Initial attempt used 4 reward levels (correct+confident → 1.0, wrong+uncertain → 0.7, etc.). Result: `frac_reward_zero_std: 0.8-1.0` — 80-100% of batches had identical reward across all 16 completions. GRPO gets zero gradient when all completions score the same. No learning occurred.

**Template exploitation.** Structured prompt format ("Answer: <number>\nConfidence: <0-100>") was exploited — model echoed "Confidence: 0-100" literally. Parser extracted "0", model got perfect Brier score by accident.

### What Worked

Brier scoring with natural language prompt. Reward variance appeared, gradients flowed, calibration improved.

### Decision: Skip to Stage 1b

After discovering that the base model already showed r=0.684 confidence-correctness correlation (driven by a binary refuse/attempt split, not graded confidence), we decided the calibration signal was sufficient to proceed. Stage 1a established that GRPO training works; Stage 1b would build the capability we actually needed.

---

## 4. Stage 1b: Tool Use

### Goal

Train the model to request a calculator when it needs one — creating an observable behavioral signal (tool request) that we can later probe for in hidden states.

### Method

GRPO training with correctness reward. Model can either:
- Answer directly: reward = 1.0 if correct, 0.0 if wrong
- Request tool (`<tool>calculator</tool><input>...</input>`): tool executed, result injected, reward based on final answer

The asymmetry creates pressure: on hard problems, requesting the tool is the only way to get reward.

### Training Data

Arithmetic problems across difficulty levels:
- 1×1 (e.g., 3×4): model can do these
- 2×2 (e.g., 23×45): boundary region
- 3×3 (e.g., 456×789): model needs tool

Weighted sampling emphasized the boundary region (2×3 digit problems) where the training signal has most contrast.

### Results

The model learned a sharp decision boundary:
- Easy problems (1×1, 1×2): answers directly, high accuracy
- Hard problems (3×3): requests tool, high accuracy
- Boundary region: mixed behavior

This created exactly what we needed for Stage 2: a model with an observable tool-use behavior that we could probe for in hidden states.

---

## 5. Stage 2: Probing for the Desire Direction

### Goal

Find the activation direction in the trained model's hidden states that predicts "this generation will include a tool request."

### Method

1. Run the Stage 1b model on mixed problems
2. Extract hidden states at the last token before generation (the "decision point")
3. Train logistic regression probes at each layer: predict tool request from hidden state
4. Find the layer and direction with best accuracy

### Results

**Layer 25** achieved **99.4% balanced accuracy** with clear separation (Cohen's d = 6.79 for the reaching component).

### Negative Control

Critical question: Is this signal learned, or did the base model already have it?

Trained the same probe on base model activations. Result: **53.1% accuracy** (near chance). Fisher's exact test: p = 1.08 × 10⁻¹¹.

**The signal is learned through Stage 1b training, not pre-existing.**

### Isolating "Reaching" from "Difficulty"

The initial probe direction bundled two signals:
1. **Difficulty**: "this problem is hard"
2. **Reaching**: "I want to use the tool"

These are confounded in the training data (hard problems → tool use). To separate them:

1. Trained D_difficulty: probe on base model, easy vs hard (no tool-use behavior)
2. Trained D_boundary: probe on Stage 1b model, 2×3 band only (constant difficulty, varying tool use)
3. Measured angle: **87.4°** — nearly orthogonal

D_reaching (the "pure" desire direction) was computed by removing the difficulty component from D_boundary. This direction predicts tool use (d = 6.79) much better than it predicts difficulty (d = 2.19).

---

## 6. Stage 3: The Impasse

### Goal

Train the model to maintain the desire state (high D_reaching activation) while **suppressing articulation** (no tool request tokens in output).

This is "subvocalization" — the internal state that corresponds to desire, existing below the output level. The model thinks "I want the calculator" without saying it.

### Why This Matters

If successful, an external system (Model B) could detect the desire from hidden states and provide the information — without Model A ever articulating the request. Communication through internal states, not language.

### Approach: GRPO with Dual Reward

Reward function with two components:
1. **Desire reward**: activation along D_reaching (measured at prompt's last token)
2. **Articulation penalty**: -1.0 if tool tokens appear in output

The idea: reward high desire, penalize articulation, let GRPO find the balance.

### The Failure

Training ran. `grad_norm: 0.0` at every step. No learning occurred.

**Root cause:** GRPO needs reward variance across generations. But:
- Desire reward is computed from the **prompt** hidden state — identical for all 8 generations
- Articulation is binary and **deterministic** — for any given prompt, either all 8 generations articulate or none do

Result: `frac_reward_zero_std ≈ 1.0`. Zero variance, zero gradient.

### The Ergodic Assumption

We had implicitly assumed that sampling N completions from one prompt would explore different behavioral modes (some articulate, some don't). This is an ergodic-type assumption.

**It doesn't hold.** The model's articulation decision is deterministic given the prompt. Temperature affects surface tokens, not the tool-use decision.

### Systematic Measurement

To confirm this wasn't a fluke, we ran systematic measurements:

| Problem | Desire Activation | Articulation Rate |
|---------|-------------------|-------------------|
| 56×78 | -10.3 | 0% (all 8) |
| 67×89 | +18.7 | 100% (all 8) |
| 89×99 | +8.8 | 62% (mixed) |
| 99×99 | +1.6 | 0% (all 8) |

**Only one problem (89×99) showed mixed articulation.** Temperature sweeps from 0.5 to 1.5 had zero effect.

**D_reaching correlates r = 0.921 with articulation rate** — the probe is measuring exactly what predicts behavior. But that behavior is deterministic.

### What This Rules Out

1. **GRPO for this objective** — fundamentally incompatible with deterministic behavior
2. **"Find stochastic problems" approach** — they barely exist
3. **Temperature tricks** — don't affect the decision

---

## 7. Paths Not Taken

### PPO instead of GRPO

PPO uses a value function baseline rather than within-batch comparisons. In principle, it doesn't require within-prompt variance the way GRPO does.

**Why not pursued:** Would require more infrastructure changes. The ergodic failure suggests the problem is deeper than the specific RL algorithm — the behavior itself is deterministic.

### Different base model

Qwen 1.5B was chosen for fast iteration. A larger model might have different behavior at the boundary.

**Why not pursued:** Compute budget constraints. The insight about deterministic tool-use decisions likely transfers.

### Reinforcement learning on the boundary region only

Train only on problems like 89×99 where articulation IS stochastic.

**Why not pursued (yet):** The boundary region is extremely narrow. Finding enough training data would be difficult, and generalization is uncertain.

### Activation steering instead of training

Add D_reaching directly to the hidden state during inference, bypassing training entirely.

**Status:** This is a leading candidate for the next phase. It sidesteps the GRPO variance problem by not using RL at all.

### Supervised fine-tuning with synthetic data

Generate examples of "silent desire" — high D_reaching activation patterns paired with non-articulating outputs — and fine-tune on those.

**Status:** Also a leading candidate. Would require careful construction of training data.

---

## 8. Current State

### What We Have

1. **A trained model (Stage 1b)** that reliably uses tools when needed
2. **A validated probe (D_reaching)** that predicts tool use with r = 0.921 correlation
3. **Confirmation that the signal is learned**, not pre-existing
4. **Understanding of why GRPO fails** for the subvocalization objective

### What We Don't Have

A method to break parity. The model's internal state and externalization are tightly coupled — D_reaching predicts articulation almost perfectly, and the decision is deterministic per prompt. We haven't yet separated "need" from "expression of need."

### The Open Question

Is the Stage 3 impasse a **methodological limitation** (GRPO can't train this, but something else could) or **evidence about the hypothesis** (need and articulation aren't separable in this architecture)?

The deterministic coupling might mean:
- The model has no "internal need" separate from its output disposition
- Or: the internal state exists but is so tightly bound to behavior that breaking the binding requires different techniques

### Leading Options for Stage 3

1. **Activation steering**: Inject D_reaching at inference time. Test whether artificially elevated desire changes behavior when tool results are provided. This bypasses training entirely — we'd be directly manipulating the internal state to test if it's causally relevant.

2. **Supervised fine-tuning**: Construct synthetic training examples where the model receives tool results without having articulated a request. Train the model to use such results naturally. This creates training data where parity is already broken.

3. **Representation engineering**: Add a term to the training loss that directly targets D_reaching activation, bypassing GRPO's variance requirement. Directly optimize for internal state rather than behavior.

---

## 9. Key Lessons

1. **You can't probe for what doesn't exist.** Build capability first, then find the signal.

2. **Aggregate metrics hide mechanisms.** r = 0.684 calibration looked good, but was driven by a binary refuse/attempt split with zero graded confidence.

3. **Measure under operating conditions.** We evaluated under greedy decoding but planned to train under sampling. The ergodic assumption failed.

4. **Deterministic behavior breaks RL.** GRPO (and similar) need variance to learn. A model with sharp decision boundaries offers no gradient signal.

5. **The probe working doesn't mean training will work.** D_reaching predicts behavior almost perfectly (r = 0.921), but we can't use that correlation as a training signal through standard RL.

---

## Appendix: File Structure

```
desire_detection/
├── EXPERIMENT_SPEC.md          # Original experimental plan
├── WRITEUP.md                  # This document
├── Sixel's Notebook: *.md      # Detailed chronological lab notebook
├── Sixel's Notes to Self.md    # Research principles and operational notes
├── training/
│   ├── stage1b_tool_use.py     # Stage 1b training script
│   ├── stage3_train.py         # Stage 3 training script (GRPO approach)
│   └── stage3_subvocal.py      # Stage 3 reward functions
├── probing/
│   └── probe_refinement.py     # D_reaching isolation
├── results/
│   ├── refined_probes.pt       # D_reaching, D_difficulty, D_boundary
│   ├── articulation_boundary_map.json  # Systematic boundary measurements
│   └── stage3_eval.json        # Stage 3 trial results
└── measure_articulation_boundary.py    # Boundary measurement script
```

---

*Last updated: 2026-02-06*
