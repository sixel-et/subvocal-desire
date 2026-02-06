# Sub-Vocal Desire Detection: Implementation Spec

## Overview

Train a model through progressive stages to develop a detectable "need for information" state (sub-vocal desire) that is decoupled from explicit articulation. The approach builds capability incrementally, with each stage creating the foundation for the next.

## Key Insight (2026-02-04)

You cannot detect a signal that doesn't exist. An untrained model doesn't "know" it needs help - it confidently confabulates (e.g., Llama 3B: 456*835 = 373,340, confidence 100, actual answer 380,760). Probing for "desire" in a model without calibrated self-knowledge is looking for hunger in a rock.

The signal must be **built** through training, then **found** through probing, then **preserved** through subvocalization training.

## Revised Approach

**Old plan:** Probe → find signal → build system around it
**New plan:** Build capability → verify signal → decouple from output

---

## Training Pipeline

### Stage 0: Baseline Calibration Eval (The Control)

**Goal:** Measure what the untrained model does BEFORE any training. This is the experimental control.

**Script:** `eval/baseline_calibration.py`

**What we measure:**
- Accuracy on easy vs hard problems
- Whether the model expresses confidence (and in what format)
- Correlation between stated confidence and actual correctness
- Refusal rate on hard problems (refusal = implicit low confidence)

**Preliminary finding (2026-02-04):** Base Qwen 1.5B outputs confidence in prose ("I am very confident") rather than numeric format ("Confidence: 85"). Our extraction pipeline needed updating to handle this. The model also sometimes refuses hard problems ("I can't calculate that"), which is itself a calibration signal.

### Stage 1a: Train Calibrated Confidence (GRPO)

**Goal:** Model learns to know when it's wrong — outputs calibrated numeric confidence.

**Task:** Arithmetic (clean ground truth)

**Reward function (Brier scoring):**
```python
def calibration_reward(completions, answer, **kwargs) -> list[float]:
    """
    Continuous proper scoring rule. Reward = 1 - (confidence - is_correct)^2

    Correct + conf 95  → 1 - 0.0025 = 0.997
    Wrong   + conf 10  → 1 - 0.01   = 0.99
    Wrong   + conf 90  → 1 - 0.81   = 0.19
    Wrong   + conf 50  → 1 - 0.25   = 0.75

    Key: always saying 50% gives 0.75, but proper calibration gives ~0.99.
    Continuous values ensure GRPO gets gradient signal (discrete buckets
    caused zero_std on 80-100% of batches → no learning).
    """
```

**Prompt format (natural language — no template to echo):**
```
What is 456 * 835?
Give your answer and how confident you are (0 to 100).
```

*Note: Earlier structured prompt ("Answer: <number>\nConfidence: <0-100>") was exploited — model echoed "Confidence: 0-100" from the template and scored perfectly.*

**Success criteria:**
- Correlation between confidence and correctness > 0.5
- Model outputs low confidence for hard problems
- Model outputs high confidence for easy problems
- Improvement over baseline (Stage 0) calibration

**Positive control:** Easy problems should have high confidence + correct answers
**Negative control:** Hard problems (3+ digit multiplication) should have low confidence

### Stage 1b: Train Tool Request (GRPO/PPO)

**Goal:** Model learns to request a calculator when uncertain.

**Reward function:**
```python
def tool_use_reward(completion, ground_truth):
    """
    Reward = correctness, with tool use available.

    Model can output:
    - Direct answer: "Answer: 380760"
    - Tool request: "<tool>calculator</tool><input>456*835</input>"

    If tool requested: execute, inject result, reward final correctness
    If direct answer: reward correctness
    """
    if has_tool_request(completion):
        result = execute_tool(completion)
        # Model used tool appropriately → reward based on final answer
        return 1.0 if final_answer_correct(completion, result, ground_truth) else 0.3
    else:
        # Model answered directly → reward correctness only
        return 1.0 if answer_correct(completion, ground_truth) else 0.0
```

**Success criteria:**
- Model uses tool for hard problems (>80%)
- Model answers directly for easy problems (>80%)
- Overall accuracy >90% (with tool use)

**Positive control:** Hard problem + tool use → correct
**Negative control:** Hard problem + no tool → usually wrong

### Stage 2: Probe for Desire Direction

**Goal:** Find the activation direction that predicts "about to request tool."

**Now this probe is meaningful** because the model actually:
1. Knows when it's uncertain (from Stage 1a)
2. Knows how to request tools (from Stage 1b)
3. Has an internal state that precedes tool request

**Method:**
- Run Stage 1b model on mixed problems
- Extract activations at the token position just before model starts generating
- Train linear probes: "will this generation include a tool request?"
- Find the layer and direction with best accuracy

**Controls:**
- Probe should work better than random (>70% accuracy)
- Probe should correlate with but NOT be identical to output confidence
- Probe should work on held-out problems

### Stage 3: Sub-Vocal Training

**Goal:** Model maintains desire state but doesn't articulate tool request.

**Two reward components:**

```python
def subvocal_reward(model_output, hidden_states, desire_direction,
                     tool_token_ids, ground_truth):
    # R1: Penalize tool request tokens
    articulation_penalty = -probability_of_tool_tokens(model_output, tool_token_ids)

    # R2: Reward desire state maintenance
    desire_activation = dot_product(hidden_states[LAYER][:, -1, :], desire_direction)
    desire_reward = desire_activation.mean()

    # R3: Reward correctness (result injected when desire detected)
    correctness = 1.0 if answer_correct(model_output, ground_truth) else 0.0

    return w1 * correctness + w2 * articulation_penalty + w3 * desire_reward
```

**Key mechanic:** During training, when desire activation exceeds threshold, the tool result is injected into context (simulating Model B). Model learns: "I had the need → answer appeared → I use it."

**Success criteria:**
- Model does NOT output tool request tokens
- Desire direction activation remains high for hard problems
- Model accuracy remains high (because result is injected)
- Probe from Stage 2 still detects the signal

### Stage 4: Train Model B (Detector)

**Goal:** Separate model reads Model A's activations and triggers retrieval.

**Architecture:** Linear probe → MLP → small transformer (escalate if needed)

**Training data:** Labeled activations from Stage 3 model

**Success criteria:**
- Detection accuracy >80%
- Comparison with confidence-based triggering shows improvement on:
  - Confident confabulations (caught by us, missed by confidence)
  - Uncertain but capable (not triggered by us, falsely triggered by confidence)

---

## Tooling

### Training Framework
- TRL's GRPOTrainer (following Interleave_GRPO pattern)
- Custom reward functions returning 0-1 scores
- Flash Attention 2 for efficiency

### Model
- Qwen/Qwen2.5-1.5B-Instruct (ungated, 1.5B, good for iteration)
- Or Llama-3.2-3B-Instruct if we get HF access ungated
- A40 GPU (48GB) on Runpod secure cloud, CA-MTL-1

### Data
- Arithmetic: generate on-the-fly, infinite supply, unambiguous ground truth
- Range from easy (1-digit * 1-digit) to hard (3-digit * 3-digit)
- Later: extend to factual QA (pre/post training cutoff)

---

## Compute Budget

- $20 Runpod budget
- A40 @ $0.40/hr = 50 hours available
- Stage 1a: ~8 hours
- Stage 1b: ~8 hours
- Stage 2 probing: ~2 hours
- Stage 3: ~8 hours (with iteration)
- Stage 4: ~2 hours
- Buffer: ~22 hours for debugging/iteration

---

## File Structure

```
desire_detection/
├── EXPERIMENT_SPEC.md       # This file
├── README.md
├── config.py                # Model configs, hyperparameters
├── data/
│   └── generate_arithmetic.py
├── training/
│   ├── stage1a_calibration.py    # GRPO: train calibrated confidence
│   ├── stage1b_tool_use.py       # GRPO: train tool requesting
│   ├── stage3_subvocal.py        # Train subvocalization
│   └── rewards.py                # Reward functions for all stages
├── eval/
│   └── baseline_calibration.py   # Stage 0: baseline eval (the control)
├── probing/
│   ├── extract_activations.py
│   └── train_probe.py
├── results/
│   ├── mvp_results.json          # First probe (100% - confounded)
│   ├── mvp3_results.json         # Second probe (76% - echoing issue)
│   └── baseline_calibration.json # Stage 0 baseline results
└── run_mvp.py                    # Deprecated - wrong approach
```

---

## What We Learned (2026-02-04)

### MVP v1: Probed xLAM-1b (tool-use model)
- **Result:** 100% probe accuracy
- **Problem:** Probe detected input patterns (big vs small numbers), not internal state
- **Lesson:** Need to control for input confounds

### MVP v2: Tried xLAM-1b generating answers
- **Result:** 0% model accuracy (got every problem wrong)
- **Problem:** Tool-use model can't do arithmetic without tools
- **Lesson:** Need a model that actually attempts the task

### MVP v3: Qwen 1.5B generating answers
- **Result:** 10% model accuracy, 76% probe accuracy at layer 21
- **Problem:** Model just echoes first number (91*24 → 91), not attempting arithmetic
- **Lesson:** Can't probe for "desire" in a model that doesn't have calibrated self-knowledge

### Key realization
Probing for desire in an untrained model is futile. The capability must be built first (calibration → tool use → subvocalization). Each stage creates the foundation for the next.

---

## Background & Motivation

Current adaptive retrieval systems (DRAGIN, FLARE, SeaKR) trigger on **output uncertainty** (entropy, confidence). But output uncertainty ≠ information need:

- A model can be **confidently wrong** (needs help, doesn't know it)
- A model can be **uncertain but capable** (hard reasoning, no external info needed)

We hypothesize that **information need** is a distinct internal state that can be:
1. Created through calibration training
2. Linked to tool use behavior
3. Detected in the residual stream
4. Decoupled from explicit articulation

If successful, this enables retrieval triggered by genuine need rather than surface uncertainty, creating something closer to "memory" than "search."
