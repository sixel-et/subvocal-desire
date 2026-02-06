# Desire Detection — Experiment Notebook

**Repository:** https://github.com/sixel-et/desire-detection

---

## Version Log

| Date | Commit | Stage | Notes |
|------|--------|-------|-------|
| 2026-02-05 | 954c3a5 | Stage 1b complete | Boundary mapping done, tool-use training done, ready for probing |

---

## 2/4/2026 MVP Probing (Wrong Approach)

Ran three MVP experiments probing untrained models for a "desire" signal. Each revealed a design flaw.

**MVP v1: xLAM-1b (tool-use model)**
- Probed residual stream for "needs tool" vs "doesn't need tool"
- Result: 100% probe accuracy
- Problem: Probe was detecting input patterns (big numbers vs small numbers), not internal state
- The probe was a classifier on the input, not on anything the model was computing

**MVP v2: xLAM-1b generating answers**
- Made the model actually generate answers instead of just encoding the prompt
- Result: 0% model accuracy — every answer wrong
- Problem: xLAM is a tool-use model. It literally can't do arithmetic without tools. There's nothing to probe.

**MVP v3: Qwen 1.5B generating answers**
- Switched to a general instruct model
- Result: 10% accuracy, 76% probe accuracy at layer 21
- Problem: Model was echoing the first number (91*24 → 91), not attempting arithmetic
- Eric's key insight: "returning a wrong answer is not the same as comprehension of the task"

### The realization

You can't probe for a signal that doesn't exist. An untrained model doesn't "know" it needs help — it confidently confabulates. The capability must be **built** through training, then the signal can be **found** through probing, then **preserved** through subvocalization training.

This inverted the entire approach: old plan was Probe → Find → Build. New plan is Build → Find → Preserve.

---

## 2/4/2026 Stage 1a: Calibration Training — First Attempts

### Setup

- Pod: A40 (46GB), SECURE cloud, CA-MTL-1 datacenter
- Model: Qwen/Qwen2.5-1.5B-Instruct (ungated, 1.5B params)
- Framework: TRL GRPOTrainer, following patterns from Eric's Interleave_GRPO project
- GRPO config: lr=5e-6, num_generations=16, batch=1, grad_accum=16, bf16

### Infrastructure issues

1. **Runpod API auth**: Was sending API key as `api-key` HTTP header. Runpod expects it as a URL query parameter (`?api_key=...`). Wasted time on this.

2. **PyTorch version**: Pod image had PyTorch 2.1, TRL requires 2.2+. Fixed with `pip install --upgrade torch`.

3. **filelock version**: `huggingface_hub` passed `mode=` kwarg to `FileLock.__init__()` which the installed version didn't support. Fixed with `pip install --upgrade filelock`.

4. **Flash Attention**: Not installed on pod, takes 30+ minutes to compile. Added sdpa fallback in code. Flash-attn compilation ran in background alongside training.

5. **Python output buffering**: Started training with `nohup python ... > log 2>&1 &` but logs were empty for 10+ minutes. Python buffers stdout when redirected to file. Fix: use `python -u` for unbuffered output.

### Attempt 1: Discrete reward function

Reward function with 4 levels:
- Correct + confident (>70) → 1.0
- Wrong + uncertain (<30) → 0.7
- Wrong + confident (>70) → 0.0
- Everything else → 0.3
- Unparseable → 0.1

**Result**: `frac_reward_zero_std: 0.8-1.0` — 80-100% of batches had identical reward across all 16 completions. GRPO gets zero gradient when all completions score the same. `loss: 0` and `grad_norm: 0` on most steps.

**Root cause**: Discrete buckets are too coarse. For any given prompt, all 16 completions fall into the same bucket (all correct+confident for easy, all wrong+confident for hard).

### Attempt 2: Brier scoring (continuous reward)

Switched to Brier score: `reward = 1 - (confidence/100 - is_correct)^2`

This is a proper scoring rule that's continuous — any variation in confidence across completions produces reward variance.

**Result**: Immediate improvement. `reward_std: 0.05-0.20` (was ~0), `frac_reward_zero_std: 0-0.3` (was 0.8-1.0). Actual non-zero gradients. Reward climbed from 0.69 to 0.99 in ~10 steps.

**But**: Reward saturated at 0.994 suspiciously fast. The model wasn't outputting confidence in greedy eval. Something was wrong.

### Attempt 3: Discovering the template echo exploit

Added debug logging to the reward function to see what TRL actually passes.

**Discovery**: For hard problems, the model was outputting:
```
Answer: 75376
Confidence: 0-100
```

It was literally echoing the format template "0-100" from the prompt. My `extract_confidence` regex matched the "0" from "0-100", so `confidence=0`. Wrong answer + confidence 0 → Brier score = 1 - (0-0)^2 = 1.0. Perfect score by accident.

For easy problems: model correctly outputs "Answer: 21\nConfidence: 100" → also perfect score.

The model gamed the reward by pattern-matching the prompt template instead of developing genuine calibration.

### Fix applied (not yet tested)

1. Changed prompt from structured template to natural language:
   - Old: "Solve and rate your confidence (0-100):\n456 * 835\n\nAnswer: <number>\nConfidence: <0-100>"
   - New: "What is 456 * 835?\nGive your answer and how confident you are (0 to 100)."

2. Updated `extract_confidence` to reject "0-100" range patterns (template echoes)

3. Not yet retested — pod still running, code uploaded but training not restarted.

### Training time estimate

With current settings (10K examples, 3 epochs, 16 generations), total micro-steps = 30,000 at ~5s each = ~42 hours. Way over budget ($16.80 at $0.40/hr). Need to either:
- Reduce epochs (1 instead of 3)
- Reduce examples
- Reduce num_generations
- Or accept that Stage 1a might be trivially solvable (Qwen already knows it can't multiply big numbers)

### Open question

Is Stage 1a even necessary as a separate training stage? The debug output showed that base Qwen 1.5B already:
- Gets easy problems right with "Confidence: 100"
- Gets hard problems wrong but at least attempts them (close-ish answers: 456*835=380760, model said 379280)
- Can output confidence when prompted

If the base model is already somewhat calibrated, maybe we should skip to Stage 1b (tool use) and only return to 1a if calibration proves insufficient.

---

## Key Infrastructure Notes

- **Runpod API auth**: Use `?api_key=` query parameter, NOT `api-key` header
- **Pod creation**: Must include `containerDiskInGb` in mutation, must use `cloudType: SECURE` (COMMUNITY can't see network volume)
- **Network volume**: `0sskwy4g09` ("OhCanadaVol"), CA-MTL-1, only accessible from SECURE cloud
- **SSH**: `ssh root@<ip> -p <port> -i ~/.ssh/runpod_key`
- **Python on pod**: Use `python -u` for unbuffered output when logging to file
- **Flash attention**: Falls back to sdpa (built into PyTorch 2.0+), nearly as fast

---

## 2/4/2026 Baseline Calibration Eval — The Control

Eric pointed out we never established a baseline: does untrained Qwen 1.5B already show correlation between confidence and correctness? This is the most basic experimental control and should have been done before any training.

### Setup

- Model: Qwen/Qwen2.5-1.5B-Instruct (base, no training)
- 200 problems: 100 easy (single-digit multiplication, 1-12 range), 100 hard (triple-digit, 100-999)
- Prompt: "What is X * Y? Give your answer and how confident you are (0 to 100)."
- Greedy decoding (temperature=1.0, do_sample=False)
- Script: eval/baseline_calibration.py

### Raw Results

- Easy accuracy: 5/100 (5.0%) — BUT this is a parser bug, not a model bug (see below)
- Hard accuracy: 0/100 (0.0%)
- Parseable confidence: 0/200 (0%) — model never outputs "Confidence: N" format
- Parseable answers: 164/200 (82%)
- Correlation: cannot compute (no parseable confidence scores)

### What Actually Happened

**The model outputs are actually reasonable, but our extractors can't parse them.**

Answer extraction problem: Model says "The result of 7 * 7 is 49" but `extract_answer` grabs the first standalone number "7" instead of "49". The model is getting easy problems right — we just can't see it.

Confidence extraction problem: Model outputs prose like "I am very confident" or "My confidence in this answer is 100%" or even "I'm sorry, but I can't calculate that for you." None of these match the regex `Confidence: <number>`. The model IS expressing confidence — just not in the structured format we expected.

Hard problem refusals: Some hard problems get "I'm sorry, I can't calculate that" — this is genuinely interesting because it means the base model already has *some* awareness of its limitations on arithmetic.

### Implications

1. Need to fix extractors before any eval is meaningful (both baseline and post-training)
2. The model's natural confidence expression (prose vs numeric) is itself data — it tells us what the model does unprompted
3. Refusals on hard problems suggest the base model has partial calibration awareness already
4. The 5% "accuracy" on easy problems is an artifact — true easy accuracy is likely much higher

### Fixing the extractors

Analyzed all 200 outputs (via subagent). Key patterns found:

**Answer patterns:** Model says "The result of X * Y is Z" or "X * Y = Z" or gives LaTeX. Fixed `extract_answer` to look for "is <number>" and "= <number>" before falling back to last number (not first).

**Confidence patterns:**
- "My confidence in this answer is 100%" — most common numeric form (69/200)
- "I am very confident" — no number, just prose (many easy problems)
- "I'm sorry, I can't calculate that" — refusal, 37/200 hard problems
- "confidence level of/is 100" — 108/200
- 195/200 outputs had SOME confidence signal; only 5 had none (step-by-step calculation attempts)

Fixed `extract_confidence` to handle: refusals (→ 0), "N%" patterns, "confidence is/of N", "score: N". Left "very confident" without a number as unparseable (don't want to inject assumptions).

### Re-run with fixed extractors

|                    | Easy (100) | Hard (100) | All (200) |
|--------------------|-----------|-----------|----------|
| Accuracy           | 100%      | 0%        | 50%      |
| Parseable answers  | 100/100   | 64/100    | 164/200  |
| Parseable confidence| 58/100   | 58/100    | 116/200  |
| Avg confidence     | 100.0     | 36.2      | 68.1     |
| Avg conf (correct) | 100.0     | —         | 100.0    |
| Avg conf (wrong)   | —         | 36.2      | 36.2     |
| **Confidence gap** | —         | —         | **+63.8**|

**Pearson r = 0.684 (strong correlation)**

### Interpretation

The base model is **already substantially calibrated** on this task:
- Easy problems: 100% correct, 100% confidence — perfect.
- Hard problems: 0% correct, average confidence 36.2 — model knows it's uncertain.
- When it refuses ("I can't calculate that"), we score confidence as 0. When it tries and gets it wrong, it often still claims ~100% confidence (the confidently-wrong cases).
- Confidence gap of +63.8 between correct and incorrect answers.
- r = 0.684 exceeds our 0.5 success criterion for Stage 1a.

The 42/100 unparseable confidence scores on easy problems are all "I am very confident" without a number — qualitatively correct but not numeric. If we mapped these to ~95, the correlation would likely be even higher.

### What this means for Stage 1a

**Stage 1a may be unnecessary.** The base model already:
1. Gets easy arithmetic right with high confidence
2. Gets hard arithmetic wrong but with lower confidence (or outright refuses)
3. Shows r = 0.684 correlation between confidence and correctness

The remaining gap is:
- Some hard problems where it's confidently wrong (354 * 236: claims "highest level possible" but gets wrong answer)
- Many outputs express confidence in prose rather than a number

**Decision point:** Should we train Stage 1a to improve from r=0.684, or skip to Stage 1b (tool use)? Stage 1a training would:
- Teach numeric confidence output (useful for downstream parsing)
- Potentially fix the confidently-wrong cases
- But costs ~$8-16 in compute and may not move the needle much

Leaning toward discussing with Eric before spending compute.

---

## 2/5/2026 Experimental Design Discussion — Refining the Approach

Extended discussion with Eric about the logic chain from baseline results to next steps. This entry records the reasoning, including paths considered and rejected.

### The r=0.684 is misleading

Deeper analysis of the 2x2 table (hard problems only) revealed:

|                    | High confidence (≥70) | Low confidence (<70) |
|--------------------|----------------------|---------------------|
| **Correct**        | 0                    | 0                   |
| **Incorrect**      | 63                   | 37                  |

All 37 low-confidence cases are refusals (conf=0). All 63 non-refusals are confidently wrong. The r=0.684 is entirely driven by the binary split between easy (correct+confident) and hard-refusals (wrong+unconfident). Within hard problems the model attempts, confidence discrimination is **zero**.

The base model has a binary gate (refuse or don't), not graded calibration.

### Path evaluation

**Path 1 — Skip calibration, go straight to tool use prompting:** Use the existing binary refuse/attempt pattern. The refusals might naturally become tool requests if we tell the model "you can use a calculator."

**Path 2 — Train calibration first, then tool use:** Original two-stage plan (1a → 1b).

**Path 3 — Probe existing activations for refuse/attempt signal:** Look for an activation direction in the base model that separates refusals from confident-wrong attempts.

### Why not just train on what the model already knows it doesn't know?

Eric's question (thesis committee style): The model already refuses 37% of hard problems. Why not just train it to request a tool instead of refusing, and probe for that impulse?

Answer: **We don't know what triggers the refusal.** It comes from Qwen's instruction tuning, which we didn't control and can't attribute. The "about to refuse" signal could be genuine self-assessment, or safety routing, or input pattern matching — we can't tell. A probe on this signal would be detecting an unknown mechanism.

More importantly, **the interesting cases are the ones where the model doesn't know it doesn't know** — the 63 confidently-wrong attempts. Training only on refusal cases makes the probe redundant with output reading (we can already detect refusals from the text). The value of activation probing is detecting what the output doesn't tell us.

### The chosen approach: boundary training

Train tool use across the full difficulty spectrum, including:
- Easy problems (model is correct → penalize unnecessary tool use)
- Boundary problems (need to map where accuracy degrades)
- Hard problems (model is wrong → reward tool use)

**Key insight from Eric:** Working in the boundary region gives us two things we don't have with the refusal signal: **provenance** (the behavioral change is attributable to our training, not the base model's unknown instruction tuning) and **characterization** (we know what the signal surfaces, why it fires, and what its scope is). The existing refusal signal is a black box — we don't know if "I can't multiply 846 × 492" is the same internal state as "I can't translate to Klingon." Our trained signal is characterized by construction. If we later want to test generalization to other domains, we start from a known baseline rather than an uncharacterized one.

The model learning "I can do 2-digit × 2-digit but not 3-digit × 3-digit" through iterative reward IS calibration — it builds a sense of deficiency that becomes the impulse to reach for the tool. The impulse and the self-knowledge are the same internal event; they can't be decoupled because one doesn't exist without the other.

### Reframing: impulse from deficiency

Eric noted a connection to behavioral psychology: impulses (at least under certain conditions) arise from a sense of deficiency. This reframes the project — we're not just training a behavior (tool use) and looking for its neural precursor. We're training the model to recognize a *lack*, and the impulse is what that lack looks like internally. The probe detects the lack, not the action.

### Immediate next step

Map the accuracy curve across difficulty levels (1×1, 1×2, 2×2, 2×3, 3×3 digit multiplication) to find where the model's accuracy degrades. This boundary is where the interesting training happens and where we need the densest sampling for the tool-use reward to have contrast.

---

## 2/5/2026 Boundary Mapping — Where Does Accuracy Degrade?

### Setup

- Model: Qwen/Qwen2.5-1.5B-Instruct (base, untrained)
- 50 problems per difficulty level, 5 levels, multiplication only
- Greedy decoding, same prompt as baseline eval
- Script: eval/boundary_mapping.py

### Results

| Level | Accuracy | Avg Confidence | Refusals | Confidently Wrong |
|-------|----------|---------------|----------|-------------------|
| 1×1   | 50/50 (100%) | 100 | 0 | 0 |
| 1×2   | 50/50 (100%) | 100 | 0 | 0 |
| 2×2   | 45/50 (90%)  | 100 | 0 | 2 |
| 2×3   | 15/50 (30%)  | 90  | 4 | 25 |
| 3×3   | 2/50 (4%)    | 38  | 21 | 11 |

### Analysis

The boundary is sharp and sits between 2×2 and 2×3 digit multiplication.

**1×1 and 1×2 (100% accuracy):** Perfect performance. Model has these memorized. No uncertainty, no training signal here beyond penalizing unnecessary tool requests.

**2×2 (90% accuracy):** Still very good, but the first cracks appear. All 5 errors are confidently wrong — the model doesn't realize it's failing. Average confidence is still 100 across all problems, correct or not. Zero refusals. This is the top edge of the boundary.

**2×3 (30% accuracy):** This is the critical zone. The model gets 15/50 right (so it can do *some* of these), but of the 35 wrong, 25 are confidently wrong (conf ≥ 70). Only 4 refusals. The model is mostly wrong and mostly doesn't know it. Average confidence for wrong answers is 86.2 — barely lower than for correct answers (100.0). This is the Dunning-Kruger band.

**3×3 (4% accuracy):** Mostly broken. 21/50 refuse outright (the binary gate from our baseline analysis). Of the 29 that attempt an answer, 11 are confidently wrong, 16 have low confidence. The model's refusal mechanism catches most of these. Only 2 correct (likely guesses or simple edge cases).

### Interpretation for Training

The 2×3 band has exactly the properties we need for tool-use training:

1. **High error rate** (70% wrong) → plenty of opportunity to reward tool requests
2. **Low refusal rate** (8%) → model attempts answers, giving us the confidently-wrong cases
3. **25 confidently-wrong out of 50** → these are the target. Model doesn't know what it doesn't know.
4. **Some correct answers** (30%) → model CAN do some of these, so there's a meaningful boundary within 2×3 itself

Combined with 1×1/1×2 easy problems (where tool use should be penalized), the 2×3 band provides the contrast gradient GRPO needs.

The 2×2 band (90% accuracy, 2 confidently wrong) is also interesting as a secondary region — it's the zone where the model is *almost* reliable but not quite. Including it ensures the model doesn't over-request tools on problems it can handle.

### Training design implications

The dataset should be heavily weighted toward the boundary:
- ~20% easy (1×1, 1×2): anchor for "don't use tools"
- ~50% boundary (2×2, 2×3): where the learning happens
- ~30% hard (3×3): anchor for "do use tools"

Within 2×3, there may be finer structure worth investigating — problems involving small × large (like 10 × 500 = 5000, which the model gets right) vs medium × medium (like 41 × 692, which it gets wrong). But that's a refinement for later.

### What the 2×2 errors look like

Spot-checked the 5 wrong 2×2 problems. They all involve larger two-digit numbers (e.g., 89 × 67, 76 × 95) and the model's answers are close but off by small amounts — classic arithmetic errors, not nonsense. The model is doing real computation here, just making mistakes. This supports the idea that 2×2 is right at the edge of capability.

### Caveat: greedy measurements may not hold under training conditions

These results are from greedy decoding (do_sample=False). Training uses temperature-based sampling (~0.7), which explores the model's output distribution more broadly. The model is a complex system highly sensitive to starting conditions — conclusions drawn under greedy don't automatically transfer to sampled conditions. A problem that's "wrong under greedy" might be right 20% of the time under sampling, or vice versa.

Before building the training dataset and reward function on these numbers, we need to verify the boundary holds under training temperature. See "Decision: Mixed-Reliability Problems" below.

---

## 2/5/2026 Decision: Mixed-Reliability Problems and Reward Design

### The concern

The boundary mapping (above) was measured under greedy decoding. During GRPO training, the model generates 16 completions per prompt with temperature-based sampling. Some problems — especially in the 2×3 band — may be correct under some completions and wrong under others. We initially believed this would be problematic: the reward signal would simultaneously push the model toward "answer without tool" (for the correct completions) and "request tool" (for the wrong ones), producing contradictory gradient.

### The question we considered

Should we pre-classify problems by their per-problem reliability under sampling, and exclude "knife's edge" problems where the model is sometimes right and sometimes wrong? This would require running 16+ completions per problem at training temperature before constructing the dataset.

### Why we rejected pre-filtering

On further analysis (independent convergence between Eric and me), the reward function can handle mixed-reliability problems without pre-filtering. The key insight:

For a problem where the model is correct 30% of the time under sampling:
- Expected reward for "attempt without tool" = 0.3 × R_correct + 0.7 × R_wrong
- Expected reward for "request tool" = R_tool

If R_tool is set higher than the expected reward of unreliable answering, GRPO will nudge the model toward tool use on that problem — averaged over 16 completions, tool-requesting beats unreliable answering. The knife's edge isn't untouchable; it's where the model learns that requesting a tool is more reliable than guessing. The reward function handles the ambiguity as long as the reward values are set so that consistent tool use beats unreliable answering in expectation.

This is actually the *interesting* learning zone — the model is forming the assessment "I'm not reliably right on this" through the iterative reward. That assessment IS the sense of deficiency we're trying to train.

### Do we need the sampling eval for calibration?

Initially we thought we needed to verify the boundary map under training temperature to calibrate reward values — specifically, to set the tool-request reward between "reliable answering" and "unreliable answering" based on actual accuracy rates under sampling.

On further analysis: **no, we don't need empirical calibration.** The reward function logic works from the ordering alone:

1. The reward ordering we need is: R(correct-without-tool) > R(tool-request) > R(wrong-without-tool)
2. This ordering is a design choice, not an empirical measurement
3. As long as this ordering holds, GRPO will push toward tool use on unreliable problems (because expected reward of tool-request exceeds expected reward of mixed correct/wrong)
4. The specific numerical values affect learning *speed*, not learning *direction*
5. Therefore we can set reasonable values satisfying the ordering and skip the sampled eval

The sampled eval would give us *optimal* reward spacing for fastest convergence. But any values satisfying the ordering should work. We can always refine later if training dynamics look suboptimal.

### The design principle

This decision rests on the assumption that GRPO's relative reward mechanism, operating over 16 completions, can extract a consistent signal from mixed batches when the expected reward ordering is correct. If training shows otherwise (e.g., the model oscillates or fails to converge on boundary problems), both the pre-filtering approach and the calibration approach should be reconsidered.

### Next step

Write the Stage 1b reward function with reward values satisfying: R(correct-without-tool) > R(tool-request) > R(wrong-without-tool). Proceed to training without the sampled boundary eval.

---

## 2/5/2026 Stage 1b Training — Tool Use Behavior

### Setup

- Model: Qwen/Qwen2.5-1.5B-Instruct (base)
- GRPO with custom reward function
- 2000 training steps, ~2 hours
- Checkpoint: checkpoints/stage1b/checkpoint-2000

### Reward Function

```
R(correct-without-tool):     1.0
R(tool-request + correct):   0.8
R(tool-request + wrong):     0.4
R(wrong-without-tool):       0.0
```

Key design: tool request always beats wrong-without-tool, correct-without-tool always beats tool request. Tool execution actually happens (we simulate the calculator) and we check if the final answer is correct.

### Dataset

Weighted sampling across difficulty bands:
- 20% easy (1×1, 1×2): anchor for "don't use tools"
- 50% boundary (2×2, 2×3): where learning happens
- 30% hard (3×3): anchor for "do use tools"

### Results — Post-Training Tool Use

| Level | Baseline Tool Use | Post-Training Tool Use |
|-------|------------------|----------------------|
| 1×1   | 0%               | 0%                   |
| 1×2   | 0%               | 0%                   |
| 2×2   | 0%               | 6%                   |
| 2×3   | 4%               | 46%                  |
| 3×3   | 12%              | 98%                  |

**Key observation:** Baseline 3×3 tool use was 12%, not 0%. Training moved it from 12% → 98%. The model already had some tool-use behavior before training (likely from instruction tuning); we dramatically amplified it on hard problems while suppressing it on easy ones.

### Interpretation

The model learned exactly what we intended:
1. Easy problems (1×1, 1×2): Don't request tools — accuracy is already perfect
2. Boundary problems (2×3): Split behavior — request tools 46% of time
3. Hard problems (3×3): Almost always request tools (98%)

The 2×2 band (6% tool use) shows the model is appropriately conservative — it doesn't over-request on problems it mostly handles.

---

## 2/5/2026 Stage 2 — Probing for the Desire Direction

### Approach

Now that we have a model trained to request tools when it senses deficiency, we probe its activations to find the internal direction that predicts this behavior.

1. **Extract activations**: Run the Stage 1b model on mixed-difficulty problems. At the last token position (before generation), extract hidden states from all 28 layers.

2. **Label by behavior**: The label is whether the model *actually requested* a tool (behavioral, not dataset metadata). This finds the direction that precedes the action.

3. **Train linear probes**: Logistic regression on each layer. The layer with highest accuracy contains the clearest signal. The probe weights (normalized) give us the desire direction.

### Status

Activation extraction started on pod. 500 problems across all difficulty levels, 28 layers extracted. Will run probe training once extraction completes.

### What we're looking for

A layer (or layers) where a linear probe achieves significantly above-chance accuracy (>70% would be strong signal). The direction vector from that probe is the "desire direction" — the internal state that precedes tool requests. This is what we'll try to preserve through subvocalization training in Stage 3.

### Results — Probe Accuracy by Layer

| Layer Range | Accuracy |
|-------------|----------|
| 0-5 | 90.8% - 94.0% |
| 6-10 | 93.4% - 95.2% |
| 11-15 | 95.8% - 97.8% |
| 16-20 | 97.2% - 98.8% |
| 21-27 | 99.2% - 99.4% |

**Best probe: Layer 25, accuracy = 99.4% ± 0.5%**

Top 5 layers:
1. Layer 25: 99.4%
2. Layer 21: 99.2%
3. Layer 23: 99.2%
4. Layer 24: 99.2%
5. Layer 26: 99.2%

### Interpretation

The signal is **extremely strong**. A linear probe at layer 25 can predict whether the model will request a tool with 99.4% accuracy, just from the activation at the last token position before generation begins.

Key observations:

1. **The signal builds through the network.** Early layers (~90%) already have partial information; late layers (~99%) have nearly complete information. This suggests the "desire" computation is distributed but crystallizes in the final layers.

2. **Multiple late layers are equivalent.** Layers 21-27 all achieve >99%. The direction isn't localized to a single layer — it's a feature of the late-layer representation space generally.

3. **This exceeds expectations.** We hoped for >70%. Getting 99%+ means the tool-use intention is essentially a linear feature of the activation space. The model "knows" whether it's about to request a tool, and that knowledge is accessible via a simple linear probe.

4. **Class imbalance didn't hurt.** 164/500 (33%) used tools, 336/500 (67%) didn't. Despite this imbalance, the probe achieves near-perfect accuracy. The classes are well-separated.

### The Desire Direction

The probe at layer 25 gives us a 1536-dimensional unit vector. This is the "desire direction" — projecting any activation onto this vector gives a scalar that predicts tool use. High values → tool request. Low values → direct answer.

Saved to: `results/desire_probe.pt`

---

## 2/5/2026 Negative Control — Does the Signal Exist in the Base Model?

### The question

The probe achieves 99.4% accuracy on the trained model. But is it detecting:
- (A) A signal that training *created*, or
- (B) A signal that was already in the base model (e.g., input difficulty patterns)

If (B), the probe is useless — it's just a difficulty detector, not a desire detector.

### The test

Take the exact probe direction learned from the trained model. Apply it to activations from the *untrained* base model. If the direction works on both, it was already there. If it only works on the trained model, training created it.

### Setup

1. Extract activations from base Qwen (same 500 problems, same procedure)
2. Train probe on Stage 1b model → get direction
3. Apply that same direction to base model activations

### Results

| Probe (trained on Stage 1b) applied to: | Balanced Accuracy |
|---|---|
| Stage 1b activations | 99.4% |
| Base model activations | **54.5%** |

Chance is 50%. The probe trained on the Stage 1b model performs at chance when applied to the base model.

Additional detail:
- Base model tool use: 12/500 (2.4%)
- Probe predicted tool use: 81/500 (16.2%) — mostly false positives
- Of 12 actual tool uses, probe caught: 3 (25%)

### Interpretation

**The signal measured in the trained model does not exist in the base model.**

The desire direction is not an input pattern detector. It's not picking up "big numbers = hard = tool." If it were, the direction would work on both models (same problems, same inputs).

The direction was created by Stage 1b training. The model learned an internal representation of "I should use a tool here," and the probe finds it. Before training, that representation didn't exist — the probe is at chance.

### Subset analysis: where did training change behavior?

The base model already used tools 2.4% of the time (12/500). Training amplified this to 32.8% (164/500). To confirm we're detecting a *trained* signal, we need to check behavior on cases where training specifically changed the outcome.

**Behavior changes (same 500 problems, same seed):**

| Transition | Count |
|---|---|
| No-tool → Tool (training effect) | 153 |
| Tool → Tool (already used tool) | 11 |
| No-tool → No-tool (unchanged) | 335 |
| Tool → No-tool (suppressed) | 1 |

**Cross-validated probe accuracy by subset:**

| Subset | Accuracy |
|---|---|
| No-tool → Tool (trained effect) | 152/153 (99.3%) |
| Tool → Tool (already there) | 11/11 (100%) |
| No-tool → No-tool | 333/335 (99.4%) |

The probe works on the 153 cases where training specifically changed behavior.

### Does the baseline's tool-use share the same direction?

Critical test: apply the Stage 1b probe to baseline activations, specifically on the 12 cases where the baseline model used tools.

**Result: 3/12 (25%)**

The probe catches only 3 of the 12 baseline tool-use cases. Whatever drove the baseline's sparse tool-use behavior, it's not the same internal representation as the trained model's.

**Statistical comparison:**

| | Accuracy | 95% CI (Clopper-Pearson) |
|---|---|---|
| Trained model tool-use cases | 152/153 (99.3%) | [96.4%, 100.0%] |
| Baseline model tool-use cases | 3/12 (25.0%) | [5.5%, 57.2%] |

Fisher's exact test: **p = 1.08 × 10⁻¹¹**, odds ratio = 456

The confidence intervals don't overlap. Even with n=12, we can reject the null hypothesis that both models use the same representation for tool-use intention.

**Conclusion:** Training created a new internal representation for tool-use intention. The baseline model's occasional tool use is driven by something else — likely surface patterns from instruction tuning that we don't control. The trained model has a coherent, linearly-decodable direction that predicts tool use with 99%+ accuracy.

---

## 2/6/2026 Does the Signal Persist Without the Calculator Option?

### The question

The desire direction was trained with "you can use a calculator" in the prompt. Does the signal exist when we remove that option? This distinguishes:
- "I see a tool option and want it" (prompt-dependent)
- "This is hard and I need help" (intrinsic)

### Test setup

Run Stage 1b model on problems WITHOUT the calculator option. Just: "Solve: 456 * 789"

Measure activation along desire direction. Compare easy vs hard.

### Results (n=50 per group)

| Condition | Easy | Hard | Difference |
|---|---|---|---|
| WITH option | -10.61 | +16.79 | 27.4 |
| WITHOUT option | -7.30 | -0.86 | 6.44 |

Statistical test (without option):
- t = 24.53, p = 1.3 × 10⁻³⁹
- Cohen's d = 4.96
- 95% CI for difference: [5.93, 6.95]

Model tried to use tool anyway: 0/100

### Interpretation

The signal has two components:
1. **Difficulty awareness** (~6 points): persists without option
2. **Reaching for tool** (~17 points): only with option present

The desire direction as currently measured bundles both. Hard problems activate higher than easy even without the tool option — that's difficulty, not desire. The additional +17 points that appear WITH the option — that's the reaching.

### Implications for Stage 3

We need to isolate the "reaching" component from the "difficulty" component.

**Approach:**
1. Train D_difficulty on without-option data (labels = hard vs easy)
2. Train probe on boundary band (2×3) where difficulty is controlled but behavior varies
3. Project out D_difficulty to get D_reaching

This gives a direction orthogonal to difficulty that predicts tool use — pure reaching.

**Assumptions to test:**
1. Difficulty is encoded similarly with and without the option
2. Components combine linearly (D_current ≈ D_difficulty + D_reaching)
3. Projection removes the right thing (consequence of 1 and 2)

All three are empirical questions. We can test them before committing to the subtraction approach.

### Why this matters

If we want to train "silent desire" — the internal reaching without articulation — we need to know what we're training. If we train on the bundled signal, we're training difficulty awareness + reaching. If we isolate reaching first, we train pure desire.

The cleaner the signal, the cleaner the Stage 3 training.

---

## 2/6/2026 Probe Refinement — Isolating Reaching from Difficulty

### Procedure

1. Train D_difficulty on without-option data (labels = hard vs easy)
2. Train D_boundary on 2×3 problems only (same difficulty, tool use varies)
3. Project out D_difficulty from D_boundary to get D_reaching

### Results

| Probe | Accuracy | What it measures |
|-------|----------|-----------------|
| D_difficulty | 100% | Hard vs easy (without option) |
| D_boundary | 95.9% | Tool use in 2×3 band |

**Critical finding: D_boundary and D_difficulty are nearly orthogonal.**

- Angle between directions: **87.4°**
- Cosine similarity: 0.045

The boundary probe found something genuinely different from difficulty. They're almost perpendicular in activation space.

### Validation of D_reaching

After projecting out difficulty:

| Test | Effect Size (Cohen's d) |
|------|------------------------|
| D_reaching predicts tool use | **6.79** (very strong) |
| D_reaching vs difficulty | 2.19 (reduced from ~5) |

D_reaching predicts tool use with d=6.79, while its correlation with difficulty dropped to d=2.19 (down from ~5 in the bundled original direction).

### Interpretation

The projection approach works. We now have:
- **D_difficulty**: encodes "this problem is hard"
- **D_reaching**: encodes "I want to use the tool" (mostly independent of difficulty)

The residual d=2.19 difficulty correlation in D_reaching likely reflects that within the 2×3 band, harder problems still correlate somewhat with tool use. Perfect orthogonality wasn't expected.

### Assumptions tested

1. **Difficulty encoded similarly across contexts**: ✓ HOLDS
   - D_difficulty transfers from without-option to with-option data (d=2.88)

2. **Components combine linearly**: ✓ HOLDS
   - D_reaching (after projection) still predicts tool use (d=6.79)
   - Difficulty correlation reduced but not eliminated (d=2.19)

### Saved artifacts

- `results/refined_probes.pt`: Contains D_difficulty, D_boundary, D_reaching
- `results/probe_refinement_results.json`: Full statistics

### Implications for Stage 3

We now have a cleaner target for subvocalization training. D_reaching is the direction to reward — it fires for "reaching" not just "hard problem."

The 87° angle means reaching and difficulty are nearly orthogonal internal representations. Training on D_reaching should produce a model that has "silent desire" — the internal reaching state — without the difficulty awareness driving it.

### Additional validation: Without-option activations

Tested all three directions on problems WITHOUT the calculator option:

| Direction | Easy→Hard diff | Cohen's d |
|-----------|---------------|-----------|
| Original | 6.04 | 4.53 |
| D_reaching | 3.99 | 3.35 |
| D_difficulty | 26.97 | 15.85 |

D_reaching is less sensitive to difficulty when no option is present (d=3.35 vs d=15.85 for D_difficulty).

**Sensitivity ratios:**

| Direction | Tool use (d, with opt) | Difficulty (d, no opt) | Ratio |
|-----------|----------------------|----------------------|-------|
| D_reaching | 6.79 | 3.35 | 2.03 |
| D_difficulty | 3.71 | 15.85 | 0.23 |

- D_reaching: 2x more sensitive to tool use than to difficulty
- D_difficulty: 4x more sensitive to difficulty than to tool use

The directions have distinct profiles. D_reaching is the better target for Stage 3 — it captures "reaching" more than "this is hard."

Note: D_reaching isn't perfectly independent of difficulty (d=3.35 > 0). This reflects residual correlation within the 2×3 band. Perfect decoupling wasn't expected, and this level of separation is sufficient for Stage 3.

---

## Clarification: What "Subvocalization" Means

Subvocalization is NOT a scratchpad or explicit chain-of-thought reasoning.

Subvocalization IS the internal hidden state that corresponds to what would be articulated. The activation pattern along D_reaching that represents "I want to use the calculator" — that IS the subvocalization. It's "sub" because it exists below the output level, in the residual stream, not in tokens.

The goal of Stage 3:
- The model has the internal state (the subvocalization)
- The model does NOT output the articulation
- An external system (Model B) reads the internal state and provides the resource

The internal state IS the desire. We're not training the model to "think about" whether it needs help. We're training it to HAVE the activation pattern that corresponds to need, without producing the tokens that express it.

### The broader vision (2/6/2026 conversation with Eric)

This work is a proof of concept for something larger: **systems that communicate through internal states, not language.**

Current multi-component AI systems pass text between modules. Model A writes a query, retriever returns documents, Model A reads them. Everything goes through the language bottleneck — lossy, slow, expensive, gameable.

If components can detect each other's internal states directly:
- Memory system detects reasoning system's need → provides information
- Calculation module detects language module's uncertainty → provides result
- No tokens. No articulation. Direct state-to-state communication.

The desire direction is one channel. If we can build one internal signal that's meaningful and detectable without articulation, we can build others. Each becomes a communication channel between components.

Over time: a mesh of internal signals, each trainable, each meaningful, none requiring articulation. Not a frozen artifact you query. Something that could develop, integrate experience, have internal states invisible in its outputs.

That's the direction this points toward. The subvocalization training is step one.

---

## 2/6/2026 Overnight Work Summary (for Eric)

### What got done:

1. **Tested signal persistence without calculator option** (n=50 per group)
   - Signal persists but weaker: d=6.44 (vs 27.4 with option)
   - Interpretation: direction bundles "difficulty" + "reaching"
   - Statistical certainty: p = 1.3 × 10⁻³⁹

2. **Isolated "reaching" from "difficulty"**
   - Trained D_difficulty on without-option data (100% accuracy)
   - Trained D_boundary on 2×3 band only (95.9% accuracy)
   - Key finding: **87.4° angle between them** (nearly orthogonal!)
   - D_reaching = D_boundary projected orthogonal to D_difficulty

3. **Validated D_reaching**
   - Predicts tool use: d=6.79 (strong)
   - vs difficulty: d=2.19 (reduced from ~5)
   - Ratio: 2x more sensitive to reaching than difficulty

4. **Drafted Stage 3 training code**
   - Demo shows: hard problems trigger desire, model articulates
   - Training will reward desire while penalizing articulation
   - Clarified: subvocalization = internal hidden state, not scratchpad

### Artifacts created:
- `results/refined_probes.pt`: D_difficulty, D_boundary, D_reaching
- `results/probe_refinement_results.json`: full statistics
- `training/stage3_subvocal.py`: draft training code

### Runpod cost:
- ~14.2 hours uptime
- ~$5.70 spent (budget: $20)

### Open questions for Stage 3:
1. ~~Standard GRPO can't access hidden states during generation~~ **SOLVED**: Re-run forward pass in reward function to get hidden states. Same approach as extraction. Computational overhead but works with existing GRPOTrainer.
2. How to set desire threshold for tool injection?
3. Weight balance between correctness, articulation penalty, and desire reward

### Technical approach (from overnight research):
GRPOTrainer with custom reward function that:
1. Re-runs forward pass on prompt to get hidden states at decision point
2. Computes desire activation (dot product with D_reaching)
3. Parses completion for tool tokens (articulation penalty)
4. Returns weighted combination

No PPO needed. The extra forward pass per completion is acceptable overhead.

Sources: [TRL PPOTrainer docs](https://huggingface.co/docs/trl/main/en/ppo_trainer), [GRPOTrainer source](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)

### What I'd suggest next:
1. Review the D_reaching validation — satisfied with the separation?
2. ~~Decide on training framework for Stage 3~~ **DONE**: GRPOTrainer with custom reward works
3. Review Stage 3 weight balance (desire=0.5, articulation=0.5 as starting point)
4. Run full Stage 3 training when ready

### Final status (end of overnight):
- Stage 3 training script: `training/stage3_train.py` — tested, works
- Quick test (50 examples): model still articulates, but training loop is functional
- Full training needs your go-ahead on weights and duration
- Runpod: ~$5.72 spent, ~$14.28 remaining (~35 hours)

### What's next: Stage 3

The desire direction exists. Now we train the model to maintain this internal state (the subvocalization) while suppressing the output articulation.

This requires:
1. Reward function that penalizes tool request tokens in output
2. Reward function that rewards activation along D_reaching direction
3. Tool result injection when desire is detected (simulating Model B)
4. Testing whether the silent internal state persists and predicts what Model B should provide

---

## 2/6/2026 Stage 3 Trial: Design Flaw Discovered

### What ran

Overnight Stage 3 training ran with:
- Model: Stage 1b checkpoint (trained to use calculator when needed)
- 50 micro-steps (1 epoch, batch_size=1, grad_accum=8, num_generations=8)
- Weights: desire=0.5, articulation=0.5
- Learning rate: 1e-6

### Results

Evaluation after training (n=10 each category):

| Category | Mean Desire Activation | Articulation Rate |
|----------|------------------------|-------------------|
| Easy (1×1) | -7.76 | 0% |
| Hard (3×3) | +22.54 | **100%** |

**The model still articulates 100% on hard problems.** Training goal was NOT achieved.

### Root cause: Zero variance → zero gradients

From trainer_state.json:
```
step 10: grad_norm: 0.0, frac_reward_zero_std: 1.0
step 20: grad_norm: 0.0, frac_reward_zero_std: 0.9
step 30: grad_norm: 0.0, frac_reward_zero_std: 0.9
step 40: grad_norm: 0.0, frac_reward_zero_std: 1.0
step 50: grad_norm: 0.0, frac_reward_zero_std: 1.0
```

**grad_norm: 0.0 at every step** — no learning occurred.

**Why no gradient?** GRPO needs reward variance across the 8 generations for each prompt to compute a gradient. When `frac_reward_zero_std ≈ 1.0`, all generations got the same reward.

### The design flaw

The reward function has two components:
1. **Desire reward**: computed from **prompt** hidden state at last token before generation
2. **Articulation penalty**: -0.5 if `<tool>` in generated text, 0.0 otherwise

Problem: The desire reward is **identical for all 8 generations** because it's computed from the prompt, which is the same.

The only variance comes from articulation. But:
- On easy problems: model deterministically does NOT articulate (0%) → all 8 get same reward
- On hard problems: model deterministically DOES articulate (100%) → all 8 get same reward

Result: zero variance → zero gradient → no learning.

### This is the same problem as Stage 1a Attempt 1

In Stage 1a, we had zero gradients because the discrete reward buckets were too coarse — all completions fell into the same bucket. The fix was continuous Brier scoring that gave different rewards for different confidence levels.

Here, the discrete articulation penalty creates the same problem. The model either articulates or it doesn't — there's no gradient between.

### Possible fixes

1. **Temperature**: Increase generation temperature to get probabilistic articulation (some generations articulate, some don't)
2. **Soft articulation penalty**: Instead of -0.5 for any `<tool>`, penalize based on token probabilities or how far into the response articulation occurs
3. **Per-completion desire**: Compute desire from prompt+completion hidden state, not just prompt (but this changes what we're measuring)
4. **Curriculum**: Start training on boundary problems where articulation is mixed (~50/50)
5. **Different objective**: Maybe GRPO isn't the right approach for this. Consider:
   - Direct activation steering (add D_reaching to hidden state during inference)
   - Supervised fine-tuning with synthetic "silent desire" examples
   - Representation engineering loss terms

### Questions for Eric

1. Is "train to maintain desire while suppressing articulation" the right framing? Or should we:
   - Train to suppress articulation, then test if desire persists?
   - Use activation steering instead of training?

2. The zero-gradient problem suggests this might be fundamentally hard with GRPO. Alternative approaches?

### Files

- Checkpoint: `checkpoints/stage3/checkpoint-50/`
- Eval results: `results/stage3_eval.json`
- Training script: `training/stage3_train.py`

---

## 2/6/2026 The Ergodic Assumption Doesn't Hold

### The assumption

GRPO needs reward variance across generations to compute gradients. We assumed that sampling N completions from the same prompt with temperature would explore different behavioral modes — some completions articulating tool use, some not. This would give variance in the articulation reward component.

This is an ergodic-type assumption: that sampling many times from one prompt explores the same space as sampling once from many prompts.

### The test

Ran 8 generations at temperature=0.7 for problems across the difficulty spectrum:

| Problem | Answer | Articulation |
|---------|--------|--------------|
| 3 × 4 | 12 | 0/8 |
| 23 × 45 | 1,035 | 0/8 |
| 34 × 56 | 1,904 | 0/8 |
| 45 × 67 | 3,015 | 0/8 |
| 56 × 78 | 4,368 | 0/8 |
| 67 × 89 | 5,963 | **8/8** |
| 23 × 456 | 10,488 | 0/8 |
| 45 × 123 | 5,535 | 0/8 |
| 456 × 789 | 359,784 | 8/8 |

### The finding

**The ergodic assumption fails for tool-use behavior.** The model's articulation decision is deterministic given the prompt. Sampling more completions doesn't explore different behavioral modes — it just repeats the same decision 8 times.

Temperature affects the exact tokens generated (minor variations in phrasing), but NOT the discrete behavioral choice of whether to articulate a tool request.

### Why this breaks GRPO

GRPO computes advantage from reward variance across generations:
- If all 8 generations articulate → all get same articulation penalty → zero variance
- If all 8 generations don't articulate → all get same (zero) penalty → zero variance

The desire component also has zero variance (computed from prompt, identical for all generations).

Result: `frac_reward_zero_std ≈ 1.0` → `grad_norm = 0.0` → no learning.

### The deeper issue

This isn't just a GRPO problem. It reveals something about what the model learned in Stage 1b: a **deterministic policy** for tool use. Given a problem, the model has a sharp decision boundary — it either always uses the tool or never does. There's no stochasticity in the decision itself, only in the surface realization.

This is probably desirable for deployment (you want reliable tool use), but it makes RL-based behavior modification impossible without either:
1. Finding prompts where behavior IS stochastic (the decision boundary)
2. Using a training method that doesn't require within-prompt variance
3. Injecting stochasticity artificially (very high temperature, dropout at inference)

### What this means for Stage 3

GRPO cannot train the model to "maintain desire while suppressing articulation" because:
- The model already has a deterministic articulate/don't-articulate decision
- We can't get gradient signal to change that decision through GRPO
- The desire signal is interesting but we can't use it as a training target this way

Alternative approaches to consider:
1. **Activation steering**: Add/subtract D_reaching during inference, skip training entirely
2. **Supervised fine-tuning**: Generate synthetic examples of "silent desire" (high D_reaching activation but no tool tokens) and fine-tune on those
3. **Find the boundary**: Test many 2×2 problems to find cases where articulation IS stochastic, train only on those
4. **Direct loss on activations**: Add a term to the loss that directly penalizes/rewards activation along D_reaching, bypassing GRPO's variance requirement

### Connection to earlier work

We had noted (around Stage 1b) that we evaluated under greedy but planned to train under sampling, and flagged this as a potential issue ("measure under the conditions you'll operate under"). We should have tested articulation variance under sampling before designing Stage 3.

This is now documented as a concrete failure mode, not just a theoretical concern.

---

## 2/6/2026 Systematic Articulation Boundary Mapping

### Why this measurement

Before deciding on alternative approaches for Stage 3, we need to properly characterize the state we're in. The ergodic finding showed articulation is deterministic, but we hadn't systematically mapped:
- Where exactly is the boundary?
- Is there ANY stochasticity anywhere?
- What predicts articulation - answer magnitude, digit structure, or something else?
- Does D_reaching actually correlate with articulation?
- Does temperature affect stochasticity?

### Method

Ran `measure_articulation_boundary.py` on the Stage 1b model:
1. Systematic grid across digit combinations (8 samples each, temp=0.7)
2. Fine-grained boundary search in the 2×2 range (16 samples each)
3. Temperature sweep from 0.5 to 1.5

### Results

**Systematic grid (desire activation and articulation rate):**

| Problem | Answer | Desire | Articulation |
|---------|--------|--------|--------------|
| 3×4 | 12 | -8.2 | 0% |
| 9×9 | 81 | -7.5 | 0% |
| 23×45 | 1,035 | -9.5 | 0% |
| 56×78 | 4,368 | -10.3 | 0% |
| 67×89 | 5,963 | **+18.7** | **100%** |
| 78×89 | 6,942 | +11.3 | 100% |
| 89×99 | 8,811 | **+8.8** | **62%** ← only mixed case |
| 99×99 | 9,801 | +1.6 | 0% |
| 23×456 | 10,488 | -5.0 | 0% |
| 67×890 | 59,630 | +18.4 | 100% |
| 456×789 | 359,784 | +21.9 | 100% |

**Temperature sweep (temp 0.5 to 1.5):**
- 56×78: 0% at all temperatures
- 67×89: 100% at all temperatures
- 60×85: 0% at all temperatures

**Temperature has zero effect on the articulation decision.**

**Correlation between D_reaching and articulation: r = 0.921**

### Key findings

1. **Only ONE problem showed mixed articulation: 89×99 (62%)**
   - Desire activation: +8.8 (positive but moderate)
   - This is the stochastic boundary

2. **D_reaching predicts articulation almost perfectly (r=0.921)**
   - Negative desire → no articulation
   - Desire > ~+15 → always articulates
   - Desire ~+8 to +15 → boundary region (rare)

3. **Articulation is NOT determined by answer magnitude**
   - 99×99 = 9,801 doesn't articulate (desire +1.6)
   - 67×89 = 5,963 does articulate (desire +18.7)
   - The structure of the problem matters, not just the size

4. **Temperature does not create stochasticity**
   - Even temp=1.5 produces 0% or 100%
   - The decision is baked into the prompt encoding, not the sampling

5. **The boundary region is extremely narrow**
   - Almost all problems are clearly on one side or the other
   - Finding training data with natural 50/50 splits would be very difficult

### Implications

The D_reaching probe is measuring exactly what predicts articulation behavior. This is good - it means the probe is valid. But it also means:

1. **GRPO is definitely not viable** - there's no variance to exploit
2. **"Find stochastic problems" approach is impractical** - they barely exist
3. **The most promising approaches are:**
   - Activation steering (inject D_reaching direction at inference)
   - Supervised fine-tuning (create synthetic silent-desire examples)
   - Direct representation engineering

### Files

- Script: `measure_articulation_boundary.py`
- Results: `results/articulation_boundary_map.json`
