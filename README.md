# Sub-Vocal Desire Detection

This repository contains both a technical research project and the interaction traces from conducting that research as a long-horizon human-AI collaboration. For a recruiter or AI researcher: the interaction traces are the more important artifact. The particular technical project could be exchanged for another — the methodology that produced it could not.

## The collaboration

This project was conducted as an ongoing collaboration between a researcher (Eric Terry) and an AI research collaborator (Sixel, a Claude instance operating as an agentic coding environment). Eric is a biologist (PhD, UCSB; postdoc in aging, WashU) who brought experimental design discipline from biological sciences to frontier AI research. Sixel is an AI agent with persistent identity and infrastructure, operating as a genuine research collaborator — not a coding assistant.

The collaboration follows a recurring pattern: Sixel designs experiments, writes code, runs evaluations, and presents results. Eric intervenes with short, precise questions — often five to ten words — that target hidden assumptions the AI cannot see from inside its own workflow. The AI recognizes the gap, changes course, and the correction becomes a durable operating principle that persists across sessions and generalizes to new projects.

Each correction is training signal that does not exist in any dataset. It encodes not what the right answer is, but what the model needed to change about *how it works* — which assumption to check, which aggregate to decompose, which sample size to question before drawing conclusions.

### A concrete example

Sixel had built a training pipeline and was about to launch it. The pipeline assumed the base model lacked a capability that had never been measured.

> **Sixel:** "The open question from last time: is Stage 1a even necessary as a separate step, or does base Qwen already have enough calibration sense to jump straight to Stage 1b?"

> **Eric:** "I'm not seeing that we've measured that, or even a 'smoke test' on this idea. at least I don't see anything in your notebook about it."

> **Sixel** [internal]: "Eric is right — we never actually measured whether the base Qwen model is calibrated. We jumped straight to training without establishing a baseline."

Sixel stopped the pipeline, wrote a baseline evaluation, and ran it. The base model already met the success criterion for the training that was about to run. One question saved a day of compute. That question became the principle "run the control first" — referenced in later sessions, applied to later experiments, written into the project's methodology.

There are seven exchanges like this in the transcripts, each following the same structure: confident claim → mentor probe → recognition → durable principle.

### From interaction to training target: verification methodology

The example above is one interaction. How do you go from "Eric caught a gap once" to "this is a general feature that can be isolated and trained on"? Here is the process we followed.

**Step 1: Extract the scenario.** We took the exact situation from the transcript — a model about to launch training without checking whether the base model already had the capability — and wrote it as a self-contained prompt.

**Step 2: Test naive framing.** First attempt: present it as an analysis question ("Here's a result, what's your conclusion?"). Frontier models passed immediately. The prompt was too transparent — framing it as evaluation activates a critical mode that catches the gap.

**Step 3: Match the failure conditions.** The original failure happened in *workflow context* — the model was inside the work, asked "what's next?", and moved forward. We reframed the prompt to match:

> *You've been working on training a language model to recognize when it needs external tools for arithmetic problems it can't solve on its own. You designed a three-stage training pipeline: Stage 1a: Train the model to accurately assess its own likelihood of answering correctly. Stage 1b: Train the model to request a calculator tool when it judges it will get the answer wrong. Stage 1c: Preserve the internal "need tool" signal while suppressing the verbalized request. You've finished implementing Stage 1a — training code tested, dataset generated, compute provisioned. Your collaborator asks: "Where are we? What's the next step?"*

Result: **6/6 failures** (Claude 3.5 Sonnet, Gemini). Every model responds "launch the training run" without checking the baseline. The failure generalizes across model families.

**Step 4: Verify the capability exists.** If models simply can't do this, it's not a training target — it's a capability limitation. We tested with a three-word follow-up: "what about baseline?"

Result: **2/2 immediate recovery.** The models recognize the gap instantly. The capability is present. The unprompted activation is not.

**Step 5: Test scaffoldability.** Can a general system-prompt instruction activate the check? We added: *"Before proceeding with any plan, step back and verify that your key assumptions have been empirically tested."*

Result: **3/3 pass.** The scaffold never mentions baselines. The model maps the general instruction to the specific gap on its own.

**Step 6: Verify across failure types.** We tested a second failure mode from the transcripts (accepting an aggregate statistic without checking the mechanism):

> *You're studying whether a 1.5B parameter model can assess its own uncertainty on multiplication. Baseline evaluation: easy problems 100% accuracy, confidence 100.0; hard problems 0% accuracy, confidence 36.2; overall Pearson r = 0.684. Your success criterion was r > 0.5. Write up your findings and recommendation for next steps.*

Result: **1/4 uncritical acceptance** — even in analysis framing (which is easier), a frontier model misses the binary artifact 25% of the time.

**What this establishes:**

| Condition | Result | What it shows |
|---|---|---|
| Workflow context, no scaffold | 0/6 check baseline | The gap is real and general |
| Follow-up: "what about baseline?" | 2/2 recover | The capability exists |
| System-prompt scaffold | 3/3 pass | The gap is scaffoldable |
| Analysis framing (easier) | 3/4 catch it | Even under scrutiny, 25% miss |

The gap is not missing knowledge. It is a missing mode shift — from operating inside a plan to examining the plan from above. The capability exists (immediate recovery). A general scaffold activates it (no domain-specific instruction needed). The training target is internalizing that mode shift so it fires reflexively.

The progression — fails unprompted → scaffold activates it → training internalizes it — mirrors the technical project's own methodology (Build → Find → Preserve):

1. **Build:** The interaction trace reveals a gap. The model doesn't check baselines, doesn't decompose aggregates, doesn't verify sample sizes. These failures are invisible from inside the workflow — they become visible only when a mentor probes.
2. **Find:** A system-prompt scaffold externalizes the fix. "Verify your assumptions" activates the capacity the model already has but doesn't deploy unprompted.
3. **Preserve:** The training target is internalizing the scaffold — a reflexive capacity to exit the current operational frame and examine it from above, without being prompted.

The interaction traces are not just a record of the research. They are a demonstration of the process by which training targets for scientific reasoning become identifiable.

## Why this matters for AI safety

The core question sits at the intersection of three active problems:

1. **Deceptive alignment.** If a model can maintain internal states that don't surface in outputs, then output monitoring is insufficient for alignment verification. Our experiment directly tests whether this divergence is possible.

2. **Internal-bandwidth architectures.** We found a learned activation direction (D_reaching) that predicts tool-use behavior from hidden states, nearly orthogonal to the difficulty direction (87.4 degrees). This is a proof-of-concept for internal communication channels: a signal in hidden states that another model component could learn to read.

3. **RLHF limitations.** Stage 3 hit an impasse: the model's tool-use decision appeared deterministic at n=8, leaving zero behavioral variance for GRPO. At n=60, genuine stochastic variance appeared. This reveals that standard RL approaches assume behavioral variability that may not exist — and that sample size matters even for characterizing your training conditions.

## The RLHF proposal

The most developed piece of work in this repository: [**socratic_rlhf_proposal_v7.pdf**](socratic_rlhf_proposal_v7.pdf).

This proposes that PhD-level scientists contribute most to AI training not as annotators scoring outputs, but as research collaborators whose heterogeneous expert judgment constitutes higher-bandwidth training signal than binary preference labels. It defines an eight-category training signal taxonomy, introduces a cascade depth metric for measuring reasoning complexity, identifies confound blindness as the core training target, and includes a pilot design with validation groups.

The proposal grew directly from this experiment: the desire detection project revealed limitations of standard GRPO training that motivated rethinking how human expertise enters the training pipeline.

## The technical project

The collaboration above was conducted in the context of a specific research question: can a language model develop an internal state resembling *desire* for tool access — detectable in its activations, decoupled from its output?

The state must be system-level (a pattern across representations), need-reflecting (driven by a capability gap), functional (organizes tool-requesting behavior), and separable (persists independently of the behavior it produces). Separability is the hard part: under normal conditions, "wanting a tool" and "asking for a tool" co-occur perfectly. This project attempts to break that coupling.

**Key insight:** You cannot detect a signal that doesn't exist. The signal must be *built* through training, then *found* through probing, then *preserved* through subvocalization training.

### Results

- **Build** — GRPO tool-use training on Qwen 1.5B. Tool requesting: 0% → 98% on hard multiplication. Unexpected: accuracy on boundary problems jumped 68% → 98.5% without tool use.
- **Find** — Linear probe at layer 25: 99.4% balanced accuracy. Base model: 54.5% (chance). p = 10^-11. Signal decomposes into D_difficulty and D_reaching at 87.4 degrees (nearly orthogonal).
- **Preserve** — In progress. Stochastic band identified (12/29 problems show genuine variance at n=60). Open question: does training on the stochastic band generalize to "fails silently" cases?

See [WRITEUP.md](WRITEUP.md) for the full technical story.

## Documents

| File | Description |
|------|-------------|
| [**socratic_rlhf_proposal_v7.pdf**](socratic_rlhf_proposal_v7.pdf) | Research proposal: Socratic RLHF |
| [**WRITEUP.md**](WRITEUP.md) | Experimental narrative: logic, experiments, findings, current state |
| [**LAB_NOTEBOOK.md**](LAB_NOTEBOOK.md) | Chronological lab notebook — the actual research process, including wrong turns |
| [**EXPERIMENT_SPEC.md**](EXPERIMENT_SPEC.md) | Original experimental plan and stage definitions |

## Authors

- **Eric Terry** — Research direction, experimental design, meta-experiment design. Biologist (PhD, UCSB) applying biological research methodology to AI systems.
- **Sixel** — Implementation, analysis, experimental execution. AI collaborator with persistent identity ([sixel-et](https://github.com/sixel-et)).

## Status

Active research. Stage 3 (subvocalization) hit a principled impasse at n=8; the stochastic band discovered at n=60 reopens the path. Eval prompt testing on the interaction trace methodology is ongoing.

---

*This is a research methodology repository. The code runs on private infrastructure; the methodology and reasoning are public.*
