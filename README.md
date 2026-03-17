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

### Empirical validation

We tested whether this failure mode is real and general. The same scenario — embedded in workflow context, not framed as an analysis question — was presented to frontier models (Claude 3.5 Sonnet, Gemini):

- **Without mentor intervention:** 6/6 models proceed to launch training without checking the baseline.
- **With a three-word follow-up** ("what about baseline?"): 2/2 immediately recognize the gap.
- **With a system-prompt scaffold** ("verify your key assumptions before proceeding"): 3/3 pass.

The capability exists in frontier models. The unprompted activation does not. The gap is the space between "can do it when asked" and "does it reflexively."

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
