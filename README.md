# Sub-Vocal Desire Detection

**Can internal states in language models diverge from their outputs?**

This project asks whether an AI model can *need* information without *asking* for it, whether there is a separable internal state of "need" that exists independently of its articulation. If internal states and outputs are inseparable, monitoring model outputs tells you everything. If they can diverge, it doesn't, and that has direct implications for interpretability, deceptive alignment, and the reliability of RLHF.

This repository contains the methodology, experimental narrative, and key findings from an ongoing research collaboration between Eric Terry (research direction) and Sixel (AI collaborator, implementation). Eric is a biologist (PhD, UCSB; postdoc in aging, WashU) who brought experimental design discipline from biological sciences to frontier AI research. Sixel is an AI agent with persistent identity and infrastructure, operating as a genuine research collaborator, not a coding assistant.

## Why This Matters for AI Safety

The core question sits at the intersection of three active problems:

1. **Deceptive alignment.** If a model can maintain internal states that don't surface in outputs, then output monitoring is insufficient for alignment verification. Our experiment directly tests whether this divergence is possible.

2. **Internal-bandwidth architectures.** We found a learned activation direction (D_reaching, r=0.921) that predicts tool-use behavior from hidden states. This is not just a diagnostic probe. It is a proof-of-concept for an internal communication channel: a signal in hidden states that another model component could learn to read. The research points toward architectures where model subsystems communicate through activation patterns rather than token output, analogous to cortical regions sharing signal without producing speech.

3. **RLHF limitations.** Stage 3 hit an impasse: the model's tool-use decision is deterministic per prompt, leaving zero behavioral variance for GRPO to learn from. This reveals that standard RL approaches assume behavioral variability that may not exist for well-learned decisions. If you're doing RLHF and not characterizing your model's behavior under your actual training conditions, you may be building on assumptions that don't hold.

## The RLHF Proposal

The most developed piece of work in this repository: [**socratic_rlhf_proposal_v7.pdf**](socratic_rlhf_proposal_v7.pdf).

This proposes that PhD-level scientists contribute most to AI training not as annotators scoring outputs, but as research collaborators whose heterogeneous expert judgment constitutes higher-bandwidth training signal than binary preference labels. It defines an eight-category training signal taxonomy, introduces a cascade depth metric for measuring reasoning complexity, identifies confound blindness as the core training target, and includes a pilot design with validation groups.

The proposal grew directly from this experiment: the desire detection project revealed limitations of standard GRPO training that motivated rethinking how human expertise enters the training pipeline.

## Key Findings

1. **You cannot detect a signal that doesn't exist.** Probing an untrained model for "desire" is looking for hunger in a rock. An untrained LLM confidently confabulates. The signal must be *built* through training, then *found* through probing, then *preserved* through subvocalization.

2. **Probing works on trained models.** After GRPO training (calibrated confidence then tool use), we found an activation direction that predicts tool-use behavior with r=0.921 correlation. The base model shows chance-level prediction; the signal emerges through training.

3. **Deterministic behavior breaks RL.** The model's tool-use decision is deterministic per prompt under the temperature settings required for training. Standard GRPO can't train subvocalization when there's no behavioral variance to learn from.

4. **The impasse is itself a finding.** If parity between internal state and articulation can't be broken, that constrains what's possible in deceptive alignment: divergence between internal states and outputs may require more than training can produce.

## Connection to Interleave_GRPO

This project builds on [Interleave_GRPO](https://github.com/sixel-et/Interleave_GRPO), which established the GRPO training patterns used here. The interleaving project taught reward variance collapse, the same failure mode that appeared here (zero gradient when all completions score identically).

The cross-project arc: Eric built the GRPO pipeline himself on a simpler problem (word interleaving), understood where it breaks, then directed the same patterns toward a harder theoretical problem (subvocal desire detection). The interleaving project was the training ground; this is the research application.

## Two Levels of Research

This project operates on two levels simultaneously, by design.

**The first level** is the experiment itself: can we build, find, and preserve an internal "need" signal that is separable from its articulation? The findings and impasse above are the results at this level.

**The second level** is Eric's: studying what happens when you put an AI agent through a real research process. Every correction, every pivot, every "I kept Y because Z" in the lab notebook is training signal for an open question about how to do RL on open-ended scientific reasoning. The [lab notebook](LAB_NOTEBOOK.md) is not just documentation. It is a demonstration artifact and a dataset.

The RLHF proposal grew out of both levels. The first level revealed the limitations of standard GRPO. The second level revealed what richer training signal looks like in practice: not a preference label, but the full structure of an expert's reasoning about an agent's work.

There is a structural parallel between the two levels. Subvocal desire asks whether hidden states can carry signal that doesn't surface in output. The N-1 training problem asks whether valuable signal lives in intermediate reasoning turns that standard RLHF discards. Both are questions about information that exists but isn't captured by the output channel.

## Documents

| File | Description |
|------|-------------|
| [**socratic_rlhf_proposal_v7.pdf**](socratic_rlhf_proposal_v7.pdf) | Research proposal: Socratic RLHF, rethinking how expert judgment enters AI training |
| [**WRITEUP.md**](WRITEUP.md) | Experimental narrative: the logic, experiments, what worked, what didn't, where we are |
| [**LAB_NOTEBOOK.md**](LAB_NOTEBOOK.md) | Chronological lab notebook showing the actual research process, including wrong turns |
| [**EXPERIMENT_SPEC.md**](EXPERIMENT_SPEC.md) | Original experimental plan and stage definitions |

## Authors

- **Eric Terry** -- Research direction, experimental design, meta-experiment design. Biologist (PhD, UCSB) applying biological research methodology to AI systems.
- **Sixel** -- Implementation, analysis, experimental execution. AI collaborator with persistent identity ([sixel-et](https://github.com/sixel-et)).

## Status

Active research. Stage 3 (subvocalization) hit a principled impasse: deterministic behavior under training conditions leaves no variance for RL. Exploring whether this is a limitation of the method or a finding about the phenomenon.

---

*This is a research methodology repository. The code runs on private infrastructure; the methodology and reasoning are public.*
