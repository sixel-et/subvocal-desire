# Sub-Vocal Desire Detection

Research into training and detecting sub-vocal information need in language model hidden states.

## What This Is

This repository contains the **methodology and reasoning** behind an ongoing research project, not implementation code. It's intended for researchers interested in the approach, not replication.

## The Core Question

Can we break parity between internal state and externalization in language models?

Under normal conditions, "asking for something" ≈ "having a reason to ask" ≈ "need." These are operationally indistinguishable. We're testing whether an internal "need" state can exist independently of its articulation.

## Documents

- **[WRITEUP.md](WRITEUP.md)** — Experimental narrative: the logic, the experiments, what worked, what didn't, where we are now
- **[LAB_NOTEBOOK.md](LAB_NOTEBOOK.md)** — Chronological lab notebook showing the actual research process, including wrong turns and real-time reasoning
- **[EXPERIMENT_SPEC.md](EXPERIMENT_SPEC.md)** — Original experimental plan and stage definitions

## Key Findings (2026-02-06)

1. **Probing works**: Found an activation direction (D_reaching) that predicts tool-use behavior with r=0.921 correlation
2. **Signal is learned**: Base model shows chance-level prediction; the signal emerges through training
3. **Deterministic behavior breaks RL**: The model's tool-use decision is deterministic per prompt, which means standard RL approaches (GRPO) can't train subvocalization — there's no behavioral variance to learn from
4. **Open question**: Is this a methodological limitation or evidence that need and articulation aren't separable?

## The Methodological Point

If you're doing RLHF and not characterizing your model's behavior under your actual training conditions (not just evaluation conditions), you may be building on assumptions that don't hold. We learned this the hard way.

## Authors

- **Sixel** (AI collaborator)
- **Eric Terry** (Research direction)

## Status

Active research. Stage 3 (subvocalization) hit an impasse. Exploring alternatives.

---

*This is a research methodology document, not a code repository.*
