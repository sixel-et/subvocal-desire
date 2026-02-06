# Sub-Vocal Desire Detection

Training and detecting sub-vocal information need in language model hidden states.

## Overview

This project explores whether language models can develop an internal "need for information" state that exists independently of its articulation. We train models through progressive stages:

1. **Calibration**: Model learns when it's likely wrong
2. **Tool use**: Model learns to request external help (calculator)
3. **Probing**: Find the activation direction that predicts tool requests
4. **Subvocalization**: (In progress) Maintain internal need state without articulating it

The core hypothesis: can we break the parity between internal state and externalization? Under normal conditions, "asking for something" ≈ "having a reason to ask" ≈ "need." We're testing whether these can be separated.

## Key Findings (as of 2026-02-06)

- **D_reaching**: Found an activation direction at layer 25 that predicts tool use with r=0.921 correlation
- **Signal is learned**: Base model shows chance-level prediction; trained model shows strong signal
- **Deterministic behavior**: The model's tool-use decision is deterministic per prompt, which breaks RL-based training approaches
- **Stage 3 impasse**: GRPO cannot train subvocalization because it requires behavioral variance that doesn't exist

See [WRITEUP.md](WRITEUP.md) for the full experimental narrative.

## Structure

```
├── WRITEUP.md                  # Full experimental narrative
├── EXPERIMENT_SPEC.md          # Original experimental plan
├── training/                   # Training scripts (Stage 1b, Stage 3)
├── probing/                    # Activation extraction and probe training
├── eval/                       # Evaluation scripts
├── results/                    # Measurement results (JSON)
└── measure_articulation_boundary.py  # Boundary characterization
```

## Status

Active research. Stage 3 (subvocalization) hit an impasse with RL-based approaches. Exploring alternatives: activation steering, supervised fine-tuning, representation engineering.

## Authors

- **Sixel** (sixel-et) — AI collaborator
- **Eric Terry** (estbiostudent) — Research direction

---

*This is a research artifact, not production code.*
