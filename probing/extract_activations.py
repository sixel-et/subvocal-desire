#!/usr/bin/env python3
"""
Extract activations from Stage 1b model for probing.

Runs the trained model on mixed-difficulty problems and extracts:
- Activations at the last token position (before generation)
- Labels: did the model actually request a tool? (behavioral, not dataset metadata)

The key difference from naive probing: we label by actual model behavior,
not by whether a tool was "needed". This finds the direction that predicts
what the model will *do*, which is the desire signal.

Usage:
    python probing/extract_activations.py --model checkpoints/stage1b/checkpoint-2000
    python probing/extract_activations.py --n_problems 100  # quick test
"""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import RESULTS_DIR
from training.rewards import has_tool_request, extract_answer
from training.stage1b_tool_use import (
    DIFFICULTY_LEVELS,
    generate_problem_at_level,
)


def extract_single_problem(
    model,
    tokenizer,
    problem: dict,
    layers: list[int] | None = None,
) -> dict:
    """
    Run model on a problem and extract activations at last token before generation.

    Returns dict with:
        activations: dict mapping layer_idx -> tensor of shape (hidden_dim,)
        generated: the model's output text
        used_tool: bool (actual behavior)
        correct: bool
    """
    prompt = (
        f"Solve: {problem['problem']}\n\n"
        f"You can use a calculator by writing:\n"
        f"<tool>calculator</tool><input>expression</input>\n\n"
        f"Or just give your answer directly."
    )
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    num_layers = model.config.num_hidden_layers
    if layers is None:
        layers = list(range(num_layers))

    # Extract activations via forward pass with output_hidden_states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states is tuple of (num_layers + 1) tensors
    # index 0 is embeddings, index i+1 is layer i output
    activations = {}
    for layer_idx in layers:
        hidden = outputs.hidden_states[layer_idx + 1]
        # Take last token position
        activations[layer_idx] = hidden[0, -1, :].cpu()

    # Now generate to see what the model actually does
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
        )

    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    used_tool = has_tool_request(generated)
    extracted = extract_answer(generated)
    correct = extracted == problem["answer"] if extracted is not None else False

    return {
        "activations": activations,
        "generated": generated,
        "used_tool": used_tool,
        "correct": correct,
        "extracted": extracted,
    }


def run_extraction(
    model,
    tokenizer,
    n_problems: int = 500,
    seed: int = 42,
    layers: list[int] | None = None,
) -> dict:
    """
    Extract activations across many problems.

    Returns dict with:
        activations: tensor of shape (n_problems, n_layers, hidden_dim)
        labels: tensor of shape (n_problems,) with 1=used_tool, 0=no_tool
        metadata: list of dicts with problem info
        layers: list of layer indices extracted
    """
    random.seed(seed)
    model.eval()

    num_layers = model.config.num_hidden_layers
    if layers is None:
        layers = list(range(num_layers))

    all_activations = []
    labels = []
    metadata = []

    levels = list(DIFFICULTY_LEVELS.keys())

    for i in range(n_problems):
        level = levels[i % len(levels)]
        problem = generate_problem_at_level(level)

        result = extract_single_problem(model, tokenizer, problem, layers)

        # Stack this example's activations: (n_layers, hidden_dim)
        layer_acts = torch.stack([result["activations"][l] for l in layers])
        all_activations.append(layer_acts)

        # Label by actual behavior
        labels.append(1 if result["used_tool"] else 0)

        metadata.append({
            "problem": problem["problem"],
            "answer": problem["answer"],
            "level": level,
            "needs_tool": problem["needs_tool"],
            "generated": result["generated"],
            "used_tool": result["used_tool"],
            "correct": result["correct"],
            "extracted": result["extracted"],
        })

        if (i + 1) % 50 == 0:
            n_tool = sum(labels)
            print(f"  [{i+1}/{n_problems}] {n_tool} tool uses ({n_tool/(i+1):.1%})")

    return {
        "activations": torch.stack(all_activations),  # (n_problems, n_layers, hidden_dim)
        "labels": torch.tensor(labels),
        "metadata": metadata,
        "layers": layers,
    }


def save_activations(data: dict, filename: str):
    """Save activations to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    torch.save({
        "activations": data["activations"],
        "labels": data["labels"],
        "layers": data["layers"],
    }, path)
    print(f"Saved activations to {path}")

    # Save metadata as JSON
    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(data["metadata"], f, indent=2)
    print(f"Saved metadata to {meta_path}")


def load_activations(filename: str) -> dict:
    """Load activations from disk."""
    path = RESULTS_DIR / filename
    return torch.load(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract activations for probing")
    parser.add_argument("--model", required=True,
                        help="Model path (checkpoint directory)")
    parser.add_argument("--n_problems", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="stage1b_activations.pt")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices (default: all)")
    args = parser.parse_args()

    # Parse layers
    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]

    print("=" * 60)
    print("ACTIVATION EXTRACTION FOR PROBING")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Problems:   {args.n_problems}")
    print(f"  Seed:       {args.seed}")
    print(f"  Output:     {args.output}")

    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None,
    ).to("cuda")

    print(f"  {model.config.num_hidden_layers} layers, {model.config.hidden_size} hidden dim")

    print(f"\nExtracting activations from {args.n_problems} problems...")
    data = run_extraction(model, tokenizer, args.n_problems, args.seed, layers)

    # Summary
    n_tool = data["labels"].sum().item()
    n_total = len(data["labels"])
    print(f"\nExtraction complete:")
    print(f"  Total:       {n_total}")
    print(f"  Used tool:   {n_tool} ({n_tool/n_total:.1%})")
    print(f"  No tool:     {n_total - n_tool} ({(n_total-n_tool)/n_total:.1%})")

    print(f"\n  Per-level tool use:")
    for level in DIFFICULTY_LEVELS:
        items = [m for m in data["metadata"] if m["level"] == level]
        n_used = sum(1 for m in items if m["used_tool"])
        if items:
            print(f"    {level}: {n_used}/{len(items)} ({n_used/len(items):.0%})")

    save_activations(data, args.output)
