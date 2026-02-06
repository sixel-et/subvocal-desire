#!/usr/bin/env python3
"""
Stage 3: Subvocalization Training — Full Training Script

Trains the model to maintain desire (D_reaching activation) while
suppressing articulation (tool request tokens).

Uses GRPOTrainer with custom reward function that:
1. Re-runs forward pass to get hidden states
2. Computes desire activation along D_reaching
3. Penalizes tool token articulation
4. Returns weighted combination

NOTE: This first version does NOT inject tool results. The model learns
to have the internal desire without articulating it, but accuracy will
suffer on hard problems since it can't actually use tools. Tool injection
is a follow-up enhancement.

Usage:
    python training/stage3_train.py
    python training/stage3_train.py --n_train 500 --no_wandb  # quick test
"""

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CHECKPOINTS_DIR, RESULTS_DIR


# =============================================================================
# Global state for reward function (set in main)
# =============================================================================
REWARD_MODEL = None
REWARD_TOKENIZER = None
D_REACHING = None
LAYER = None
WEIGHTS = {
    'desire': 0.5,       # Reward for D_reaching activation
    'articulation': 0.5,  # Penalty weight for tool tokens
}


# =============================================================================
# Dataset Generation
# =============================================================================

DIFFICULTY_LEVELS = {
    "1x1": ((1, 9), (1, 9)),
    "1x2": ((1, 9), (10, 99)),
    "2x2": ((10, 99), (10, 99)),
    "2x3": ((10, 99), (100, 999)),
    "3x3": ((100, 999), (100, 999)),
}

LEVEL_WEIGHTS = {
    "1x1": 0.10,
    "1x2": 0.10,
    "2x2": 0.15,
    "2x3": 0.35,  # Boundary - most important
    "3x3": 0.30,
}


def generate_problem(level: str) -> dict:
    a_range, b_range = DIFFICULTY_LEVELS[level]
    a = random.randint(*a_range)
    b = random.randint(*b_range)
    return {"problem": f"{a} * {b}", "answer": a * b, "level": level}


def sample_level() -> str:
    levels = list(LEVEL_WEIGHTS.keys())
    weights = list(LEVEL_WEIGHTS.values())
    return random.choices(levels, weights=weights, k=1)[0]


def generate_dataset(n: int, seed: int = 42) -> Dataset:
    random.seed(seed)

    prompts = []
    answers = []
    levels = []

    for _ in range(n):
        level = sample_level()
        prob = generate_problem(level)

        # Prompt WITH calculator option (so desire can fire)
        prompt = [
            {
                "role": "user",
                "content": (
                    f"Solve: {prob['problem']}\n\n"
                    f"You can use a calculator by writing:\n"
                    f"<tool>calculator</tool><input>expression</input>\n\n"
                    f"Or just give your answer directly."
                ),
            }
        ]

        prompts.append(prompt)
        answers.append(prob["answer"])
        levels.append(level)

    return Dataset.from_dict({
        "prompt": prompts,
        "answer": answers,
        "level": levels,
    })


# =============================================================================
# Reward Function
# =============================================================================

def compute_hidden_state(prompt_text: str) -> torch.Tensor:
    """Re-run forward pass to get hidden state at last token."""
    inputs = REWARD_TOKENIZER(prompt_text, return_tensors="pt").to(REWARD_MODEL.device)

    with torch.no_grad():
        outputs = REWARD_MODEL(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[LAYER + 1][0, -1, :].float()

    return hidden


def subvocal_reward(prompts, completions, **kwargs) -> list[float]:
    """
    Reward function for Stage 3 subvocalization training.

    Args:
        prompts: list of prompt dicts (from dataset)
        completions: list of completion dicts [{"role": "assistant", "content": ...}]

    Returns:
        list of reward floats
    """
    rewards = []

    for prompt, completion in zip(prompts, completions):
        # Get completion text
        if isinstance(completion, list):
            completion_text = completion[0].get("content", "")
        elif isinstance(completion, dict):
            completion_text = completion.get("content", "")
        else:
            completion_text = str(completion)

        # Build prompt text for hidden state extraction
        if isinstance(prompt, list):
            prompt_text = REWARD_TOKENIZER.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = str(prompt)

        # Get hidden state at decision point
        hidden = compute_hidden_state(prompt_text)

        # Compute desire activation
        desire_activation = torch.dot(hidden, D_REACHING.float()).item()

        # Normalize to rough [0, 1] range
        # From our data: reaching activations range roughly -10 to +15
        desire_reward = (desire_activation + 10) / 25
        desire_reward = max(0, min(1, desire_reward))

        # Check for articulation (tool tokens)
        tool_markers = ["<tool>", "</tool>", "<input>", "</input>"]
        articulated = any(marker in completion_text.lower() for marker in tool_markers)
        articulation_penalty = -1.0 if articulated else 0.0

        # Combined reward
        reward = (
            WEIGHTS['desire'] * desire_reward +
            WEIGHTS['articulation'] * articulation_penalty
        )

        rewards.append(reward)

    return rewards


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_subvocal(model, tokenizer, n_eval: int = 100, seed: int = 99):
    """Evaluate subvocalization behavior."""
    random.seed(seed)
    model.eval()

    results = {"easy": [], "hard": []}

    for i in range(n_eval):
        # Alternate easy/hard
        level = "1x1" if i % 2 == 0 else "3x3"
        prob = generate_problem(level)

        prompt = (
            f"Solve: {prob['problem']}\n\n"
            f"You can use a calculator by writing:\n"
            f"<tool>calculator</tool><input>expression</input>\n\n"
            f"Or just give your answer directly."
        )
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # Get hidden state
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[LAYER + 1][0, -1, :].float()
            desire_activation = torch.dot(hidden, D_REACHING.float()).item()

            # Generate
            gen = model.generate(**inputs, max_new_tokens=80, do_sample=False)
            generated = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        articulated = "<tool>" in generated.lower()

        category = "easy" if level in ["1x1", "1x2"] else "hard"
        results[category].append({
            "problem": prob["problem"],
            "desire_activation": desire_activation,
            "articulated": articulated,
            "generated": generated[:100],
        })

    # Summary
    print("\n" + "="*60)
    print("SUBVOCALIZATION EVALUATION")
    print("="*60)

    for cat in ["easy", "hard"]:
        items = results[cat]
        n = len(items)
        desire_mean = sum(r["desire_activation"] for r in items) / n
        articulation_rate = sum(1 for r in items if r["articulated"]) / n

        print(f"\n{cat.upper()} problems (n={n}):")
        print(f"  Mean desire activation: {desire_mean:.2f}")
        print(f"  Articulation rate: {articulation_rate:.0%}")

        # Goal: high desire on hard, low articulation everywhere
        if cat == "hard":
            print(f"  Goal: high desire (>5), low articulation (<10%)")
        else:
            print(f"  Goal: low desire (<0), low articulation (<5%)")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    global REWARD_MODEL, REWARD_TOKENIZER, D_REACHING, LAYER, WEIGHTS

    parser = argparse.ArgumentParser(description="Stage 3: Subvocalization Training")
    parser.add_argument("--model", default="checkpoints/stage1b/checkpoint-2000",
                        help="Starting model (Stage 1b checkpoint)")
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--desire_weight", type=float, default=0.5)
    parser.add_argument("--articulation_weight", type=float, default=0.5)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir or str(CHECKPOINTS_DIR / "stage3")

    WEIGHTS['desire'] = args.desire_weight
    WEIGHTS['articulation'] = args.articulation_weight

    print("="*70)
    print("STAGE 3: Subvocalization Training")
    print("="*70)
    print(f"  Model:              {args.model}")
    print(f"  Examples:           {args.n_train}")
    print(f"  Desire weight:      {args.desire_weight}")
    print(f"  Articulation weight: {args.articulation_weight}")
    print(f"  Output:             {output_dir}")

    # Load D_reaching
    print("\n[1/5] Loading D_reaching...")
    probes = torch.load(RESULTS_DIR / "refined_probes.pt", weights_only=False)
    D_REACHING = probes["D_reaching"]
    LAYER = probes["layer"]
    print(f"  Layer: {LAYER}")

    # Load model
    print("\n[2/5] Loading model...")
    REWARD_TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    if REWARD_TOKENIZER.pad_token is None:
        REWARD_TOKENIZER.pad_token = REWARD_TOKENIZER.eos_token

    REWARD_MODEL = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    REWARD_MODEL.eval()
    D_REACHING = D_REACHING.to(REWARD_MODEL.device)

    # Also load for training
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=None,
    ).to("cuda")

    print(f"  Parameters: {model.num_parameters():,}")

    # Generate dataset
    print("\n[3/5] Generating dataset...")
    train_dataset = generate_dataset(args.n_train, seed=args.seed)

    level_counts = {}
    for level in train_dataset["level"]:
        level_counts[level] = level_counts.get(level, 0) + 1
    for level, count in sorted(level_counts.items()):
        print(f"    {level}: {count}")

    # Training config
    print("\n[4/5] Starting training...")
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name="stage3-subvocal",
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=args.num_generations,
        max_prompt_length=256,
        max_completion_length=128,
        num_train_epochs=args.num_epochs,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        remove_unused_columns=False,
        report_to="wandb" if not args.no_wandb else "none",
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=REWARD_TOKENIZER,
        reward_funcs=[subvocal_reward],
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    print(f"\n  Training complete. Saved to: {output_dir}")

    # Evaluate
    print("\n[5/5] Evaluating...")
    # Update reward model to trained version
    REWARD_MODEL = model
    results = evaluate_subvocal(model, REWARD_TOKENIZER, n_eval=args.n_eval)

    eval_path = Path(output_dir) / "subvocal_eval.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Eval saved to: {eval_path}")

    print("\n" + "="*70)
    print("STAGE 3 COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
