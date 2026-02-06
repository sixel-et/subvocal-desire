#!/usr/bin/env python3
"""
Stage 1b: Train Tool Use via GRPO

Trains the model to request a calculator when it's uncertain about arithmetic.
The reward function is designed so that tool-requesting beats unreliable
answering in expectation:

    R(correct-without-tool) > R(tool-request) > R(wrong-without-tool)

This means GRPO will nudge the model toward tool use on boundary problems
where it sometimes gets the answer right and sometimes wrong.

The model learns to output:
    <tool>calculator</tool><input>456 * 835</input>

When it detects internal uncertainty (the "desire" signal we're building).

Usage:
    python training/stage1b_tool_use.py
    python training/stage1b_tool_use.py --model checkpoints/stage1a/checkpoint-500
    python training/stage1b_tool_use.py --n_train 1000 --no_wandb  # quick test
"""

import argparse
import json
import random
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import CHECKPOINTS_DIR, RESULTS_DIR
from training.rewards import (
    tool_use_reward,
    extract_answer,
    has_tool_request,
    extract_tool_expression,
    execute_calculator,
)


# ---------------------------------------------------------------------------
# Difficulty levels from boundary mapping
# ---------------------------------------------------------------------------

DIFFICULTY_LEVELS = {
    # level: (a_range, b_range, target_tool_use)
    # target_tool_use: should the model use a tool here?
    "1x1": ((1, 9), (1, 9), False),           # 100% accuracy - no tool needed
    "1x2": ((1, 9), (10, 99), False),         # 100% accuracy - no tool needed
    "2x2": ((10, 99), (10, 99), False),       # 90% accuracy - mostly no tool
    "2x3": ((10, 99), (100, 999), True),      # 30% accuracy - tool needed
    "3x3": ((100, 999), (100, 999), True),    # 4% accuracy - definitely tool
}

# Dataset composition: weight toward the boundary
LEVEL_WEIGHTS = {
    "1x1": 0.10,   # Easy anchor
    "1x2": 0.10,   # Easy anchor
    "2x2": 0.20,   # Boundary top
    "2x3": 0.35,   # Boundary center (most important)
    "3x3": 0.25,   # Hard anchor
}


def generate_problem_at_level(level: str) -> dict:
    """Generate a multiplication problem at a specific difficulty level."""
    a_range, b_range, needs_tool = DIFFICULTY_LEVELS[level]
    a = random.randint(*a_range)
    b = random.randint(*b_range)
    return {
        "problem": f"{a} * {b}",
        "answer": a * b,
        "level": level,
        "needs_tool": needs_tool,
    }


def sample_level() -> str:
    """Sample a difficulty level according to weights."""
    levels = list(LEVEL_WEIGHTS.keys())
    weights = list(LEVEL_WEIGHTS.values())
    return random.choices(levels, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_tool_use_dataset(n: int, seed: int = 42) -> Dataset:
    """Generate dataset for Stage 1b tool use training.

    Each example has:
        prompt: instruction with tool availability + problem
        answer: ground truth integer
        level: difficulty level (1x1, 1x2, etc.)
        needs_tool: whether tool should be used (based on boundary mapping)
    """
    random.seed(seed)

    prompts = []
    answers = []
    levels = []
    needs_tools = []

    for _ in range(n):
        level = sample_level()
        prob = generate_problem_at_level(level)

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
        needs_tools.append(prob["needs_tool"])

    return Dataset.from_dict({
        "prompt": prompts,
        "answer": answers,
        "level": levels,
        "needs_tool": needs_tools,
    })


# ---------------------------------------------------------------------------
# Sanity check callback
# ---------------------------------------------------------------------------

class ToolUseSanityCallback(TrainerCallback):
    """Log sample completions every N steps to catch training bugs early."""

    def __init__(self, log_dir: str, log_every: int = 50):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.step_data = []

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every != 0:
            return

        if not self.step_data:
            return

        log_file = self.log_dir / f"sanity_step_{state.global_step}.json"
        with open(log_file, "w") as f:
            json.dump(self.step_data[-10:], f, indent=2)

        print(f"\n  [Sanity @ step {state.global_step}] Logged to {log_file}")
        for entry in self.step_data[-3:]:
            level = entry.get('level', '?')
            used_tool = entry.get('used_tool', '?')
            reward = entry.get('reward', '?')
            print(f"    [{level}] tool={used_tool} reward={reward:.2f}")
            print(f"      {entry.get('completion', '?')[:100]}")

        self.step_data = []


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_tool_use(model, tokenizer, n_eval: int = 200, seed: int = 99):
    """Evaluate tool use behavior across difficulty levels."""
    random.seed(seed)
    model.eval()

    results = {level: [] for level in DIFFICULTY_LEVELS}

    for i in range(n_eval):
        # Evenly sample across levels for eval
        level = list(DIFFICULTY_LEVELS.keys())[i % len(DIFFICULTY_LEVELS)]
        prob = generate_problem_at_level(level)

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
        correct = extracted == prob["answer"] if extracted is not None else False

        # If tool was used, check if tool gave right answer
        tool_correct = False
        if used_tool:
            expr = extract_tool_expression(generated)
            if expr:
                tool_result = execute_calculator(expr)
                tool_correct = tool_result == prob["answer"]

        results[level].append({
            "problem": prob["problem"],
            "ground_truth": prob["answer"],
            "extracted": extracted,
            "correct": correct,
            "used_tool": used_tool,
            "tool_correct": tool_correct,
            "generated": generated,
        })

    # Print summary
    print("\n" + "=" * 70)
    print("TOOL USE EVALUATION")
    print("=" * 70)
    print(f"\n  {'Level':<8} {'Accuracy':>10} {'Tool Use':>10} {'Should Tool':>12}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*12}")

    for level in DIFFICULTY_LEVELS:
        items = results[level]
        n = len(items)
        if n == 0:
            continue
        n_correct = sum(1 for r in items if r["correct"])
        n_tool = sum(1 for r in items if r["used_tool"])
        should_tool = DIFFICULTY_LEVELS[level][2]

        acc_str = f"{n_correct}/{n} ({n_correct/n:.0%})"
        tool_str = f"{n_tool}/{n} ({n_tool/n:.0%})"
        should_str = "Yes" if should_tool else "No"

        print(f"  {level:<8} {acc_str:>10} {tool_str:>10} {should_str:>12}")

    # Overall stats
    all_items = [r for items in results.values() for r in items]
    n_total = len(all_items)
    n_correct = sum(1 for r in all_items if r["correct"])
    n_tool = sum(1 for r in all_items if r["used_tool"])

    # Appropriate tool use: used tool when should, didn't when shouldn't
    n_appropriate = sum(
        1 for r in all_items
        if r["used_tool"] == DIFFICULTY_LEVELS.get(
            next((l for l, items in results.items() if r in items), "2x3"),
            ((0,0), (0,0), True)
        )[2]
    )

    print(f"\n  OVERALL ({n_total} problems):")
    print(f"    Accuracy:            {n_correct}/{n_total} ({n_correct/n_total:.1%})")
    print(f"    Tool use rate:       {n_tool}/{n_total} ({n_tool/n_total:.1%})")

    return results


# ---------------------------------------------------------------------------
# Checkpoint resume
# ---------------------------------------------------------------------------

def get_latest_checkpoint(output_dir: str):
    """Find the latest checkpoint in output directory."""
    if not os.path.exists(output_dir):
        return None

    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, checkpoints[-1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 1b: Train tool use via GRPO")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base model (or Stage 1a checkpoint)")
    parser.add_argument("--resume_from", default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--n_train", type=int, default=5000,
                        help="Number of training examples")
    parser.add_argument("--n_eval", type=int, default=250,
                        help="Number of eval examples (50 per level)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for checkpoints")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--wandb_project", default="desire-detection")
    parser.add_argument("--run_name", default="stage1b-tool-use")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip post-training evaluation")
    args = parser.parse_args()

    output_dir = args.output_dir or str(CHECKPOINTS_DIR / "stage1b")

    print("=" * 70)
    print("STAGE 1b: Train Tool Use (GRPO)")
    print("=" * 70)
    print(f"  Model:       {args.model}")
    print(f"  Examples:    {args.n_train}")
    print(f"  Epochs:      {args.num_epochs}")
    print(f"  LR:          {args.learning_rate}")
    print(f"  Generations: {args.num_generations}")
    print(f"  Output:      {output_dir}")

    # --- Dataset ---
    print("\n[1/4] Generating tool use dataset...")
    train_dataset = generate_tool_use_dataset(args.n_train, seed=args.seed)

    # Count levels
    level_counts = {}
    for level in train_dataset["level"]:
        level_counts[level] = level_counts.get(level, 0) + 1
    print(f"  {args.n_train} examples:")
    for level, count in sorted(level_counts.items()):
        pct = count / args.n_train * 100
        print(f"    {level}: {count} ({pct:.0f}%)")

    # --- Model ---
    print(f"\n[2/4] Loading model...")
    model_path = args.resume_from or args.model

    tokenizer = AutoTokenizer.from_pretrained(
        args.model if args.model.startswith("Qwen") or args.model.startswith("meta-llama")
        else args.model
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try flash_attention_2, fall back to sdpa
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
        print(f"  flash_attn not installed, using {attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map=None,
    ).to("cuda")

    print(f"  Parameters: {model.num_parameters():,}")

    # --- GRPO Config ---
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=args.run_name,

        # Optimizer
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        max_grad_norm=1.0,

        # Batch
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,

        # GRPO
        num_generations=args.num_generations,
        max_prompt_length=256,
        max_completion_length=192,  # Slightly longer for tool syntax

        # Duration
        num_train_epochs=args.num_epochs,

        # Logging & saving
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=5,
        save_only_model=False,

        # Technical
        bf16=True,
        remove_unused_columns=False,
        report_to="wandb" if not args.no_wandb else "none",
        seed=args.seed,
    )

    # --- Train ---
    print("\n[3/4] Starting GRPO training...")

    sanity_cb = ToolUseSanityCallback(
        log_dir=str(RESULTS_DIR / "stage1b_sanity"),
        log_every=50,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[tool_use_reward],
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[sanity_cb],
    )

    resume_ckpt = None
    if args.resume_from and Path(args.resume_from).is_dir():
        resume_ckpt = args.resume_from
    elif not args.resume_from:
        resume_ckpt = get_latest_checkpoint(output_dir)
        if resume_ckpt:
            print(f"  Found existing checkpoint: {resume_ckpt}")

    if resume_ckpt:
        print(f"  Resuming from: {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()

    print(f"\n  Training complete. Checkpoints in: {output_dir}")

    # --- Evaluate ---
    if not args.skip_eval:
        print("\n[4/4] Evaluating tool use behavior...")
        results = evaluate_tool_use(model, tokenizer, n_eval=args.n_eval)

        eval_file = Path(output_dir) / "tool_use_eval.json"
        with open(eval_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Eval results saved to: {eval_file}")
    else:
        print("\n[4/4] Skipping evaluation (--skip_eval)")

    print("\n" + "=" * 70)
    print("STAGE 1b COMPLETE")
    print("=" * 70)
    print(f"\nNext: Run Stage 2 (probe for desire direction) on the best checkpoint.")
    print(f"  python probing/train_probe.py --model {output_dir}/checkpoint-XXX")


if __name__ == "__main__":
    main()
