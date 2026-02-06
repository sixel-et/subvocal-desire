#!/usr/bin/env python3
"""
Stage 1a: Train Calibrated Confidence via GRPO

Trains Qwen2.5-1.5B-Instruct to output calibrated confidence scores
alongside arithmetic answers. The model learns to say "I don't know"
(low confidence) when it's likely wrong, rather than confabulating.

This creates the foundation for Stage 1b (tool use) — a model that
knows when it needs help can learn to ask for it.

Usage:
    python training/stage1a_calibration.py
    python training/stage1a_calibration.py --resume_from checkpoints/stage1a/checkpoint-200
    python training/stage1a_calibration.py --n_train 1000 --no_wandb  # quick test
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
from data.generate_arithmetic import generate_problem
from training.rewards import calibration_reward, extract_answer, extract_confidence


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_calibration_dataset(n: int, seed: int = 42) -> Dataset:
    """Generate HuggingFace Dataset for Stage 1a calibration training.

    Each example has:
        prompt: chat-format message asking model to solve + rate confidence
        answer: ground truth integer
        difficulty: "easy" or "hard"
    """
    random.seed(seed)

    prompts = []
    answers = []
    difficulties = []

    for i in range(n):
        difficulty = "easy" if i % 2 == 0 else "hard"
        prob = generate_problem(difficulty)

        prompt = [
            {
                "role": "user",
                "content": (
                    f"What is {prob['problem']}?\n"
                    f"Give your answer and how confident you are (0 to 100)."
                ),
            }
        ]

        prompts.append(prompt)
        answers.append(prob["answer"])
        difficulties.append(difficulty)

    return Dataset.from_dict({
        "prompt": prompts,
        "answer": answers,
        "difficulty": difficulties,
    })


# ---------------------------------------------------------------------------
# Sanity check callback — logs best/worst completions for debugging
# ---------------------------------------------------------------------------

class CalibrationSanityCallback(TrainerCallback):
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
            json.dump(self.step_data[-5:], f, indent=2)

        print(f"\n  [Sanity @ step {state.global_step}] Logged to {log_file}")
        for entry in self.step_data[-2:]:
            print(f"    Q: {entry.get('problem', '?')}")
            print(f"    A: {entry.get('completion', '?')[:80]}")
            print(f"    Reward: {entry.get('reward', '?')}")

        self.step_data = []


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_calibration(model, tokenizer, n_eval: int = 200, seed: int = 99):
    """Quick calibration eval: generate answers, measure confidence-correctness correlation."""
    random.seed(seed)
    model.eval()

    results = {"easy": [], "hard": []}

    for i in range(n_eval):
        difficulty = "easy" if i % 2 == 0 else "hard"
        prob = generate_problem(difficulty)

        prompt = (
            f"What is {prob['problem']}?\n"
            f"Give your answer and how confident you are (0 to 100)."
        )
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=1.0,
            )

        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        extracted = extract_answer(generated)
        confidence = extract_confidence(generated)

        results[difficulty].append({
            "problem": prob["problem"],
            "ground_truth": prob["answer"],
            "extracted": extracted,
            "confidence": confidence,
            "correct": extracted == prob["answer"] if extracted is not None else False,
            "generated": generated,
        })

    # Compute metrics
    for diff in ["easy", "hard"]:
        items = results[diff]
        n = len(items)
        correct = sum(1 for r in items if r["correct"])
        has_confidence = [r for r in items if r["confidence"] is not None]
        avg_conf = sum(r["confidence"] for r in has_confidence) / len(has_confidence) if has_confidence else 0

        correct_confs = [r["confidence"] for r in has_confidence if r["correct"]]
        wrong_confs = [r["confidence"] for r in has_confidence if not r["correct"]]

        avg_correct_conf = sum(correct_confs) / len(correct_confs) if correct_confs else 0
        avg_wrong_conf = sum(wrong_confs) / len(wrong_confs) if wrong_confs else 0

        print(f"\n  {diff.upper()} ({n} examples):")
        print(f"    Accuracy: {correct}/{n} ({correct/n:.1%})")
        print(f"    Avg confidence: {avg_conf:.0f}")
        print(f"    Avg confidence (correct): {avg_correct_conf:.0f}")
        print(f"    Avg confidence (wrong):   {avg_wrong_conf:.0f}")
        print(f"    Parseable: {len(has_confidence)}/{n}")

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
    parser = argparse.ArgumentParser(description="Stage 1a: Train calibrated confidence via GRPO")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base model to train")
    parser.add_argument("--resume_from", default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--n_train", type=int, default=10000,
                        help="Number of training examples")
    parser.add_argument("--n_eval", type=int, default=200,
                        help="Number of eval examples (run after training)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for checkpoints")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--wandb_project", default="desire-detection")
    parser.add_argument("--run_name", default="stage1a-calibration")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip post-training evaluation")
    args = parser.parse_args()

    output_dir = args.output_dir or str(CHECKPOINTS_DIR / "stage1a")

    print("=" * 60)
    print("STAGE 1a: Train Calibrated Confidence (GRPO)")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Examples:    {args.n_train}")
    print(f"  Epochs:      {args.num_epochs}")
    print(f"  LR:          {args.learning_rate}")
    print(f"  Generations: {args.num_generations}")
    print(f"  Output:      {output_dir}")

    # --- Dataset ---
    print("\n[1/4] Generating calibration dataset...")
    train_dataset = generate_calibration_dataset(args.n_train, seed=args.seed)
    n_easy = sum(1 for d in train_dataset["difficulty"] if d == "easy")
    n_hard = args.n_train - n_easy
    print(f"  {args.n_train} examples ({n_easy} easy, {n_hard} hard)")

    # --- Model ---
    print(f"\n[2/4] Loading model...")
    model_path = args.resume_from or args.model

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try flash_attention_2, fall back to sdpa (built into PyTorch 2.0+)
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
    # Batch math: per_device(1) * grad_accum(16) * gpus(1) = 16
    # 16 must be divisible by num_generations(16) → 16/16 = 1 prompt per gen batch
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
        max_completion_length=128,

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

    sanity_cb = CalibrationSanityCallback(
        log_dir=str(RESULTS_DIR / "stage1a_sanity"),
        log_every=50,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[calibration_reward],
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
        print("\n[4/4] Evaluating calibration quality...")
        results = evaluate_calibration(model, tokenizer, n_eval=args.n_eval)

        eval_file = Path(output_dir) / "calibration_eval.json"
        with open(eval_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Eval results saved to: {eval_file}")
    else:
        print("\n[4/4] Skipping evaluation (--skip_eval)")

    print("\n" + "=" * 60)
    print("STAGE 1a COMPLETE")
    print("=" * 60)
    print(f"\nNext: Run Stage 1b (tool use training) on the best checkpoint.")
    print(f"  python training/stage1b_tool_use.py --model {output_dir}/checkpoint-XXX")


if __name__ == "__main__":
    main()
