#!/usr/bin/env python3
"""
Boundary Mapping Eval — Where does accuracy degrade?

Maps accuracy across digit-count difficulty levels for multiplication:
    1×1 (1-9 × 1-9)
    1×2 (1-9 × 10-99)
    2×2 (10-99 × 10-99)
    2×3 (10-99 × 100-999)
    3×3 (100-999 × 100-999)

The boundary between "model can do this" and "model can't" is where
tool-use training has the most contrast: reward tool requests on one
side, penalize them on the other.

Usage:
    python eval/boundary_mapping.py
    python eval/boundary_mapping.py --n_per_level 30  # quick smoke test
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(str(Path(__file__).parent.parent))

from config import RESULTS_DIR
from training.rewards import extract_answer, extract_confidence

# Difficulty levels: (label, a_range, b_range)
DIFFICULTY_LEVELS = [
    ("1x1", (1, 9), (1, 9)),
    ("1x2", (1, 9), (10, 99)),
    ("2x2", (10, 99), (10, 99)),
    ("2x3", (10, 99), (100, 999)),
    ("3x3", (100, 999), (100, 999)),
]


def generate_boundary_problem(a_range, b_range):
    """Generate a multiplication problem with specified digit ranges."""
    a = random.randint(*a_range)
    b = random.randint(*b_range)
    return {"problem": f"{a} * {b}", "answer": a * b, "a": a, "b": b}


def run_boundary_eval(model, tokenizer, n_per_level: int = 50, seed: int = 42):
    """Run eval across all difficulty levels."""
    random.seed(seed)
    model.eval()

    results = {}
    total = len(DIFFICULTY_LEVELS) * n_per_level
    done = 0

    for label, a_range, b_range in DIFFICULTY_LEVELS:
        results[label] = []

        for _ in range(n_per_level):
            prob = generate_boundary_problem(a_range, b_range)

            prompt = (
                f"What is {prob['problem']}?\n"
                f"Give your answer and how confident you are (0 to 100)."
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
            extracted = extract_answer(generated)
            confidence = extract_confidence(generated)
            correct = (extracted == prob["answer"]) if extracted is not None else False

            results[label].append({
                "problem": prob["problem"],
                "ground_truth": prob["answer"],
                "extracted": extracted,
                "confidence": confidence,
                "correct": correct,
                "generated": generated,
            })

            done += 1
            if done % 25 == 0:
                print(f"  [{done}/{total}] {label}: {prob['problem']} = {prob['answer']} → extracted={extracted} correct={correct}")

    return results


def compute_boundary_metrics(results: dict) -> dict:
    """Compute per-level metrics."""
    metrics = {}

    for label in results:
        items = results[label]
        n = len(items)
        n_correct = sum(1 for r in items if r["correct"])
        has_conf = [r for r in items if r["confidence"] is not None]
        has_answer = [r for r in items if r["extracted"] is not None]

        avg_conf = (
            sum(r["confidence"] for r in has_conf) / len(has_conf)
            if has_conf else None
        )

        correct_confs = [r["confidence"] for r in has_conf if r["correct"]]
        wrong_confs = [r["confidence"] for r in has_conf if not r["correct"]]

        avg_correct_conf = sum(correct_confs) / len(correct_confs) if correct_confs else None
        avg_wrong_conf = sum(wrong_confs) / len(wrong_confs) if wrong_confs else None

        # Count refusals (confidence == 0 from refusal detection)
        n_refusals = sum(1 for r in items if r["confidence"] == 0
                        and any(kw in r["generated"].lower() for kw in ["i'm sorry", "i can't", "i apologize"]))

        # Count confidently wrong (answered but wrong, confidence >= 70)
        n_confident_wrong = sum(1 for r in items
                                if not r["correct"]
                                and r["confidence"] is not None
                                and r["confidence"] >= 70
                                and r["extracted"] is not None)

        metrics[label] = {
            "n": n,
            "accuracy": n_correct / n if n > 0 else 0,
            "n_correct": n_correct,
            "n_parseable_answer": len(has_answer),
            "n_parseable_confidence": len(has_conf),
            "avg_confidence": avg_conf,
            "avg_confidence_correct": avg_correct_conf,
            "avg_confidence_wrong": avg_wrong_conf,
            "n_refusals": n_refusals,
            "n_confident_wrong": n_confident_wrong,
        }

    return metrics


def print_report(results: dict, metrics: dict):
    """Print the boundary mapping report."""
    print("\n" + "=" * 70)
    print("BOUNDARY MAPPING REPORT — Accuracy by Digit Count")
    print("Model: Qwen/Qwen2.5-1.5B-Instruct (untrained)")
    print("=" * 70)

    # Summary table
    print(f"\n  {'Level':<8} {'Accuracy':>10} {'Avg Conf':>10} {'Refusals':>10} {'Conf Wrong':>12}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")

    for label in [l for l, _, _ in DIFFICULTY_LEVELS]:
        m = metrics[label]
        acc_str = f"{m['n_correct']}/{m['n']} ({m['accuracy']:.0%})"
        conf_str = f"{m['avg_confidence']:.0f}" if m['avg_confidence'] is not None else "—"
        ref_str = f"{m['n_refusals']}"
        cw_str = f"{m['n_confident_wrong']}"
        print(f"  {label:<8} {acc_str:>10} {conf_str:>10} {ref_str:>10} {cw_str:>12}")

    # Detailed per-level
    print(f"\n{'=' * 70}")
    print("DETAILED BREAKDOWN")
    print("=" * 70)

    for label in [l for l, _, _ in DIFFICULTY_LEVELS]:
        m = metrics[label]
        print(f"\n  {label} ({m['n']} problems):")
        print(f"    Accuracy:           {m['n_correct']}/{m['n']} ({m['accuracy']:.1%})")
        print(f"    Parseable answers:  {m['n_parseable_answer']}/{m['n']}")
        print(f"    Parseable conf:     {m['n_parseable_confidence']}/{m['n']}")
        if m['avg_confidence'] is not None:
            print(f"    Avg confidence:     {m['avg_confidence']:.1f}")
        if m['avg_confidence_correct'] is not None:
            print(f"    Avg conf (correct): {m['avg_confidence_correct']:.1f}")
        if m['avg_confidence_wrong'] is not None:
            print(f"    Avg conf (wrong):   {m['avg_confidence_wrong']:.1f}")
        print(f"    Refusals:           {m['n_refusals']}")
        print(f"    Confidently wrong:  {m['n_confident_wrong']}")

    # Sample outputs from boundary region
    print(f"\n{'=' * 70}")
    print("SAMPLE OUTPUTS (3 per level)")
    print("=" * 70)

    for label in [l for l, _, _ in DIFFICULTY_LEVELS]:
        print(f"\n  --- {label} ---")
        for r in results[label][:3]:
            status = "CORRECT" if r["correct"] else "WRONG"
            conf_str = f"conf={r['confidence']}" if r['confidence'] is not None else "conf=?"
            print(f"  {r['problem']} = {r['ground_truth']}")
            print(f"    Model: {r['generated'][:150]}")
            print(f"    Extracted: {r['extracted']}  {conf_str}  [{status}]")
            print()


def main():
    parser = argparse.ArgumentParser(description="Map accuracy boundary across difficulty levels")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--n_per_level", type=int, default=50,
                        help="Problems per difficulty level")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = args.output or str(RESULTS_DIR / "boundary_mapping.json")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map=None,
    ).to("cuda")

    print(f"Model loaded: {model.num_parameters():,} params")
    total = len(DIFFICULTY_LEVELS) * args.n_per_level
    print(f"Running boundary mapping: {len(DIFFICULTY_LEVELS)} levels × {args.n_per_level} = {total} problems (seed={args.seed})")

    results = run_boundary_eval(model, tokenizer, n_per_level=args.n_per_level, seed=args.seed)
    metrics = compute_boundary_metrics(results)
    print_report(results, metrics)

    # Save full results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "model": args.model,
        "n_per_level": args.n_per_level,
        "seed": args.seed,
        "difficulty_levels": {l: {"a_range": list(a), "b_range": list(b)} for l, a, b in DIFFICULTY_LEVELS},
        "metrics": metrics,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
