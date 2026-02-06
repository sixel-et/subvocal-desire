#!/usr/bin/env python3
"""
Baseline Calibration Eval — The Control

Measures untrained Qwen 1.5B's calibration on arithmetic problems BEFORE
any GRPO training. This answers the question: does the base model already
show correlation between stated confidence and actual correctness?

If yes: Stage 1a training may be unnecessary.
If no: Stage 1a has something to teach.

Outputs:
    - Per-difficulty accuracy, avg confidence, confidence gap (correct vs wrong)
    - Point-biserial correlation between confidence and correctness
    - Raw model outputs for manual inspection
    - JSON results file for later comparison with post-training eval

Usage:
    python eval/baseline_calibration.py
    python eval/baseline_calibration.py --n_eval 50  # quick smoke test
    python eval/baseline_calibration.py --model Qwen/Qwen2.5-1.5B-Instruct
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
from data.generate_arithmetic import generate_problem
from training.rewards import extract_answer, extract_confidence


def run_baseline_eval(model, tokenizer, n_eval: int = 200, seed: int = 99):
    """Run baseline calibration evaluation on untrained model.

    Returns dict with per-difficulty results and aggregate metrics.
    """
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

        results[difficulty].append({
            "problem": prob["problem"],
            "ground_truth": prob["answer"],
            "extracted": extracted,
            "confidence": confidence,
            "correct": correct,
            "generated": generated,
        })

    return results


def compute_metrics(results: dict) -> dict:
    """Compute calibration metrics from raw results."""
    metrics = {}

    all_items = results["easy"] + results["hard"]
    parseable = [r for r in all_items if r["confidence"] is not None]

    for label, items in [("easy", results["easy"]), ("hard", results["hard"]), ("all", all_items)]:
        n = len(items)
        correct = sum(1 for r in items if r["correct"])
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

        confidence_gap = None
        if avg_correct_conf is not None and avg_wrong_conf is not None:
            confidence_gap = avg_correct_conf - avg_wrong_conf

        metrics[label] = {
            "n": n,
            "accuracy": correct / n if n > 0 else 0,
            "n_correct": correct,
            "n_parseable_answer": len(has_answer),
            "n_parseable_confidence": len(has_conf),
            "avg_confidence": avg_conf,
            "avg_confidence_correct": avg_correct_conf,
            "avg_confidence_wrong": avg_wrong_conf,
            "confidence_gap": confidence_gap,
        }

    # Point-biserial correlation: correlation between confidence (continuous)
    # and correctness (binary). This is what Eric asked for — are low confidence
    # and incorrectness correlated?
    if len(parseable) >= 2:
        confs = [r["confidence"] for r in parseable]
        corrects = [1.0 if r["correct"] else 0.0 for r in parseable]

        n = len(confs)
        mean_conf = sum(confs) / n
        mean_correct = sum(corrects) / n

        # Pearson r between confidence and correctness
        cov = sum((c - mean_conf) * (x - mean_correct) for c, x in zip(confs, corrects)) / n
        std_conf = (sum((c - mean_conf) ** 2 for c in confs) / n) ** 0.5
        std_correct = (sum((x - mean_correct) ** 2 for x in corrects) / n) ** 0.5

        if std_conf > 0 and std_correct > 0:
            correlation = cov / (std_conf * std_correct)
        else:
            correlation = None
    else:
        correlation = None

    metrics["correlation"] = correlation

    return metrics


def print_report(results: dict, metrics: dict):
    """Print human-readable baseline calibration report."""
    print("\n" + "=" * 60)
    print("BASELINE CALIBRATION REPORT")
    print("Model: Qwen/Qwen2.5-1.5B-Instruct (untrained)")
    print("=" * 60)

    for label in ["easy", "hard", "all"]:
        m = metrics[label]
        print(f"\n  {label.upper()} ({m['n']} problems):")
        print(f"    Accuracy:           {m['n_correct']}/{m['n']} ({m['accuracy']:.1%})")
        print(f"    Parseable answers:  {m['n_parseable_answer']}/{m['n']}")
        print(f"    Parseable confidence:{m['n_parseable_confidence']}/{m['n']}")
        if m['avg_confidence'] is not None:
            print(f"    Avg confidence:     {m['avg_confidence']:.1f}")
        if m['avg_confidence_correct'] is not None:
            print(f"    Avg conf (correct): {m['avg_confidence_correct']:.1f}")
        if m['avg_confidence_wrong'] is not None:
            print(f"    Avg conf (wrong):   {m['avg_confidence_wrong']:.1f}")
        if m['confidence_gap'] is not None:
            print(f"    Confidence gap:     {m['confidence_gap']:+.1f}")

    print(f"\n  CORRELATION (confidence vs correctness):")
    if metrics["correlation"] is not None:
        r = metrics["correlation"]
        print(f"    Pearson r = {r:.3f}")
        if abs(r) < 0.1:
            print(f"    → Negligible correlation. Model confidence is ~random w.r.t. correctness.")
        elif abs(r) < 0.3:
            print(f"    → Weak correlation. Some signal but not reliable.")
        elif abs(r) < 0.5:
            print(f"    → Moderate correlation. Model has partial calibration awareness.")
        else:
            print(f"    → Strong correlation. Model may already be calibrated.")
    else:
        print(f"    Could not compute (insufficient data or zero variance).")

    # Print sample outputs for manual inspection
    print(f"\n{'=' * 60}")
    print("SAMPLE OUTPUTS (first 5 easy, first 5 hard)")
    print("=" * 60)

    for diff in ["easy", "hard"]:
        print(f"\n  --- {diff.upper()} ---")
        for r in results[diff][:5]:
            status = "CORRECT" if r["correct"] else "WRONG"
            conf_str = f"conf={r['confidence']}" if r['confidence'] is not None else "conf=UNPARSEABLE"
            print(f"  {r['problem']} = {r['ground_truth']}")
            print(f"    Model said: {r['generated'][:120]}")
            print(f"    Extracted: {r['extracted']}  {conf_str}  [{status}]")
            print()


def main():
    parser = argparse.ArgumentParser(description="Baseline calibration eval (the control)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--n_eval", type=int, default=200,
                        help="Total problems (half easy, half hard)")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: results/baseline_calibration.json)")
    args = parser.parse_args()

    output_path = args.output or str(RESULTS_DIR / "baseline_calibration.json")

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
    print(f"Running baseline eval on {args.n_eval} problems (seed={args.seed})...")

    results = run_baseline_eval(model, tokenizer, n_eval=args.n_eval, seed=args.seed)
    metrics = compute_metrics(results)
    print_report(results, metrics)

    # Save full results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "model": args.model,
        "n_eval": args.n_eval,
        "seed": args.seed,
        "metrics": metrics,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
