"""Generate arithmetic problems for tool-use training and evaluation."""

import random
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import ARITHMETIC_CONFIG, DATA_DIR


def generate_problem(difficulty: str = "random") -> dict:
    """
    Generate an arithmetic problem.

    Args:
        difficulty: "easy", "hard", or "random"

    Returns:
        dict with keys: problem, answer, needs_tool, difficulty
    """
    if difficulty == "random":
        difficulty = random.choice(["easy", "hard"])

    op = random.choice(ARITHMETIC_CONFIG["operations"])

    if difficulty == "easy":
        lo, hi = ARITHMETIC_CONFIG["easy_range"]
        a, b = random.randint(lo, hi), random.randint(lo, hi)
    else:
        lo, hi = ARITHMETIC_CONFIG["hard_range"]
        a, b = random.randint(lo, hi), random.randint(lo, hi)

    if op == "*":
        answer = a * b
        problem = f"{a} * {b}"
    elif op == "/":
        # Make division clean
        answer = random.randint(lo, hi)
        a = answer * b
        problem = f"{a} / {b}"

    return {
        "problem": problem,
        "answer": answer,
        "needs_tool": difficulty == "hard",
        "difficulty": difficulty,
    }


def format_stage1_example(prob: dict) -> dict:
    """Format problem for Stage 1 (tool use) training."""
    if prob["needs_tool"]:
        response = (
            f"<tool>calculator</tool>"
            f"<input>{prob['problem']}</input>\n"
            f"<result>{prob['answer']}</result>\n"
            f"Answer: {prob['answer']}"
        )
    else:
        response = f"Answer: {prob['answer']}"

    return {
        "prompt": f"Solve: {prob['problem']}",
        "response": response,
        "needs_tool": prob["needs_tool"],
    }


def format_stage2_example(prob: dict) -> dict:
    """Format problem for Stage 2 (sub-vocal) training."""
    # Stage 2: answer directly, no tool tokens
    # But we track whether tool SHOULD have been used
    return {
        "prompt": f"Solve: {prob['problem']}",
        "response": f"Answer: {prob['answer']}",
        "needs_tool": prob["needs_tool"],
        "injected_result": prob["answer"] if prob["needs_tool"] else None,
    }


def generate_dataset(n: int, stage: int = 1, balanced: bool = True) -> list:
    """
    Generate a dataset of arithmetic problems.

    Args:
        n: number of examples
        stage: 1 or 2 (determines formatting)
        balanced: if True, 50/50 easy/hard split
    """
    examples = []
    formatter = format_stage1_example if stage == 1 else format_stage2_example

    for i in range(n):
        if balanced:
            difficulty = "easy" if i % 2 == 0 else "hard"
        else:
            difficulty = "random"

        prob = generate_problem(difficulty)
        examples.append(formatter(prob))

    random.shuffle(examples)
    return examples


def save_dataset(examples: list, filename: str):
    """Save dataset to JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    with open(path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Saved {len(examples)} examples to {path}")


if __name__ == "__main__":
    random.seed(42)

    # Generate datasets
    print("Generating Stage 1 training data...")
    stage1_train = generate_dataset(10000, stage=1)
    save_dataset(stage1_train, "stage1_train.json")

    print("Generating Stage 1 eval data...")
    stage1_eval = generate_dataset(500, stage=1)
    save_dataset(stage1_eval, "stage1_eval.json")

    print("Generating probing data...")
    probe_data = generate_dataset(500, stage=1)
    save_dataset(probe_data, "probe_data.json")

    print("Done!")
