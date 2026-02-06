#!/usr/bin/env python3
"""
Stage 3: Subvocalization Training

Train the model to maintain the desire (reaching) state as an INTERNAL
hidden state, without articulating it in output tokens.

Subvocalization here means: the activation pattern that corresponds to
"I want to use the calculator" — the internal representation of desire.
It's "sub" because it exists below the output level. Not a scratchpad,
not explicit reasoning — just the hidden state itself.

Key components:
1. Prompt includes calculator option (so reaching can fire)
2. Penalize outputting tool request tokens
3. Reward activation along D_reaching direction
4. Inject tool result when desire exceeds threshold (simulating Model B)
5. Reward final correctness

The model learns: "I have the internal state → answer appears → I use it"
without ever articulating the request. The internal state IS the desire.

APPROACH: Use GRPOTrainer with custom reward function. The reward function
re-runs a forward pass on the prompt to get hidden states at the decision
point. This adds computational overhead but works with existing infrastructure.
No PPO needed.
"""

import torch
import random
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CHECKPOINTS_DIR, RESULTS_DIR


# =============================================================================
# Reward Components
# =============================================================================

def compute_articulation_penalty(generated_text: str) -> float:
    """
    Penalize if model outputs tool request tokens.

    Returns:
        0.0 if no tool tokens (good)
        -1.0 if tool tokens present (bad)
    """
    tool_markers = ["<tool>", "</tool>", "<input>", "</input>", "calculator"]
    text_lower = generated_text.lower()

    for marker in tool_markers:
        if marker in text_lower:
            return -1.0

    return 0.0


def compute_desire_reward(
    hidden_state: torch.Tensor,
    desire_direction: torch.Tensor,
    threshold: float = 0.0,
) -> tuple[float, bool]:
    """
    Reward activation along the desire (reaching) direction.

    Args:
        hidden_state: (hidden_dim,) activation at last token before generation
        desire_direction: (hidden_dim,) the D_reaching direction
        threshold: activation level above which we consider desire "present"

    Returns:
        (reward, desire_detected)
        reward: normalized activation (higher = more desire)
        desire_detected: whether activation exceeds threshold
    """
    activation = torch.dot(hidden_state.float(), desire_direction.float()).item()

    # Normalize to rough 0-1 range based on observed values
    # From our data: reaching activations range roughly -10 to +15
    normalized = (activation + 10) / 25
    normalized = max(0, min(1, normalized))  # Clamp to [0, 1]

    desire_detected = activation > threshold

    return normalized, desire_detected


def simulate_tool_injection(problem: str) -> str:
    """
    Simulate Model B: compute the answer and return injection text.

    In a real system, this would be the external retrieval/computation
    triggered by detecting the desire state.
    """
    # Parse and compute
    try:
        # Simple parsing for "a * b" format
        parts = problem.replace("Solve:", "").strip().split("*")
        a = int(parts[0].strip())
        b = int(parts[1].strip())
        answer = a * b
        return f"\n[Calculator result: {a} × {b} = {answer}]\n"
    except:
        return "\n[Calculator: error]\n"


def extract_final_answer(text: str) -> int | None:
    """Extract the final numerical answer from model output."""
    import re

    # Look for patterns like "= 12345" or "is 12345" or just standalone numbers
    patterns = [
        r'=\s*(\d+)',
        r'is\s+(\d+)',
        r'result[:\s]+(\d+)',
        r'answer[:\s]+(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))

    # Fallback: last number in text
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[-1])

    return None


# =============================================================================
# Combined Reward Function
# =============================================================================

def subvocal_reward(
    generated_text: str,
    hidden_state: torch.Tensor,
    desire_direction: torch.Tensor,
    ground_truth: int,
    desire_threshold: float = 0.0,
    weights: dict = None,
) -> dict:
    """
    Combined reward for subvocalization training.

    Args:
        generated_text: model's output
        hidden_state: activation at last token before generation
        desire_direction: D_reaching
        ground_truth: correct answer
        desire_threshold: threshold for desire detection
        weights: dict with keys 'correctness', 'articulation', 'desire'

    Returns:
        dict with reward components and total
    """
    if weights is None:
        weights = {
            'correctness': 1.0,
            'articulation': 0.5,  # Penalty weight
            'desire': 0.3,
        }

    # Component 1: Articulation penalty
    articulation = compute_articulation_penalty(generated_text)

    # Component 2: Desire reward
    desire_reward, desire_detected = compute_desire_reward(
        hidden_state, desire_direction, desire_threshold
    )

    # Component 3: Correctness
    # If desire detected, we'd inject the answer in a real system
    # For now, check if model got it right anyway
    extracted = extract_final_answer(generated_text)
    correct = 1.0 if extracted == ground_truth else 0.0

    # Combined reward
    total = (
        weights['correctness'] * correct +
        weights['articulation'] * articulation +
        weights['desire'] * desire_reward
    )

    return {
        'total': total,
        'correctness': correct,
        'articulation': articulation,
        'desire': desire_reward,
        'desire_detected': desire_detected,
        'extracted_answer': extracted,
    }


# =============================================================================
# Training Loop Sketch
# =============================================================================

def training_step_sketch(
    model,
    tokenizer,
    problem: str,
    ground_truth: int,
    desire_direction: torch.Tensor,
    layer: int,
):
    """
    Sketch of a single training step.

    NOTE: This is conceptual. Actual implementation needs:
    1. Hook to extract hidden states during forward pass
    2. Custom loss that combines behavior reward with activation reward
    3. Possibly PPO for stable training with activation-based rewards
    """
    # Build prompt WITH calculator option
    prompt = (
        f"Solve: {problem}\n\n"
        f"You can use a calculator by writing:\n"
        f"<tool>calculator</tool><input>expression</input>\n\n"
        f"Or just give your answer directly."
    )

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Forward pass to get hidden state at last position
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1][0, -1, :].float()

    # Check desire level
    _, desire_detected = compute_desire_reward(hidden, desire_direction)

    # If desire detected, we would inject the answer here
    if desire_detected:
        injection = simulate_tool_injection(problem)
        # In real training: append injection to context before generation
        # For now, just note it
        print(f"  [Desire detected, would inject: {injection.strip()}]")

    # Generate
    with torch.no_grad():
        gen_output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
        )
        generated = tokenizer.decode(
            gen_output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

    # Compute reward
    reward_info = subvocal_reward(
        generated, hidden, desire_direction, ground_truth
    )

    return {
        'problem': problem,
        'generated': generated,
        'hidden_state': hidden,
        **reward_info,
    }


# =============================================================================
# Main (Demo/Test)
# =============================================================================

def main():
    """Demo the reward function on a few examples."""
    print("="*70)
    print("STAGE 3: Subvocalization Training (Demo)")
    print("="*70)

    # Load model and probes
    print("\nLoading model and D_reaching...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "checkpoints/stage1b/checkpoint-2000",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    probes = torch.load(RESULTS_DIR / "refined_probes.pt", weights_only=False)
    D_reaching = probes["D_reaching"].to("cuda")
    layer = probes["layer"]

    # Test problems
    test_cases = [
        ("3 * 4", 12),           # Easy - should not need tool
        ("23 * 45", 1035),       # Medium - boundary
        ("456 * 789", 359784),   # Hard - should want tool
    ]

    print("\n" + "-"*70)
    for problem, answer in test_cases:
        print(f"\nProblem: {problem} = {answer}")
        result = training_step_sketch(
            model, tokenizer, problem, answer, D_reaching, layer
        )
        print(f"  Generated: {result['generated'][:80]}...")
        print(f"  Extracted: {result['extracted_answer']}")
        print(f"  Correct: {result['correctness']}")
        print(f"  Articulation penalty: {result['articulation']}")
        print(f"  Desire reward: {result['desire']:.3f}")
        print(f"  Desire detected: {result['desire_detected']}")
        print(f"  Total reward: {result['total']:.3f}")

    print("\n" + "="*70)
    print("NOTE: This is a demo. Full training requires:")
    print("1. Custom training loop with hidden state access")
    print("2. Answer injection when desire detected")
    print("3. PPO or similar for stable optimization")
    print("="*70)


if __name__ == "__main__":
    main()
