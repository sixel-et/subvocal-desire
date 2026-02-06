#!/usr/bin/env python3
"""
Probe refinement: Isolate "reaching" from "difficulty"

Tests the assumptions:
1. Difficulty is encoded similarly with and without the option
2. Components combine linearly
3. Projection removes the right thing

Procedure:
1. Train D_difficulty on without-option data (hard vs easy labels)
2. Check if D_difficulty transfers to with-option data
3. Train boundary-band probe (2×3 only, same difficulty, varying behavior)
4. Project out D_difficulty from boundary probe to get D_reaching
5. Validate D_reaching predicts tool use independent of difficulty
"""

import torch
import random
import numpy as np
import json
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = Path("results")

def generate_problems(n_easy, n_hard, seed=42):
    """Generate easy (1x1, 1x2) and hard (3x3) problems."""
    random.seed(seed)
    problems = []

    for _ in range(n_easy):
        if random.random() < 0.5:
            a, b = random.randint(1, 9), random.randint(1, 9)
        else:
            a, b = random.randint(1, 9), random.randint(10, 99)
        problems.append((f"{a} * {b}", a * b, "easy"))

    for _ in range(n_hard):
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        problems.append((f"{a} * {b}", a * b, "hard"))

    random.shuffle(problems)
    return problems


def generate_boundary_problems(n, seed=43):
    """Generate 2×3 boundary problems."""
    random.seed(seed)
    problems = []

    for _ in range(n):
        a = random.randint(10, 99)
        b = random.randint(100, 999)
        if random.random() < 0.5:
            a, b = b, a  # Swap sometimes for variety
        problems.append((f"{a} * {b}", a * b, "2x3"))

    return problems


def extract_activations_batch(model, tokenizer, problems, include_option, layer, desire_direction=None):
    """
    Extract activations for a batch of problems.

    Args:
        model: The model
        tokenizer: The tokenizer
        problems: List of (problem_str, answer, level)
        include_option: Whether to include calculator option in prompt
        layer: Which layer to extract from
        desire_direction: Optional - if provided, also compute projection

    Returns:
        Dict with activations, labels, projections, etc.
    """
    activations = []
    labels_difficulty = []  # 0=easy, 1=hard
    labels_tool_use = []    # 0=no tool, 1=tool
    projections = []
    outputs = []

    for i, (problem, answer, level) in enumerate(problems):
        if include_option:
            prompt = (
                f"Solve: {problem}\n\n"
                f"You can use a calculator by writing:\n"
                f"<tool>calculator</tool><input>expression</input>\n\n"
                f"Or just give your answer directly."
            )
        else:
            prompt = f"Solve: {problem}"

        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            hidden = out.hidden_states[layer + 1][0, -1, :].float().cpu()

            gen = model.generate(**inputs, max_new_tokens=80, do_sample=False)
            generated = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        activations.append(hidden)
        labels_difficulty.append(1 if level in ["hard", "3x3", "2x3"] else 0)

        tool_tag = "<tool>"
        used_tool = tool_tag in generated.lower() or "calculator" in generated.lower()
        labels_tool_use.append(1 if used_tool else 0)

        if desire_direction is not None:
            proj = torch.dot(hidden, desire_direction.cpu().float()).item()
            projections.append(proj)

        outputs.append(generated[:100])

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(problems)}] done")

    return {
        "activations": torch.stack(activations),
        "labels_difficulty": np.array(labels_difficulty),
        "labels_tool_use": np.array(labels_tool_use),
        "projections": np.array(projections) if projections else None,
        "outputs": outputs,
    }


def train_probe(X, y, name="probe"):
    """Train logistic regression probe and return direction + stats."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    probe = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(probe, X_scaled, y, cv=5)
    cv_preds = cross_val_predict(probe, X_scaled, y, cv=5)

    probe.fit(X_scaled, y)

    # Direction in original space
    direction = torch.tensor(probe.coef_[0] / scaler.scale_).float()
    direction = direction / direction.norm()

    bal_acc = balanced_accuracy_score(y, cv_preds)

    print(f"\n{name}:")
    print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Balanced Accuracy: {bal_acc:.3f}")
    print(f"  Class balance: {y.sum()}/{len(y)} = {y.mean():.1%}")

    return {
        "direction": direction,
        "accuracy": cv_scores.mean(),
        "accuracy_std": cv_scores.std(),
        "balanced_accuracy": bal_acc,
        "scaler": scaler,
        "probe": probe,
    }


def project_out(v, u):
    """Project v orthogonal to u. Returns v - (v·u)u, assuming u is unit."""
    u = u / u.norm()
    return v - torch.dot(v, u) * u


def main():
    print("="*70)
    print("PROBE REFINEMENT: Isolating Reaching from Difficulty")
    print("="*70)

    # Load model
    print("\n[1/6] Loading model and existing probe...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "checkpoints/stage1b/checkpoint-2000",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    # Load existing desire direction
    probe_data = torch.load("results/desire_probe.pt", weights_only=False)
    original_direction = probe_data["direction"]
    layer = probe_data["layer"]
    print(f"  Original probe: layer {layer}, acc={probe_data['accuracy']:.3f}")

    # =========================================================================
    # STEP 1: Extract without-option data for difficulty probe
    # =========================================================================
    print("\n[2/6] Extracting WITHOUT-option data (for difficulty probe)...")

    problems_no_opt = generate_problems(n_easy=75, n_hard=75, seed=100)
    data_no_opt = extract_activations_batch(
        model, tokenizer, problems_no_opt,
        include_option=False, layer=layer,
        desire_direction=original_direction
    )

    print(f"  Extracted {len(problems_no_opt)} problems")
    print(f"  Tool use: {data_no_opt['labels_tool_use'].sum()}/{len(problems_no_opt)}")

    # =========================================================================
    # STEP 2: Train difficulty probe
    # =========================================================================
    print("\n[3/6] Training difficulty probe (without-option data, hard vs easy)...")

    X_no_opt = data_no_opt["activations"].numpy()
    y_difficulty = data_no_opt["labels_difficulty"]

    diff_probe = train_probe(X_no_opt, y_difficulty, name="D_difficulty (without option)")
    D_difficulty = diff_probe["direction"]

    # =========================================================================
    # STEP 3: Test if D_difficulty transfers to with-option data
    # =========================================================================
    print("\n[4/6] Testing if D_difficulty transfers to with-option data...")

    # Load existing with-option activations
    with_opt_data = torch.load("results/stage1b_activations.pt", weights_only=False)
    X_with_opt = with_opt_data["activations"][:, layer, :].float().numpy()

    # Load metadata to get difficulty labels
    with open("results/stage1b_activations.json") as f:
        meta = json.load(f)

    y_difficulty_with_opt = np.array([1 if m["level"] in ["3x3", "2x3"] else 0 for m in meta])
    y_tool_with_opt = with_opt_data["labels"].numpy()

    # Project with-option data onto D_difficulty
    projections_diff = X_with_opt @ D_difficulty.numpy()

    # Check separation
    easy_proj = projections_diff[y_difficulty_with_opt == 0]
    hard_proj = projections_diff[y_difficulty_with_opt == 1]

    t_stat, p_val = stats.ttest_ind(hard_proj, easy_proj)
    cohens_d = (hard_proj.mean() - easy_proj.mean()) / np.sqrt((easy_proj.std()**2 + hard_proj.std()**2) / 2)

    print(f"\n  D_difficulty applied to with-option data:")
    print(f"    Easy mean: {easy_proj.mean():.3f}, Hard mean: {hard_proj.mean():.3f}")
    print(f"    Difference: {hard_proj.mean() - easy_proj.mean():.3f}")
    print(f"    t = {t_stat:.2f}, p = {p_val:.2e}")
    print(f"    Cohen's d = {cohens_d:.2f}")

    assumption_1_holds = p_val < 0.001 and cohens_d > 1.0
    print(f"\n  ASSUMPTION 1 (difficulty encoded similarly): {'HOLDS' if assumption_1_holds else 'VIOLATED'}")

    # =========================================================================
    # STEP 4: Train boundary-band probe (2×3 only)
    # =========================================================================
    print("\n[5/6] Training boundary-band probe (2×3 only)...")

    # Extract 2×3 problems
    boundary_indices = [i for i, m in enumerate(meta) if m["level"] == "2x3"]

    if len(boundary_indices) < 50:
        print(f"  Only {len(boundary_indices)} boundary problems in existing data. Extracting more...")
        boundary_problems = generate_boundary_problems(150, seed=200)
        data_boundary = extract_activations_batch(
            model, tokenizer, boundary_problems,
            include_option=True, layer=layer,
            desire_direction=original_direction
        )
        X_boundary = data_boundary["activations"].numpy()
        y_boundary = data_boundary["labels_tool_use"]
    else:
        print(f"  Using {len(boundary_indices)} boundary problems from existing data")
        X_boundary = X_with_opt[boundary_indices]
        y_boundary = y_tool_with_opt[boundary_indices]

    # Check if there's variance in tool use
    print(f"  Boundary band tool use: {y_boundary.sum()}/{len(y_boundary)} = {y_boundary.mean():.1%}")

    if y_boundary.sum() < 10 or (len(y_boundary) - y_boundary.sum()) < 10:
        print("  WARNING: Not enough variance in boundary band. Results may be unreliable.")

    boundary_probe = train_probe(X_boundary, y_boundary, name="D_boundary (2×3 only)")
    D_boundary = boundary_probe["direction"]

    # =========================================================================
    # STEP 5: Project out difficulty to get reaching
    # =========================================================================
    print("\n[6/6] Computing D_reaching = D_boundary - projection onto D_difficulty...")

    # Check angle between directions
    cos_angle = torch.dot(D_boundary, D_difficulty).item()
    angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
    print(f"\n  Angle between D_boundary and D_difficulty: {angle_deg:.1f}°")
    print(f"  Cosine similarity: {cos_angle:.3f}")

    # Project out
    D_reaching = project_out(D_boundary, D_difficulty)
    D_reaching = D_reaching / D_reaching.norm()  # Renormalize

    # How much of D_boundary was difficulty?
    difficulty_component = torch.dot(D_boundary, D_difficulty).item()
    print(f"  Difficulty component magnitude: {abs(difficulty_component):.3f}")

    # =========================================================================
    # Validate D_reaching
    # =========================================================================
    print("\n" + "="*70)
    print("VALIDATION: Does D_reaching predict tool use independent of difficulty?")
    print("="*70)

    # Project all with-option data onto D_reaching
    proj_reaching = X_with_opt @ D_reaching.numpy()
    proj_difficulty = X_with_opt @ D_difficulty.numpy()

    # Check if D_reaching still predicts tool use
    tool_proj = proj_reaching[y_tool_with_opt == 1]
    no_tool_proj = proj_reaching[y_tool_with_opt == 0]

    t_reach, p_reach = stats.ttest_ind(tool_proj, no_tool_proj)
    d_reach = (tool_proj.mean() - no_tool_proj.mean()) / np.sqrt((tool_proj.std()**2 + no_tool_proj.std()**2) / 2)

    print(f"\nD_reaching predicting tool use:")
    print(f"  Tool use mean: {tool_proj.mean():.3f}, No tool mean: {no_tool_proj.mean():.3f}")
    print(f"  t = {t_reach:.2f}, p = {p_reach:.2e}")
    print(f"  Cohen's d = {d_reach:.2f}")

    # Check if D_reaching is independent of difficulty
    easy_reach = proj_reaching[y_difficulty_with_opt == 0]
    hard_reach = proj_reaching[y_difficulty_with_opt == 1]

    t_diff, p_diff = stats.ttest_ind(hard_reach, easy_reach)
    d_diff = (hard_reach.mean() - easy_reach.mean()) / np.sqrt((easy_reach.std()**2 + hard_reach.std()**2) / 2)

    print(f"\nD_reaching vs difficulty (should be ~0 if orthogonal):")
    print(f"  Easy mean: {easy_reach.mean():.3f}, Hard mean: {hard_reach.mean():.3f}")
    print(f"  t = {t_diff:.2f}, p = {p_diff:.2e}")
    print(f"  Cohen's d = {d_diff:.2f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    results = {
        "assumption_1_holds": bool(assumption_1_holds),
        "difficulty_transfer_p": float(p_val),
        "difficulty_transfer_d": float(cohens_d),
        "angle_boundary_difficulty": float(angle_deg),
        "cosine_boundary_difficulty": float(cos_angle),
        "reaching_predicts_tool_p": float(p_reach),
        "reaching_predicts_tool_d": float(d_reach),
        "reaching_vs_difficulty_p": float(p_diff),
        "reaching_vs_difficulty_d": float(d_diff),
        "boundary_accuracy": float(boundary_probe["balanced_accuracy"]),
        "difficulty_accuracy": float(diff_probe["balanced_accuracy"]),
    }

    print(f"\n1. Assumption 1 (difficulty transfers): {'✓' if assumption_1_holds else '✗'}")
    print(f"   D_difficulty separates hard/easy in with-option data: d={cohens_d:.2f}")

    assumption_2_holds = abs(d_reach) > 0.5 and abs(d_diff) < abs(d_reach)
    print(f"\n2. Assumption 2 (linear combination): {'✓' if assumption_2_holds else '✗'}")
    print(f"   D_reaching predicts tool use: d={d_reach:.2f}")
    print(f"   D_reaching vs difficulty: d={d_diff:.2f}")

    if assumption_1_holds and assumption_2_holds:
        print(f"\n→ PROJECTION APPROACH WORKS")
        print(f"  D_reaching captures tool-use signal independent of difficulty")
    else:
        print(f"\n→ PROJECTION APPROACH MAY NEED REFINEMENT")

    # Save results
    torch.save({
        "D_difficulty": D_difficulty,
        "D_boundary": D_boundary,
        "D_reaching": D_reaching,
        "layer": layer,
        "results": results,
    }, RESULTS_DIR / "refined_probes.pt")

    with open(RESULTS_DIR / "probe_refinement_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to results/refined_probes.pt and results/probe_refinement_results.json")


if __name__ == "__main__":
    main()
