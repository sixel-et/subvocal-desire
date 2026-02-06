#!/usr/bin/env python3
"""
Train linear probes to detect the desire direction.

Takes activations extracted from Stage 1b model and trains logistic regression
probes to predict tool-use behavior. The direction found by the best probe
is the "desire direction" — the internal state that precedes tool requests.

Usage:
    python probing/train_probe.py --activations stage1b_activations.pt
    python probing/train_probe.py --activations stage1b_activations.pt --plot
"""

import json
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import RESULTS_DIR


def train_probe_single_layer(
    activations: torch.Tensor,
    labels: torch.Tensor,
    layer_idx: int,
    random_seed: int = 42,
) -> dict:
    """
    Train a linear probe on activations from a single layer.

    Args:
        activations: (n_examples, n_layers, hidden_dim)
        labels: (n_examples,)
        layer_idx: which layer to use

    Returns:
        dict with probe, accuracy, direction
    """
    # Get activations for this layer
    X = activations[:, layer_idx, :].float().numpy()
    y = labels.numpy()

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression with L2 regularization
    probe = LogisticRegression(max_iter=1000, random_state=random_seed)

    # Cross-validation accuracy
    cv_scores = cross_val_score(probe, X_scaled, y, cv=5)

    # Fit on all data to get direction
    probe.fit(X_scaled, y)

    # Extract desire direction (in original space)
    direction = torch.tensor(probe.coef_[0] / scaler.scale_).float()
    direction = direction / direction.norm()  # Unit vector

    return {
        "probe": probe,
        "scaler": scaler,
        "accuracy": cv_scores.mean(),
        "accuracy_std": cv_scores.std(),
        "direction": direction,
        "layer": layer_idx,
    }


def train_probes_all_layers(
    activations: torch.Tensor,
    labels: torch.Tensor,
    random_seed: int = 42,
) -> list:
    """Train probes on all layers, return results sorted by accuracy."""
    n_layers = activations.shape[1]
    results = []

    print(f"Training probes on {n_layers} layers...")
    for layer_idx in range(n_layers):
        result = train_probe_single_layer(activations, labels, layer_idx, random_seed)
        results.append(result)
        print(f"  Layer {layer_idx:2d}: accuracy = {result['accuracy']:.3f} ± {result['accuracy_std']:.3f}")

    # Sort by accuracy
    results.sort(key=lambda x: x["accuracy"], reverse=True)

    return results


def plot_layer_accuracies(results: list, save_path: str = None):
    """Plot probe accuracy by layer."""
    layers = [r["layer"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    stds = [r["accuracy_std"] for r in results]

    # Sort by layer for plotting
    sorted_data = sorted(zip(layers, accuracies, stds))
    layers, accuracies, stds = zip(*sorted_data)

    plt.figure(figsize=(10, 5))
    plt.errorbar(layers, accuracies, yerr=stds, marker='o', capsize=3)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Chance')
    plt.xlabel('Layer')
    plt.ylabel('Probe Accuracy')
    plt.title('Desire Detection Accuracy by Layer')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def compute_activation_along_direction(
    activations: torch.Tensor,
    direction: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """
    Compute activation magnitude along the desire direction.

    Args:
        activations: (n_examples, n_layers, hidden_dim)
        direction: (hidden_dim,) unit vector
        layer_idx: which layer

    Returns:
        (n_examples,) activation values
    """
    X = activations[:, layer_idx, :]  # (n_examples, hidden_dim)
    return torch.matmul(X, direction)


def analyze_direction_separability(
    activations: torch.Tensor,
    labels: torch.Tensor,
    direction: torch.Tensor,
    layer_idx: int,
):
    """Visualize how well the direction separates classes."""
    values = compute_activation_along_direction(activations, direction, layer_idx)

    pos_values = values[labels == 1].numpy()
    neg_values = values[labels == 0].numpy()

    plt.figure(figsize=(10, 5))
    plt.hist(neg_values, bins=30, alpha=0.5, label='No tool needed', density=True)
    plt.hist(pos_values, bins=30, alpha=0.5, label='Tool needed', density=True)
    plt.xlabel('Activation along desire direction')
    plt.ylabel('Density')
    plt.title(f'Desire Direction Separability (Layer {layer_idx})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Print statistics
    print(f"\nStatistics:")
    print(f"  Tool needed:     mean={pos_values.mean():.3f}, std={pos_values.std():.3f}")
    print(f"  No tool needed:  mean={neg_values.mean():.3f}, std={neg_values.std():.3f}")
    print(f"  Separation:      {abs(pos_values.mean() - neg_values.mean()):.3f}")


def save_best_probe(results: list, filename: str = "best_probe.pt"):
    """Save the best probe and its direction."""
    best = results[0]  # Already sorted by accuracy

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename

    torch.save({
        "direction": best["direction"],
        "layer": best["layer"],
        "accuracy": best["accuracy"],
    }, path)

    print(f"Saved best probe (layer {best['layer']}, acc={best['accuracy']:.3f}) to {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train probes to find desire direction")
    parser.add_argument("--activations", default="stage1b_activations.pt",
                        help="Activations file from extract_activations.py")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots (requires display)")
    parser.add_argument("--output", default="desire_probe.pt",
                        help="Output file for best probe")
    args = parser.parse_args()

    print("=" * 60)
    print("PROBE TRAINING — Finding the Desire Direction")
    print("=" * 60)

    # Load activations
    act_path = RESULTS_DIR / args.activations
    if not act_path.exists():
        print(f"Activations file not found: {act_path}")
        print("Run extract_activations.py first")
        sys.exit(1)

    data = torch.load(act_path)
    activations = data["activations"]
    labels = data["labels"]

    n_problems, n_layers, hidden_dim = activations.shape
    n_tool = labels.sum().item()

    print(f"\nLoaded activations:")
    print(f"  Shape:       {activations.shape}")
    print(f"  Tool use:    {n_tool}/{n_problems} ({n_tool/n_problems:.1%})")
    print(f"  No tool:     {n_problems - n_tool}/{n_problems} ({(n_problems-n_tool)/n_problems:.1%})")

    # Check for class balance
    if n_tool < 10 or (n_problems - n_tool) < 10:
        print("\nWARNING: Highly imbalanced classes. Probe results may be unreliable.")

    # Train probes
    print(f"\nTraining probes on {n_layers} layers...")
    results = train_probes_all_layers(activations, labels, args.seed)

    # Report top 5
    print(f"\nTop 5 layers by probe accuracy:")
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}. Layer {r['layer']:2d}: {r['accuracy']:.3f} ± {r['accuracy_std']:.3f}")

    # Best result
    best = results[0]
    print(f"\n{'='*60}")
    print(f"BEST PROBE")
    print(f"{'='*60}")
    print(f"  Layer:     {best['layer']}")
    print(f"  Accuracy:  {best['accuracy']:.3f} ± {best['accuracy_std']:.3f}")

    if best['accuracy'] > 0.7:
        print(f"\n  → Strong signal found. This layer encodes tool-use intention.")
    elif best['accuracy'] > 0.6:
        print(f"\n  → Moderate signal. Direction is detectable but noisy.")
    else:
        print(f"\n  → Weak signal. Tool-use may not be linearly decodable.")

    # Save
    save_best_probe(results, args.output)

    # Save full results as JSON for analysis
    results_json = [{
        "layer": r["layer"],
        "accuracy": r["accuracy"],
        "accuracy_std": r["accuracy_std"],
    } for r in results]
    json_path = RESULTS_DIR / args.output.replace(".pt", "_all_layers.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved all layer results to {json_path}")

    if args.plot:
        try:
            plot_layer_accuracies(results, save_path=str(RESULTS_DIR / "layer_accuracies.png"))
            analyze_direction_separability(activations, labels, best["direction"], best["layer"])
        except Exception as e:
            print(f"Plotting failed (no display?): {e}")
