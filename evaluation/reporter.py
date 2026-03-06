import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_results(results: list[dict], output_path: str):
    """Save iteration results to a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def print_results_table(results: list[dict]):
    """Print a formatted table of iteration results."""
    print(f"\n{'Stage':<20} {'Accuracy':>10} {'Correct':>10} {'Failed':>10}")
    print("-" * 52)
    for r in results:
        print(
            f"{r['stage']:<20} {r['accuracy']:>9.1%} "
            f"{r['correct']:>10} {r['failed']:>10}"
        )
    print()


def plot_accuracy_curve(results: list[dict], output_path: str):
    """Plot and save an accuracy curve across iterations."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    stages = [r["stage"] for r in results]
    accuracies = [r["accuracy"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(stages, accuracies, "b-o", linewidth=2, markersize=8)
    plt.xlabel("Training Stage")
    plt.ylabel("Accuracy")
    plt.title("Student Model Accuracy Over Iterations")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Accuracy plot saved to {output_path}")
