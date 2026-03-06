"""Run baseline evaluation of the student model before any training."""
import json
from pathlib import Path

from student.loader import load_student_model
from skills.registry import get_skill
from evaluation.evaluator import evaluate_student
from evaluation.reporter import print_results_table


def run_baseline(
    skill_name: str = "math",
    test_data_path: str = "data/processed/math/test.jsonl",
):
    """Evaluate the student model on the test set without any training."""
    skill = get_skill(skill_name)

    print("Loading student model...")
    model, tokenizer = load_student_model()

    print(f"Loading test data from {test_data_path}...")
    test_data = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))

    print(f"Evaluating on {len(test_data)} test samples...")
    accuracy, successes, failures = evaluate_student(model, tokenizer, test_data, skill)

    results = [{
        "stage": "Baseline",
        "accuracy": accuracy,
        "correct": len(successes),
        "failed": len(failures),
    }]

    print_results_table(results)

    # Save failures for inspection
    output_path = Path("outputs/results/baseline_failures.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(failures[:20], f, indent=2)  # Save first 20 failures
    print(f"Sample failures saved to {output_path}")


if __name__ == "__main__":
    run_baseline()
