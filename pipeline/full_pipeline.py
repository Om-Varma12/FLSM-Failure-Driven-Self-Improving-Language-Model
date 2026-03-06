import yaml
from pathlib import Path

from student.loader import load_student_model
from teacher.client import TeacherClient
from skills.registry import get_skill
from evaluation.evaluator import evaluate_student
from evaluation.reporter import print_results_table, plot_accuracy_curve, save_results
from pipeline.run_iteration import run_single_iteration
from pipeline.checkpoint_manager import save_checkpoint, load_checkpoint


def run_full_pipeline(config_path: str = "configs/base_config.yaml"):
    """Run the full failure-driven training pipeline for N iterations."""
    config = _load_config(config_path)

    skill_name = config.get("skill", "math")
    num_iterations = config.get("num_iterations", 3)
    model_name = config.get("student_model", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
    output_base = config.get("output_dir", "outputs")

    print(f"Skill: {skill_name}")
    print(f"Student model: {model_name}")
    print(f"Iterations: {num_iterations}")

    # Load skill
    skill = get_skill(skill_name)

    # Load student model
    print("\nLoading student model...")
    model, tokenizer = load_student_model(model_name)

    # Load teacher client
    teacher_client = TeacherClient()

    # Load datasets
    train_data = _load_dataset(config.get("train_data_path", f"data/processed/{skill_name}/train.jsonl"))
    test_data = _load_dataset(config.get("test_data_path", f"data/processed/{skill_name}/test.jsonl"))

    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    # Baseline evaluation
    print("\n--- Baseline Evaluation ---")
    baseline_accuracy, _, _ = evaluate_student(model, tokenizer, test_data, skill)
    results = [{
        "iteration": 0,
        "stage": "Baseline",
        "accuracy": baseline_accuracy,
        "correct": int(baseline_accuracy * len(test_data)),
        "failed": len(test_data) - int(baseline_accuracy * len(test_data)),
    }]
    print(f"Baseline accuracy: {baseline_accuracy:.1%}")

    # Run iterations
    for i in range(1, num_iterations + 1):
        result = run_single_iteration(
            iteration_num=i,
            skill=skill,
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            test_data=test_data,
            teacher_client=teacher_client,
            output_base=output_base,
        )
        results.append(result)

        # Save checkpoint after each iteration
        save_checkpoint(
            iteration=i,
            results=results,
            output_dir=f"{output_base}/checkpoints/{skill_name}",
        )

    # Final report
    print_results_table(results)
    save_results(results, f"{output_base}/results/{skill_name}_results.json")
    plot_accuracy_curve(results, f"{output_base}/plots/{skill_name}_accuracy.png")

    return results


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        print(f"Config not found at {config_path}, using defaults.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_dataset(data_path: str) -> list[dict]:
    import json
    path = Path(data_path)
    if not path.exists():
        print(f"Warning: Dataset not found at {data_path}")
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data
