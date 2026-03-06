import json
from pathlib import Path

from evaluation.evaluator import evaluate_student
from teacher.client import TeacherClient
from teacher.analyzer import analyze_failures_batch
from teacher.dataset_generator import generate_training_samples
from training.trainer import fine_tune
from training.curriculum import order_by_curriculum
from skills.base_skill import BaseSkill
from student.lora_manager import DEFAULT_LORA_CONFIG


def run_single_iteration(
    iteration_num: int,
    skill: BaseSkill,
    model,
    tokenizer,
    train_data: list[dict],
    test_data: list[dict],
    teacher_client: TeacherClient,
    output_base: str = "outputs",
    lora_config=None,
) -> dict:
    """Run a single training iteration of the failure-driven loop.

    Steps:
        1. Evaluate student on training data to find failures
        2. Send failures to teacher for analysis
        3. Generate training samples from analyzed failures
        4. Fine-tune student on failure-derived data
        5. Evaluate student on test data to measure improvement

    Returns:
        dict with iteration results (accuracy, num_failures, etc.)
    """
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration_num}")
    print(f"{'='*60}")

    if lora_config is None:
        lora_config = DEFAULT_LORA_CONFIG

    skill_name = skill.name
    failure_bank_dir = Path(f"data/failure_bank/{skill_name}")
    failure_bank_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = f"{output_base}/checkpoints/{skill_name}/iter_{iteration_num}"

    # Step 1: Student attempts training problems
    print(f"\n[Step 1] Student attempting {len(train_data)} training problems...")
    train_accuracy, _, train_failures = evaluate_student(
        model, tokenizer, train_data, skill
    )
    print(f"  Training accuracy: {train_accuracy:.1%} ({len(train_failures)} failures)")

    if not train_failures:
        print("  No failures found! Student has mastered the training set.")
        test_accuracy, _, _ = evaluate_student(model, tokenizer, test_data, skill)
        return {
            "iteration": iteration_num,
            "stage": f"Iteration {iteration_num}",
            "train_accuracy": train_accuracy,
            "accuracy": test_accuracy,
            "correct": int(test_accuracy * len(test_data)),
            "failed": len(test_data) - int(test_accuracy * len(test_data)),
            "num_train_failures": 0,
        }

    # Step 2: Teacher analyzes failures
    print(f"\n[Step 2] Teacher analyzing {len(train_failures)} failures...")
    analyzed_failures = analyze_failures_batch(teacher_client, train_failures)

    # Save failures to failure bank
    failure_path = failure_bank_dir / f"iter_{iteration_num}.jsonl"
    with open(failure_path, "w", encoding="utf-8") as f:
        for failure in analyzed_failures:
            f.write(json.dumps(failure, ensure_ascii=False) + "\n")
    print(f"  Failures saved to {failure_path}")

    # Step 3: Generate training samples
    print(f"\n[Step 3] Generating training samples from failures...")
    training_samples = generate_training_samples(teacher_client, analyzed_failures)
    print(f"  Generated {len(training_samples)} training samples")

    # Step 4: Order by curriculum (easy → hard) and fine-tune
    print(f"\n[Step 4] Fine-tuning student model...")
    ordered_samples = order_by_curriculum(training_samples)
    fine_tune(
        model, tokenizer, ordered_samples, checkpoint_dir, lora_config=lora_config
    )
    print(f"  Checkpoint saved to {checkpoint_dir}")

    # Step 5: Evaluate on test set
    print(f"\n[Step 5] Evaluating on test set ({len(test_data)} samples)...")
    test_accuracy, test_successes, test_failures = evaluate_student(
        model, tokenizer, test_data, skill
    )
    print(f"  Test accuracy: {test_accuracy:.1%}")

    result = {
        "iteration": iteration_num,
        "stage": f"Iteration {iteration_num}",
        "train_accuracy": train_accuracy,
        "accuracy": test_accuracy,
        "correct": len(test_successes),
        "failed": len(test_failures),
        "num_train_failures": len(train_failures),
        "num_training_samples": len(training_samples),
    }

    return result
