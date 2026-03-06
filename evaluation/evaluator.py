from transformers import PreTrainedModel, PreTrainedTokenizer

from skills.base_skill import BaseSkill
from student.inference import run_inference


def evaluate_student(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_data: list[dict],
    skill: BaseSkill,
    max_new_tokens: int = 500,
    temperature: float = 0.1,
) -> tuple[float, list[dict], list[dict]]:
    """Evaluate the student model on a test dataset.

    Returns:
        (accuracy, successes, failures)
    """
    prompts = [skill.format_prompt(sample["input"]) for sample in test_data]

    predictions = run_inference(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    successes = []
    failures = []

    for sample, prediction in zip(test_data, predictions):
        gold = sample["expected_output"]
        is_correct = skill.evaluate(prediction, gold)

        result = {
            "input": sample["input"],
            "student_output": prediction,
            "gold_output": gold,
            "correct": is_correct,
        }

        if is_correct:
            successes.append(result)
        else:
            failures.append(result)

    total = len(test_data)
    accuracy = len(successes) / total if total > 0 else 0.0

    return accuracy, successes, failures
