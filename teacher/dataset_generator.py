import json
from pathlib import Path

from .client import TeacherClient

DATAGEN_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "teacher_datagen.txt"


def _load_prompt_template() -> str:
    if DATAGEN_PROMPT_PATH.exists():
        return DATAGEN_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "You are an expert teacher. Given the following student failure analysis, "
        "generate a training sample that would teach the student the correct approach.\n\n"
        "Failure:\n"
        "Input: {input}\n"
        "Student Output: {student_output}\n"
        "Error Type: {error_type}\n"
        "Error Reason: {error_reason}\n"
        "Correct Reasoning: {correct_reasoning}\n"
        "Correct Answer: {corrected_solution}\n\n"
        "Generate a training sample in JSON with keys:\n"
        '- "instruction": the task instruction\n'
        '- "input": the problem input\n'
        '- "reasoning": step-by-step correct reasoning\n'
        '- "output": the correct final answer\n\n'
        "Respond ONLY with valid JSON."
    )


def generate_training_samples(
    client: TeacherClient,
    analyzed_failures: list[dict],
) -> list[dict]:
    """Convert analyzed failures into training samples using the teacher model."""
    template = _load_prompt_template()
    samples = []

    for failure in analyzed_failures:
        prompt = template.format(
            input=failure.get("input", ""),
            student_output=failure.get("student_output", ""),
            error_type=failure.get("error_type", ""),
            error_reason=failure.get("error_reason", ""),
            correct_reasoning=failure.get("correct_reasoning", ""),
            corrected_solution=failure.get("corrected_solution", ""),
        )

        response = client.query(prompt, temperature=0.5)

        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            sample = json.loads(response)
            samples.append(sample)
        except json.JSONDecodeError:
            # Fallback: create a sample directly from the failure data
            samples.append({
                "instruction": "Solve the following problem step by step.",
                "input": failure.get("input", ""),
                "reasoning": failure.get("correct_reasoning", ""),
                "output": failure.get("corrected_solution", ""),
            })

    return samples
