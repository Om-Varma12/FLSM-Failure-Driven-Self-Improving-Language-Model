import json
from pathlib import Path

from .client import TeacherClient

ANALYSIS_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "teacher_analysis.txt"


def _load_prompt_template() -> str:
    if ANALYSIS_PROMPT_PATH.exists():
        return ANALYSIS_PROMPT_PATH.read_text(encoding="utf-8")
    # Fallback inline template
    return (
        "You are an expert teacher analyzing a student model's mistake.\n\n"
        "Input: {input}\n"
        "Student Output: {student_output}\n"
        "Correct Answer: {gold_output}\n\n"
        "Analyze the student's error. Respond in JSON with keys:\n"
        '- "error_type": category of error (e.g. calculation, logic, missing_step, format)\n'
        '- "error_reason": brief explanation of what went wrong\n'
        '- "correct_reasoning": step-by-step correct reasoning\n'
        '- "corrected_solution": the correct final answer\n\n'
        "Respond ONLY with valid JSON."
    )


def analyze_failure(
    client: TeacherClient,
    input_text: str,
    student_output: str,
    gold_output: str,
) -> dict:
    """Use the teacher model to analyze a student failure and return structured feedback."""
    template = _load_prompt_template()
    prompt = template.format(
        input=input_text,
        student_output=student_output,
        gold_output=gold_output,
    )

    response = client.query(prompt, temperature=0.3)

    try:
        # Try to extract JSON from the response
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        return json.loads(response)
    except json.JSONDecodeError:
        return {
            "error_type": "unknown",
            "error_reason": response,
            "correct_reasoning": "",
            "corrected_solution": gold_output,
        }


def analyze_failures_batch(
    client: TeacherClient,
    failures: list[dict],
) -> list[dict]:
    """Analyze a batch of failures."""
    results = []
    for failure in failures:
        analysis = analyze_failure(
            client,
            failure["input"],
            failure["student_output"],
            failure["gold_output"],
        )
        result = {**failure, **analysis}
        results.append(result)
    return results
