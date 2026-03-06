import json
from pathlib import Path

from .client import TeacherClient

TESTGEN_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "teacher_testgen.txt"


def _load_prompt_template() -> str:
    if TESTGEN_PROMPT_PATH.exists():
        return TESTGEN_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "You are an expert teacher. Generate {count} {difficulty} difficulty "
        "test problems for the skill: {skill_name}.\n\n"
        "Each problem should test the student's ability in this skill area.\n\n"
        "Respond with a JSON array where each element has:\n"
        '- "input": the problem statement\n'
        '- "expected_output": the correct answer\n'
        '- "reasoning": step-by-step solution\n\n'
        "Respond ONLY with valid JSON array."
    )


def generate_test_cases(
    client: TeacherClient,
    skill_name: str,
    difficulty: str = "medium",
    count: int = 10,
) -> list[dict]:
    """Use the teacher model to generate new test cases for a skill."""
    template = _load_prompt_template()
    prompt = template.format(
        skill_name=skill_name,
        difficulty=difficulty,
        count=count,
    )

    response = client.query(prompt, temperature=0.8, max_tokens=4096)

    try:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        return json.loads(response)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse teacher test cases. Raw response:\n{response[:200]}")
        return []
