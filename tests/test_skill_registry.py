"""Tests for the skill registry."""
from skills.registry import get_skill, SKILL_REGISTRY
from skills.base_skill import BaseSkill


def test_math_skill_registered():
    assert "math" in SKILL_REGISTRY


def test_get_skill_returns_instance():
    skill = get_skill("math")
    assert isinstance(skill, BaseSkill)


def test_get_skill_unknown_raises():
    try:
        get_skill("nonexistent_skill")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_math_skill_name():
    skill = get_skill("math")
    assert skill.name == "math"


def test_math_skill_format_prompt():
    skill = get_skill("math")
    prompt = skill.format_prompt("What is 2 + 2?")
    assert "2 + 2" in prompt


def test_math_skill_evaluate_correct():
    skill = get_skill("math")
    assert skill.evaluate("#### 42", "#### 42") is True
    assert skill.evaluate("The answer is 42", "42") is True


def test_math_skill_evaluate_incorrect():
    skill = get_skill("math")
    assert skill.evaluate("#### 43", "#### 42") is False


if __name__ == "__main__":
    test_math_skill_registered()
    test_get_skill_returns_instance()
    test_get_skill_unknown_raises()
    test_math_skill_name()
    test_math_skill_format_prompt()
    test_math_skill_evaluate_correct()
    test_math_skill_evaluate_incorrect()
    print("All skill registry tests passed!")
