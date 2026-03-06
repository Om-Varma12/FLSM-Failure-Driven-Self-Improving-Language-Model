from .base_skill import BaseSkill
from .math_skill import MathSkill

SKILL_REGISTRY: dict[str, type[BaseSkill]] = {
    "math": MathSkill,
}


def register_skill(name: str, skill_cls: type[BaseSkill]):
    """Register a new skill class."""
    SKILL_REGISTRY[name] = skill_cls


def get_skill(name: str) -> BaseSkill:
    """Get a skill instance by name."""
    if name not in SKILL_REGISTRY:
        available = ", ".join(SKILL_REGISTRY.keys())
        raise ValueError(f"Unknown skill '{name}'. Available: {available}")
    return SKILL_REGISTRY[name]()
