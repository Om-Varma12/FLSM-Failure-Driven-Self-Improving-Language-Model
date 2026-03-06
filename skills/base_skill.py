from abc import ABC, abstractmethod


class BaseSkill(ABC):
    """Abstract base class for all skill implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifier for this skill."""
        ...

    @abstractmethod
    def format_prompt(self, input_text: str) -> str:
        """Format a raw input into a prompt suitable for the student model."""
        ...

    @abstractmethod
    def evaluate(self, prediction: str, gold: str) -> bool:
        """Evaluate whether the prediction matches the gold answer."""
        ...

    @abstractmethod
    def get_metric_name(self) -> str:
        """Return the name of the primary evaluation metric."""
        ...
