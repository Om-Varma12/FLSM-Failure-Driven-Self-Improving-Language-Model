import re

from .base_skill import BaseSkill


class MathSkill(BaseSkill):
    """Skill implementation for mathematical reasoning tasks."""

    @property
    def name(self) -> str:
        return "math"

    def format_prompt(self, input_text: str) -> str:
        return (
            "Solve the following math problem step by step. "
            "Put your final answer after '#### '.\n\n"
            f"Problem: {input_text}\n\n"
            "Solution:"
        )

    def evaluate(self, prediction: str, gold: str) -> bool:
        """Check if the predicted answer matches the gold answer numerically."""
        pred_num = self._extract_number(prediction)
        gold_num = self._extract_number(gold)

        if pred_num is None or gold_num is None:
            # Fall back to exact string match (stripped, lowered)
            return prediction.strip().lower() == gold.strip().lower()

        return abs(pred_num - gold_num) < 1e-6

    def get_metric_name(self) -> str:
        return "accuracy"

    @staticmethod
    def _extract_number(text: str) -> float | None:
        """Extract the final numeric answer from text.

        Looks for '#### <number>' pattern first (GSM8K format),
        then falls back to the last number in the text.
        """
        # Try GSM8K format: #### <number>
        match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
        if match:
            return float(match.group(1).replace(",", ""))

        # Fallback: find the last number in the text
        numbers = re.findall(r"-?[\d,]+\.?\d*", text)
        if numbers:
            try:
                return float(numbers[-1].replace(",", ""))
            except ValueError:
                return None
        return None
