import re


def exact_match(prediction: str, gold: str) -> bool:
    """Check if prediction exactly matches gold (case-insensitive, stripped)."""
    return prediction.strip().lower() == gold.strip().lower()


def numeric_match(prediction: str, gold: str, tolerance: float = 1e-6) -> bool:
    """Check if the numeric values in prediction and gold match."""
    pred_num = _extract_last_number(prediction)
    gold_num = _extract_last_number(gold)

    if pred_num is None or gold_num is None:
        return exact_match(prediction, gold)

    return abs(pred_num - gold_num) < tolerance


def _extract_last_number(text: str) -> float | None:
    """Extract the last number from text."""
    # Try GSM8K format first
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass

    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            return None
    return None
