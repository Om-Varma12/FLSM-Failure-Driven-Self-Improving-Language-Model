def order_by_curriculum(failures: list[dict]) -> list[dict]:
    """Order failures from easy to hard based on error type heuristics.

    Ordering priority (easiest first):
    1. format errors (surface-level mistakes)
    2. calculation errors (minor numeric mistakes)
    3. missing_step errors (partial understanding)
    4. logic errors (fundamental misunderstanding)
    5. unknown / other
    """
    priority = {
        "format": 0,
        "calculation": 1,
        "missing_step": 2,
        "logic": 3,
    }

    def sort_key(failure: dict) -> int:
        error_type = failure.get("error_type", "unknown").lower()
        for key, rank in priority.items():
            if key in error_type:
                return rank
        return 4  # unknown goes last

    return sorted(failures, key=sort_key)
