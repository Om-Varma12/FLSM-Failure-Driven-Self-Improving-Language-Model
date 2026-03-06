import json
from pathlib import Path


def save_checkpoint(iteration: int, results: list[dict], output_dir: str):
    """Save pipeline state after an iteration."""
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "last_iteration": iteration,
        "results": results,
    }

    checkpoint_path = checkpoint_dir / "pipeline_state.json"
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    print(f"  Checkpoint saved: {checkpoint_path}")


def load_checkpoint(output_dir: str) -> dict | None:
    """Load pipeline state from a previous run."""
    checkpoint_path = Path(output_dir) / "pipeline_state.json"
    if not checkpoint_path.exists():
        return None

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        return json.load(f)
