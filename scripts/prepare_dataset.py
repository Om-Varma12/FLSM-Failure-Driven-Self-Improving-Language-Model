"""Download and preprocess a math dataset (GSM8K) for the FSLM pipeline."""
import json
from pathlib import Path

from datasets import load_dataset


def prepare_gsm8k(
    train_size: int = 300,
    test_size: int = 200,
    output_dir: str = "data/processed/math",
):
    """Download GSM8K and convert to JSONL format for the pipeline."""
    print("Downloading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process train split
    train_samples = []
    for sample in dataset["train"].select(range(min(train_size, len(dataset["train"])))):
        train_samples.append({
            "input": sample["question"],
            "expected_output": sample["answer"],
        })

    train_path = output_path / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved {len(train_samples)} train samples to {train_path}")

    # Process test split
    test_samples = []
    for sample in dataset["test"].select(range(min(test_size, len(dataset["test"])))):
        test_samples.append({
            "input": sample["question"],
            "expected_output": sample["answer"],
        })

    test_path = output_path / "test.jsonl"
    with open(test_path, "w", encoding="utf-8") as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved {len(test_samples)} test samples to {test_path}")


if __name__ == "__main__":
    prepare_gsm8k()
