from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizer


@dataclass
class InstructionDataCollator:
    """Collator that pads and creates labels for instruction-tuning."""

    tokenizer: PreTrainedTokenizer
    max_length: int = 1024

    def __call__(self, features: list[dict]) -> dict:
        texts = [f["text"] for f in features]
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Labels are the same as input_ids (causal LM)
        batch["labels"] = batch["input_ids"].clone()
        # Mask padding tokens in labels
        batch["labels"][batch["attention_mask"] == 0] = -100
        return batch
