from pathlib import Path

from peft import LoraConfig, get_peft_model, PeftModel, TaskType


DEFAULT_LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)


def apply_lora(model, lora_config: LoraConfig = None):
    """Apply LoRA adapters to a base model."""
    if lora_config is None:
        lora_config = DEFAULT_LORA_CONFIG
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def save_lora(model, path: str):
    """Save LoRA adapter weights to disk."""
    Path(path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)


def load_lora(model, path: str):
    """Load LoRA adapter weights from disk onto a base model."""
    return PeftModel.from_pretrained(model, path)


def merge_lora(model):
    """Merge LoRA weights into the base model and unload the adapter."""
    return model.merge_and_unload()
