from pathlib import Path

from peft import LoraConfig
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


def fine_tune(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_data: list[dict],
    output_dir: str,
    lora_config: LoraConfig = None,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1024,
):
    """Fine-tune the student model on training samples using LoRA + SFTTrainer."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Format samples into text for SFT
    texts = []
    for sample in train_data:
        text = format_training_sample(sample, tokenizer)
        texts.append(text)

    dataset = Dataset.from_dict({"text": texts})

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        dataset_text_field="text",
        max_length=max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(output_dir)

    return model


def format_training_sample(sample: dict, tokenizer: PreTrainedTokenizer) -> str:
    """Format a training sample into chat-template text."""
    messages = [
        {
            "role": "user",
            "content": f"{sample.get('instruction', 'Solve this problem.')}\n\n{sample.get('input', '')}",
        },
        {
            "role": "assistant",
            "content": f"{sample.get('reasoning', '')}\n\n#### {sample.get('output', '')}",
        },
    ]

    # Use tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # Fallback: simple text format
    return (
        f"### Instruction:\n{messages[0]['content']}\n\n"
        f"### Response:\n{messages[1]['content']}"
    )
