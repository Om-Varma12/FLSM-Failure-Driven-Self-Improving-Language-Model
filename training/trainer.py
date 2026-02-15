from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import config

def fine_tune(model, tokenizer):
    try:
        dataset = load_dataset(
            "json",
            data_files=config.FAILURE_MEMORY_PATH,
            split="train"
        )
    except:
        print("No failure data yet.")
        return model

    def format_data(example):
        return tokenizer(
            example["prompt"] + example["correct_answer"],
            truncation=True
        )

    tokenized = dataset.map(format_data)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05
    )

    model = get_peft_model(model, lora_config)

    args = TrainingArguments(
        output_dir="lora-out",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=5
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized
    )

    trainer.train()
    return model