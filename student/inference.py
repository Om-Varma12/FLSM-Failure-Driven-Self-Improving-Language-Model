import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def run_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    max_new_tokens: int = 500,
    temperature: float = 0.7,
    batch_size: int = 4,
) -> list[str]:
    """Run inference on a list of prompts and return generated text."""
    model.eval()
    results = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        for output in outputs:
            # Decode only the generated tokens (skip the input)
            input_len = inputs["input_ids"].shape[1]
            generated = tokenizer.decode(
                output[input_len:], skip_special_tokens=True
            )
            results.append(generated.strip())

    return results


def run_single_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 500,
    temperature: float = 0.7,
) -> str:
    """Run inference on a single prompt."""
    return run_inference(
        model, tokenizer, [prompt], max_new_tokens, temperature, batch_size=1
    )[0]
