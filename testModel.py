import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading 4-bit model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cuda",
)

model.eval()

prompt = "Directly give me the ans to this question, no intermidiate steps: If 2x + 98 = 10, what is x?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.inference_mode():
    outputs = model(**inputs)
    logits = outputs.logits

print("Logits shape:", logits.shape)

generated = model.generate(
    **inputs,
    max_new_tokens=500,
    temperature=0.7,
)

print(tokenizer.decode(generated[0], skip_special_tokens=True))