import json
import config
import os

def store_failure(prompt, wrong, correct, skill):
    os.makedirs("data", exist_ok=True)

    record = {
        "prompt": prompt,
        "wrong_answer": wrong,
        "correct_answer": correct,
        "skill": skill
    }

    with open(config.FAILURE_MEMORY_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
