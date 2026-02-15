import requests
import config

OLLAMA_URL = "http://localhost:11434/api/chat"

def student_generate(prompt: str) -> str:
    payload = {
        "model": config.STUDENT_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {
            "temperature": config.TEMPERATURE,
            "num_predict": config.MAX_NEW_TOKENS
        },
        "stream": False
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=120)
        res.raise_for_status()
        data = res.json()
        return data["message"]["content"]

    except Exception as e:
        print("Student model error:", e)
        return ""