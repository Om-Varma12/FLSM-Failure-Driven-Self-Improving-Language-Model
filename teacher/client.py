import os
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 3
BASE_DELAY = 2  # seconds


class TeacherClient:
    """Client for querying the teacher model via Groq API."""

    def __init__(self, model: str = DEFAULT_MODEL):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment. Check your .env file.")
        self.client = Groq(api_key=api_key)
        self.model = model

    def query(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send a prompt to the teacher model and return the response."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_DELAY * (2 ** attempt)
                    print(f"Groq API error (attempt {attempt + 1}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise

    def query_batch(
        self,
        prompts: list[str],
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> list[str]:
        """Query the teacher model with multiple prompts sequentially."""
        results = []
        for prompt in prompts:
            result = self.query(prompt, system_prompt, temperature, max_tokens)
            results.append(result)
        return results
