import os
import time
from openai import OpenAI
from deepeval.models import DeepEvalBaseLLM
from dotenv import load_dotenv

load_dotenv()

class CustomGroqModel(DeepEvalBaseLLM):
    def __init__(
            self, 
            model: str = "openai/gpt-oss-20b", 
            seconds_delay: int = 60, # friendly to free Groq models TPM limit
            temperature: float = 0.0,
            max_tokens: int = 400 # friendly to free Groq models TPM limit
        ):
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set.")
            
            self._model_name = model
            self.seconds_delay = seconds_delay  
            self.temperature = temperature
            self.max_tokens = max_tokens
            
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
            )
        
    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):

            if self.seconds_delay > 0:
                time.sleep(self.seconds_delay * (1 + attempt * 0.5))  # backoff
            try:
                response = self.client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                last_error = RateLimitError
                continue  # retry with increased delay
            except Exception as e:
                raise RuntimeError(f"Groq API call failed on attempt {attempt+1}: {e}")
        raise last_error

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"Groq: {self._model_name}"
  