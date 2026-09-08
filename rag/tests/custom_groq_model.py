import os
import time
from typing import List
from openai import OpenAI
from deepeval.models import DeepEvalBaseLLM
from dotenv import load_dotenv
import time

load_dotenv()

class CustomGroqModel(DeepEvalBaseLLM):
    def __init__(
            self, 
            model: str = "groq/compound-mini", 
            seconds_delay: int = 5, 
            temperature: float = 0.0,
            max_tokens: int = 1000
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
        if self.seconds_delay > 0:
            time.sleep(self.seconds_delay)
        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise e
            # return f"ERROR: {str(e)}"

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"Groq: {self._model_name}"