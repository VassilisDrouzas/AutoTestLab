import os
from groq import Groq

class LLamaLLM:
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile"):
        """Initialize the LLama model with its API key and model."""
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate_response(self, prompt: str) -> str:
        """Call the LLama model using the Groq client."""
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model,
        )

        if chat_completion and len(chat_completion.choices) > 0:
            return chat_completion.choices[0].message.content
        else:
            return "The model did not return a valid response."
