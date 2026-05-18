import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables once when the module boots up
load_dotenv()

class LLMClient:
    def __init__(self):
        # Initialize the connection when the server starts
        self.client = Groq()
        self.model = "llama-3.1-8b-instant"

    def generate_answer(self, context_text: str, question: str) -> str:
        """
        Takes the raw context from the database and the user's question,
        builds the augmented prompt, and returns the AI's string response.
        """
        augmented_prompt = f"Context information is below.\n-----\n{context_text}\n----\nGiven the context information, answer the following question: {question}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": augmented_prompt
                }
            ]
        )
        
        return response.choices[0].message.content