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

    def generate_answer(self, context_text: str, question: str, chat_history: list[dict] = None) -> str:
        """
        Takes the database context, the new question, and the chat history.
        Structures them into a clean array for the Groq API.
        """
        if chat_history is None:
            chat_history = []

        # 1. The System Prompt (This tells the AI how to behave and gives it the database context)
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful assistant. Use the following context to answer the user's questions.\n\nContext:\n{context_text}"
            }
        ]
        
        # 2. Append the previous conversation history so the AI remembers
        messages.extend(chat_history)
        
        # 3. Append the brand new question
        messages.append({
            "role": "user",
            "content": question
        })
        
        # 4. Send the whole package to Groq
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        return response.choices[0].message.content