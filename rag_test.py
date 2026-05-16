import os
from groq import Groq
from dotenv import load_dotenv

# 1. AUTHENTICATION
# Load the environment variables
# Initialize the Groq client
load_dotenv()
client = Groq()

# 2. PAYLOAD CONSTRUCTION
# Create a variable (e.g., 'response') and assign it to client.chat.completions.create()
response  = client.chat.completions.create(
    # Inside the create method, define your 'model' (e.g., "llama3-8b-8192")
    model="llama-3.1-8b-instant",
    # Inside the create method, define your 'messages' list (role: user, content: your question)
    messages=[
        {
            "role":"user",
            "content":"Explain how to make a enterpise rag system."
        }
    ]
)
# 3. RESPONSE PARSING
# Print the extracted text using the path: response.choices[0].message.content
print(response.choices[0].message.content)
