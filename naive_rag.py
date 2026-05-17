import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()

file = open("facts.txt","r")
content = file.read()

question = "What is melodi"

augmented_prompt = f"Context information is below.\n-----\n{content}\n----\nGiven the context information answer the following question:{question}"

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role":"user",
            "content":augmented_prompt
        }
    ]
)
print(response.choices[0].message.content)
file.close()
