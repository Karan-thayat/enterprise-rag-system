import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

# --- 1. SETUP & AUTHENTICATION ---
load_dotenv()
client = Groq()

print("Waking up Embedding Model (CPU)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Notice we use get_collection here instead of get_or_create
collection = chroma_client.get_collection(name="pdf_knowledge_base")

# --- 2. THE QUESTION ---
# I am using a question directly related to the MELODI PDF you ingested!
question = "What is the key principle behind the MELODI architecture?"
print(f"\nQuestion: {question}")

# --- 3. RETRIEVAL ---
# Convert the text question into a 384-dimension embedding
question_embedding = embedding_model.encode(question).tolist()

# Query the database for the top 2 most relevant chunks
print("Searching the vector database for answers...")
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)

# Extract the raw text from the database results and join them together
retrieved_chunks = results['documents'][0]
context_text = "\n\n".join(retrieved_chunks)

# --- 4. AUGMENTATION ---
augmented_prompt = f"Context information is below.\n-----\n{context_text}\n----\nGiven the context information, answer the following question: {question}"

# --- 5. GENERATION ---
print("Sending context to Groq for generation...\n")
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "user",
            "content": augmented_prompt
        }
    ]
)

print("================ FINAL AI ANSWER ================")
print(response.choices[0].message.content)
print("=================================================")