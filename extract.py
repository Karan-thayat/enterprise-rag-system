# import fitz
from liteparse import LiteParse
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Step 1: Open the Document
# Define your file path (e.g., pdf_path = "test.pdf")
parser = LiteParse()

# pdf_path = Path("test.pdf")
pdf_path = "test.pdf"
result = parser.parse(pdf_path)
# Create a 'document' object by passing the path into fitz.open()
# doc = fitz.open(pdf_path)
# Step 2: Select the Target Page
# PDFs are essentially lists of pages. 
# Create a 'page' variable and access the first page of your document object using index [0]
# extracted_text = doc[0].get_text()
extracted_text  = result.text
# Step 3: Extract the Text
# Call the .get_text() method on your 'page' variable and store the result in a new variable
# Step 4: Output
# Print the extracted text variable to the terminal
print(extracted_text)
# Chunking the pdf
chunk_size = 1000   # How many characters per chunk
chunk_overlap = 200 # How many characters to overlap so we don't cut sentences in half
chunks = []
# Loop through the text and slice it into overlapping blocks
for i in range(0, len(extracted_text), chunk_size - chunk_overlap):
    chunk = extracted_text[i : i + chunk_size]
    chunks.append(chunk)

print(f"Successfully split the document into {len(chunks)} chunks!")
print("Here is a preview of Chunk 1:")
print("-----------------------------")
print(chunks[0])
# --- STEP 5: VECTORIZATION (EMBEDDINGS) ---

# 1. Load the local embedding model
# 'all-MiniLM-L6-v2' is the industry standard for fast, local RAG testing.
# It converts text into an array of exactly 384 numbers.
print("Loading embedding model (this may take a few seconds the first time)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2',device = 'cpu')

# 2. Generate the embeddings
# We pass our list of text chunks directly into the encode method
print(f"Generating embeddings for {len(chunks)} chunks...")
embeddings = embedding_model.encode(chunks)

# 3. Verify the output
print("\n--- Embedding Verification ---")
print(f"Total embeddings created: {len(embeddings)}")
print(f"Dimensions of a single embedding: {len(embeddings[0])} numbers")

# --- STEP 6: VECTOR DATABASE STORAGE ---

# 1. Initialize the Chroma Client
# This creates a folder named 'chroma_db' in your current directory to save data permanently.
print("\nInitializing Chroma Vector Database...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 2. Create a "Collection" (Similar to a Table in standard SQL)
# We use get_or_create so we don't throw an error if we run this script twice.
collection = chroma_client.get_or_create_collection(name="pdf_knowledge_base")

# 3. Generate unique IDs for each chunk
# Chroma requires every chunk to have a unique ID (e.g., "chunk_0", "chunk_1")
chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

# 4. Insert the data into the database
print("Inserting chunks and embeddings into the database...")
collection.upsert(
    ids=chunk_ids,
    embeddings=embeddings,      # The 384-dimension math arrays
    documents=chunks            # The raw English text (so we can read it later)
)

print(f"\nSuccess! Stored {collection.count()} chunks in the Vector Database.")
print("Check your project folder - you should see a new 'chroma_db' directory!")