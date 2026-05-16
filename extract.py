import fitz
from pathlib import Path

# Step 1: Open the Document
# Define your file path (e.g., pdf_path = "test.pdf")
pdf_path = Path("test.pdf")
print(pdf_path)
# Create a 'document' object by passing the path into fitz.open()
doc = fitz.open(pdf_path)
# Step 2: Select the Target Page
# PDFs are essentially lists of pages. 
# Create a 'page' variable and access the first page of your document object using index [0]
extracted_text = doc[0].get_text()
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