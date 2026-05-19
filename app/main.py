import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.document_parser import DocumentParser
from app.vector_db import VectorDBClient
from app.llm_generator import LLMClient

# 1. Initialize the FastAPI server
app = FastAPI(title="Enterprise RAG System API")

# 2. Boot up our modules 
print("Booting up modules... (Loading ML models into memory)")
db_client = VectorDBClient()
llm_client = LLMClient()
parser = DocumentParser() # <--- We are waking the parser up now!

# 3. Define the Question Request model
class QuestionRequest(BaseModel):
    question: str

# 4. The Upload Endpoint (NEW)
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Security check: Ensure it's actually a PDF
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    # Create a temporary file path
    temp_file_path = f"temp_{file.filename}"
    
    try:
        # Save the uploaded file to the server's disk
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Parse the PDF and generate chunks
        print(f"Parsing uploaded file: {file.filename}")
        chunks = parser.process_pdf(temp_file_path)
        
        # Upsert the chunks into ChromaDB
        print(f"Embedding and storing {len(chunks)} chunks...")
        num_chunks = db_client.upsert_chunks(chunks)
        
        return {
            "status": "success", 
            "message": f"Successfully processed and stored {num_chunks} chunks from {file.filename}."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # ALWAYS clean up temporary files, even if the parsing crashes
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# 5. The Ask Endpoint (Unchanged)
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        context = db_client.search(request.question)
        answer = llm_client.generate_answer(context_text=context, question=request.question)
        
        return {
            "status": "success",
            "question": request.question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))