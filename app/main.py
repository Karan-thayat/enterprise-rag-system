from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.document_parser import DocumentParser
from app.vector_db import VectorDBClient
from app.llm_generator import LLMClient

# 1. Initialize the FastAPI server
app = FastAPI(title="Enterprise RAG System API")

# 2. Boot up our modules (This happens exactly once when the server starts)
print("Booting up modules... (Loading ML models into memory)")
db_client = VectorDBClient()
llm_client = LLMClient()
# (We don't initialize the parser here because we aren't uploading files via API yet, 
# but it's ready for when you need it!)

# 3. Define what our incoming JSON request should look like
class QuestionRequest(BaseModel):
    question: str

# 4. Create the API Endpoint (POST /ask)
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        # Step A: Search the Vector DB for context chunks
        context = db_client.search(request.question)
        
        # Step B: Pass context and question to the LLM
        answer = llm_client.generate_answer(context_text=context, question=request.question)
        
        # Step C: Return a clean JSON response to the user
        return {
            "status": "success",
            "question": request.question,
            "answer": answer
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))