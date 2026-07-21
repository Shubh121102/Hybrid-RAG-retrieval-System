from fastapi import APIRouter
from rag_chain import rag_chain
from pydantic import BaseModel

router = APIRouter(prefix="/ingestion", tags=["Ingestion"])

class ChatRequest(BaseModel):
    question: str
    answer: str

@router.get("/chat")
def chat():
    """
    Endpoint to handle chat requests.
    """
    result = rag_chain()
    return {"result": result}