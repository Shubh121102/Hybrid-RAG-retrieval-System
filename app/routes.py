from fastapi import APIRouter
from rag_chain import rag_chain
from schema import ChatRequest

router = APIRouter(prefix="/ingestion", tags=["Ingestion"])


@router.get("/chat")
def chat(request: ChatRequest):
    """
    Endpoint to handle chat requests.
    """
    result = rag_chain()
    return {"result": result}