from fastapi import APIRouter
from fastapi.responses import HTMLResponse
import markdown
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from rag_chain import rag_chain
from .schema import ChatRequest

router = APIRouter(prefix="/rag", tags=["RAG Operations"])


@router.post("/chat", response_class=HTMLResponse)
def chat(request: ChatRequest):
    """
    Endpoint to handle chat requests.
    """
    result = rag_chain(request.question)
    return markdown.markdown(result)
    # return {"result": result}