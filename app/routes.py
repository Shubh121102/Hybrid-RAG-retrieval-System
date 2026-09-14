from fastapi import APIRouter
from fastapi.responses import HTMLResponse, StreamingResponse
import markdown
import json
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

@router.post("/stream_answer", response_class = StreamingResponse)
def stream_answer(request: ChatRequest):
    def stream():

        for chunk in rag_chain.stream(request.question):
            if chunk:
                yield json.dumps({
                    "type": "answer_chunk",
                    "content": chunk
                }) + "\n"


        yield json.dumps({
            "type": "final_message",
            "content": "Streaming Completed"
        }) + "\n"


    return StreamingResponse(stream(), media_type = "application/x-ndjson")