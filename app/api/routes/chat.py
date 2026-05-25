# app/api/routes/chat.py

from fastapi import APIRouter, HTTPException
from app.models.schemas import QuestionRequest, AnswerResponse
from app.core.rag import query_rag

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        result = query_rag(request.question, request.collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AnswerResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"]
    )