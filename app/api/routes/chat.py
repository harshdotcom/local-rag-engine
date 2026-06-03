# app/api/routes/chat.py  (updated)

from fastapi import APIRouter, HTTPException
from app.models.schemas import QuestionRequest, AnswerResponse
from app.core.conversational_rag import ConversationalRAG
from app.core.rag import get_vectorstore

router = APIRouter(prefix="/chat", tags=["Chat"])

# Store sessions in memory (Day 5 = Redis for production)
sessions: dict[str, ConversationalRAG] = {}

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    session_id = request.session_id or "default"

    # Get or create session
    if session_id not in sessions:
        vectorstore = get_vectorstore(request.collection_name)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        sessions[session_id] = ConversationalRAG(retriever)

    rag = sessions[session_id]

    try:
        answer = rag.ask(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AnswerResponse(
        question=request.question,
        answer=answer,
        status="success"
    )

@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    if session_id in sessions:
        sessions[session_id].clear_history()
    return {"message": "History cleared", "session_id": session_id}