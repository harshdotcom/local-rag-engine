# app/models/schemas.py

from pydantic import BaseModel
from typing import Optional

# ── REQUEST MODELS (what client sends) ─────────────────────

class QuestionRequest(BaseModel):
    question: str
    collection_name: Optional[str] = "default"

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is this document about?",
                "collection_name": "my_docs"
            }
        }

# ── RESPONSE MODELS (what API returns) ─────────────────────

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str] = []
    status: str = "success"

class UploadResponse(BaseModel):
    filename: str
    chunks_created: int
    status: str = "success"
    message: str

class HealthResponse(BaseModel):
    status: str
    message: str