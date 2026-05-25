# app/api/routes/document.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import UploadResponse
from app.core.rag import ingest_document
from app.config import settings
import os, shutil

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = "default"
):
    # Validate file type
    allowed = [".pdf", ".txt"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Use: {allowed}"
        )

    # Save file to disk
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    file_path = os.path.join(settings.DATA_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Ingest into RAG pipeline
    try:
        chunks_created = ingest_document(file_path, collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return UploadResponse(
        filename=file.filename,
        chunks_created=chunks_created,
        message=f"Successfully processed {file.filename}"
    )