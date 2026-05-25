# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import document, chat
from app.models.schemas import HealthResponse
from app.config import settings
import os

# ── CREATE APP ──────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production RAG API — query your documents locally"
)

# ── CORS MIDDLEWARE ─────────────────────────────────────────
# This allows your Angular frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production, set your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ROUTES ──────────────────────────────────────────────────
app.include_router(document.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")

# ── HEALTH CHECK ────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        message=f"{settings.APP_NAME} is running"
    )

# ── STARTUP ─────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.VECTORSTORE_DIR, exist_ok=True)
    print(f"✅ {settings.APP_NAME} started!")