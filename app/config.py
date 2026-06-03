# app/config.py

from pydantic_settings import BaseSettings
from pydantic import field_validator

class Settings(BaseSettings):
    # App
    APP_NAME: str = "local-rag-engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Paths
    DATA_DIR: str = "data"
    VECTORSTORE_DIR: str = "vectorstore"

    # RAG settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RETRIEVER_K: int = 3

    # Ollama
    LLM_MODEL: str = "llama3.2:latest"
    EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "production", "prod"}:
                return False
            if normalized in {"debug", "development", "dev"}:
                return True

        return value

    class Config:
        env_file = ".env"

# Single instance used across the whole app
settings = Settings()
