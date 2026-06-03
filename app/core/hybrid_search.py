# app/core/hybrid_search.py

from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from app.core.langchain_compat import EnsembleRetriever
from app.config import settings

def get_hybrid_retriever(chunks, collection_name: str = "default"):
    """
    Combines semantic search + keyword search for better retrieval
    """

    # ── RETRIEVER 1: Semantic (Dense) ──────────────────────
    embeddings = OllamaEmbeddings(model=settings.EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=settings.VECTORSTORE_DIR
    )
    semantic_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # ── RETRIEVER 2: Keyword (Sparse/BM25) ─────────────────
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 3

    # ── COMBINE BOTH ────────────────────────────────────────
    # weights = [semantic_weight, keyword_weight]
    # 0.6 semantic + 0.4 keyword = good balance
    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, keyword_retriever],
        weights=[0.6, 0.4]
    )

    return hybrid_retriever
