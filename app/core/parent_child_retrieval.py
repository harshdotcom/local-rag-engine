# app/core/parent_child_retrieval.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from app.core.langchain_compat import InMemoryStore, ParentDocumentRetriever
from app.config import settings

def get_parent_child_retriever(documents):
    """
    Small chunks for retrieval, large chunks for context
    """

    # ── CHILD SPLITTER (small - for retrieval) ─────────────
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,    # small = precise retrieval
        chunk_overlap=20
    )

    # ── PARENT SPLITTER (large - for context) ──────────────
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # large = full context for LLM
        chunk_overlap=100
    )

    # ── VECTORSTORE (stores child chunks) ──────────────────
    embeddings = OllamaEmbeddings(model=settings.EMBED_MODEL)
    vectorstore = Chroma(
        collection_name="parent_child",
        embedding_function=embeddings,
        persist_directory=settings.VECTORSTORE_DIR
    )

    # ── DOCSTORE (stores parent chunks in memory) ───────────
    docstore = InMemoryStore()

    # ── BUILD RETRIEVER ─────────────────────────────────────
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    # Index documents
    retriever.add_documents(documents)

    return retriever

# Result:
# Search uses 200-char child chunks (precise) ✅
# LLM receives 1000-char parent chunks (full context) ✅
