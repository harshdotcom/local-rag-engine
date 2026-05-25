# app/core/rag.py

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from app.config import settings
import os

# ── PROMPT ─────────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Answer ONLY from the context below.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""
)

# ── EMBEDDINGS (shared instance) ───────────────────────────
def get_embeddings():
    return OllamaEmbeddings(model=settings.EMBED_MODEL)

# ── VECTORSTORE ─────────────────────────────────────────────
def get_vectorstore(collection_name: str = "default"):
    return Chroma(
        collection_name=collection_name,
        persist_directory=settings.VECTORSTORE_DIR,
        embedding_function=get_embeddings()
    )

# ── INGEST DOCUMENT ─────────────────────────────────────────
def ingest_document(file_path: str, collection_name: str = "default") -> int:
    # Load
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    docs = loader.load()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    # Store
    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=settings.VECTORSTORE_DIR
    )

    return len(chunks)

# ── QUERY ───────────────────────────────────────────────────
def query_rag(question: str, collection_name: str = "default") -> dict:
    vectorstore = get_vectorstore(collection_name)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.RETRIEVER_K}
    )

    llm = OllamaLLM(model=settings.LLM_MODEL, temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Get sources for transparency
    retrieved_docs = retriever.invoke(question)
    sources = list(set([
        doc.metadata.get("source", "unknown")
        for doc in retrieved_docs
    ]))

    # Build and run chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)

    return {
        "answer": answer,
        "sources": sources
    }