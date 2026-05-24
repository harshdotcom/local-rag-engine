# RAG Pipeline — Day 2

A step-by-step Retrieval-Augmented Generation (RAG) implementation using LangChain, ChromaDB, and local LLMs via Ollama.

## What it does

Loads a document, chunks it, embeds the chunks into a vector store, then answers questions about it using only the document's content — no hallucination from outside knowledge.

## Project structure

```
01_document_loading.py   # Load PDFs and text files with LangChain loaders
02_chunking.py           # Split documents into overlapping chunks
03_embeddings.py         # Generate and compare vector embeddings
04_vectorstore.py        # Store and query embeddings with ChromaDB
05_retrieval.py          # Retrieve relevant chunks for a query
06_full_rag_pipeline.py  # End-to-end RAG chatbot (run this)
data/                    # Place your documents here
vectorstore/             # Auto-generated ChromaDB files (git-ignored)
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally with the required models pulled

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install langchain langchain-community langchain-ollama langchain-chroma \
            langchain-text-splitters chromadb pypdf sentence-transformers numpy
```

## Usage

1. Drop your PDF (or `.txt`) into `data/` and update `DOCS_PATH` in `06_full_rag_pipeline.py`.
2. Run the pipeline:

```bash
python 06_full_rag_pipeline.py
```

3. Ask questions about your document. Type `quit` to exit.

## How the RAG pipeline works

```
Document → Chunks → Embeddings → ChromaDB
                                     ↓
Question → Embed → Retrieve top-k chunks → Prompt + LLM → Answer
```

The LLM is instructed to answer **only** from the retrieved context, so it will say "I don't know based on the document" rather than make something up.

## Configuration (in `06_full_rag_pipeline.py`)

| Variable | Default | Description |
|---|---|---|
| `DOCS_PATH` | `data/sample.pdf` | Path to your document |
| `VECTORSTORE_PATH` | `./vectorstore` | Where ChromaDB persists |
| `LLM_MODEL` | `llama3.2` | Ollama model to use |
