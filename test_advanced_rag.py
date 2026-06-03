# test_advanced_rag.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.advanced_rag import AdvancedRAGPipeline

# Load your PDF
loader = PyPDFLoader("data/sample.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)
chunks = splitter.split_documents(docs)

# Build advanced pipeline
pipeline = AdvancedRAGPipeline(chunks)

# Test queries
questions = [
    "What is this document about?",
    "What are the main requirements?",
    "What is something NOT in this document?",  # should say "I don't know"
]

for q in questions:
    print(f"\n{'='*50}")
    print(f"Q: {q}")
    result = pipeline.query(q)
    print(f"A: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Rerank scores: {result['rerank_scores']}")