# 05_retrieval.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load the already-stored vectorstore (no re-embedding!)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="./vectorstore",
    embedding_function=embedding_model
)

# Create a retriever (fetches top 3 most relevant chunks)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}   # return top 3 chunks
)

# Test retrieval
query = "What is the refund policy?"
relevant_chunks = retriever.invoke(query)

print(f"Query: {query}")
print(f"Retrieved {len(relevant_chunks)} chunks:\n")

for i, chunk in enumerate(relevant_chunks):
    print(f"--- Chunk {i+1} ---")
    print(f"Source: {chunk.metadata}")
    print(f"Content: {chunk.page_content[:200]}")
    print()