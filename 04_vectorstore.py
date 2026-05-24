
# 04_vectorstore.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Step 1: Load
loader = PyPDFLoader("data/sample.pdf")
docs = loader.load()

# Step 2: Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Step 3: Embed + Store (all in one line!)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./vectorstore"   # saves to disk
)

print(f"✅ {len(chunks)} chunks stored in ChromaDB!")
print("Vectorstore saved to ./vectorstore folder")