# 02_chunking.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the document
loader = PyPDFLoader("data/sample.pdf")
docs = loader.load()

# Create the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # max characters per chunk
    chunk_overlap=50,     # overlap between chunks
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""]  # tries these in order
)

# Split documents into chunks
chunks = splitter.split_documents(docs)

print(f"Original pages: {len(docs)}")
print(f"Total chunks created: {len(chunks)}")
print(f"\n--- Chunk 0 ---\n{chunks[0].page_content}")
print(f"\n--- Chunk 1 ---\n{chunks[1].page_content}")

# Notice the overlap between chunk 0 and chunk 1!