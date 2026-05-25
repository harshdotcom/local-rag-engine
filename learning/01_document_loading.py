# 01_document_loading.py

from langchain_community.document_loaders import PyPDFLoader, TextLoader

# ----- Loading a PDF -----
pdf_loader = PyPDFLoader("data/sample.pdf")
pdf_docs = pdf_loader.load()

print(f"Total pages loaded: {len(pdf_docs)}")
print(f"First page content preview:\n{pdf_docs[0].page_content[:300]}")
print(f"Metadata: {pdf_docs[0].metadata}")

# ----- Loading a Text File -----
text_loader = TextLoader("data/sample.txt")
text_docs = text_loader.load()

print(f"\nText file content preview:\n{text_docs[0].page_content[:300]}")