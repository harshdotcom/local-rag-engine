# 06_full_rag_pipeline.py

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── CONFIGURATION ──────────────────────────────────────────
DOCS_PATH = "data/sample.pdf"       # change to your file
VECTORSTORE_PATH = "./vectorstore"
LLM_MODEL = "llama3.2"              # must be pulled via ollama

# ── STEP 1: LOAD DOCUMENT ──────────────────────────────────
print("📄 Loading document...")
if DOCS_PATH.endswith(".pdf"):
    loader = PyPDFLoader(DOCS_PATH)
else:
    loader = TextLoader(DOCS_PATH)
docs = loader.load()
print(f"   Loaded {len(docs)} pages")

# ── STEP 2: CHUNK ──────────────────────────────────────────
print("✂️  Chunking document...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)
print(f"   Created {len(chunks)} chunks")

# ── STEP 3: EMBED + STORE ──────────────────────────────────
print("🔢 Creating embeddings and storing in ChromaDB...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=VECTORSTORE_PATH
)
print("   Vectorstore ready!")

# ── STEP 4: RETRIEVER ──────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ── STEP 5: LLM (LOCAL via Ollama) ─────────────────────────
llm = OllamaLLM(model=LLM_MODEL, temperature=0)
# temperature=0 means deterministic answers (less random)

# ── STEP 6: PROMPT TEMPLATE ────────────────────────────────
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Answer ONLY from the context below.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}

Answer:"""
)

# ── STEP 7: BUILD THE CHAIN ────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ── STEP 8: CHAT LOOP ──────────────────────────────────────
print("\n" + "="*50)
print("🤖 RAG Chatbot Ready! (type 'quit' to exit)")
print("="*50 + "\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    if not question:
        continue

    print("Bot: Thinking...", end="\r")
    answer = rag_chain.invoke(question)
    print(f"Bot: {answer}\n")