# app/core/conversational_rag.py

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.config import settings

# ── PROMPT WITH HISTORY ─────────────────────────────────────
CONVERSATIONAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer based ONLY 
     on the context below. If not in context, say 'I don't know.'
     
     Context: {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),  # ← history goes here
    ("human", "{question}"),
])

class ConversationalRAG:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = OllamaLLM(model=settings.LLM_MODEL, temperature=0)
        self.chat_history = []   # stores conversation

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def ask(self, question: str) -> str:
        # Retrieve relevant chunks
        docs = self.retriever.invoke(question)
        context = self.format_docs(docs)

        # Build chain
        chain = CONVERSATIONAL_PROMPT | self.llm | StrOutputParser()

        # Invoke with history
        response = chain.invoke({
            "context": context,
            "chat_history": self.chat_history,  # ← pass full history
            "question": question
        })

        # Update history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response))

        return response

    def clear_history(self):
        self.chat_history = []
        print("Chat history cleared!")


# ── USAGE ───────────────────────────────────────────────────
# rag = ConversationalRAG(retriever)
#
# rag.ask("Who is the CEO?")
# → "The CEO is John Smith"
#
# rag.ask("How old is he?")
# → "John Smith is 45 years old"  ✅ remembers context!
#
# rag.ask("What is his educational background?")
# → "He studied at MIT..."  ✅ still remembers!