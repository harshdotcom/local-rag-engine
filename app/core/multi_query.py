# app/core/multi_query.py

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from app.core.langchain_compat import MultiQueryRetriever
from app.config import settings

def get_multi_query_retriever(vectorstore):
    """
    Automatically generates multiple query variations
    and combines their results
    """

    llm = OllamaLLM(model=settings.LLM_MODEL, temperature=0.3)
    # temperature=0.3 adds slight creativity for varied questions

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        llm=llm
    )

    return retriever


# ── WHAT HAPPENS INTERNALLY ─────────────────────────────────
# User: "How do I cancel my subscription?"
#
# LLM generates 3 variations:
#   1. "What is the process to cancel my account?"
#   2. "How to terminate membership?"
#   3. "Steps to discontinue service?"
#
# Retrieves chunks for ALL 3 queries
# Deduplicates results
# Returns unique, combined results ✅
