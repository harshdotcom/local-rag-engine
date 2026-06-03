# app/core/compression.py

from langchain_ollama import OllamaLLM
from app.core.langchain_compat import ContextualCompressionRetriever, LLMChainExtractor
from app.config import settings

def get_compression_retriever(base_retriever):
    """
    Extracts only the relevant parts from each retrieved chunk
    """

    llm = OllamaLLM(model=settings.LLM_MODEL, temperature=0)

    # Compressor uses LLM to extract relevant sentences
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    return compression_retriever

# BEFORE compression: 500 chars sent to LLM
# AFTER compression:  50 chars sent to LLM ✅
# Result: Faster + more accurate + saves tokens
