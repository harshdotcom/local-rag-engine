# app/core/advanced_rag.py

from app.core.hybrid_search import get_hybrid_retriever
from app.core.reranker import retrieve_and_rerank, set_retriever_k
from app.core.compression import get_compression_retriever
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config import settings

class AdvancedRAGPipeline:

    def __init__(self, chunks, collection_name="default"):
        print("Building Advanced RAG Pipeline...")

        # Step 1: Hybrid retriever (semantic + keyword)
        self.hybrid_retriever = get_hybrid_retriever(
            chunks, collection_name
        )
        print("   Hybrid retriever ready")

        # Step 2: LLM
        self.llm = OllamaLLM(model=settings.LLM_MODEL, temperature=0)
        print("   LLM ready")

        # Step 3: Prompt
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a precise assistant. Answer ONLY 
from the context. Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""
        )
        print("   Advanced RAG Pipeline ready!\n")

    def query(self, question: str) -> dict:

        # Stage 1: Configure hybrid retrieval to fetch enough candidates
        set_retriever_k(self.hybrid_retriever, 10)

        # Stage 2: Re-rank (pick best 3)
        reranked_docs = retrieve_and_rerank(question, self.hybrid_retriever)

        # Stage 3: Format context
        context = "\n\n".join(doc.page_content for doc in reranked_docs)

        # Stage 4: Generate answer
        chain = self.prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        # Stage 5: Collect sources
        sources = list(set([
            doc.metadata.get("source", "unknown")
            for doc in reranked_docs
        ]))

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(reranked_docs),
            "rerank_scores": [
                doc.metadata.get("rerank_score", 0)
                for doc in reranked_docs
            ]
        }
