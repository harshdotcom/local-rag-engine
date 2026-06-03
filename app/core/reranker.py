# app/core/reranker.py

from sentence_transformers import CrossEncoder
from app.core.langchain_compat import Document

# CrossEncoder scores pairs of (query, chunk), which is more accurate than
# embedding similarity alone.
_reranker_model = None


def get_reranker_model():
    global _reranker_model

    if _reranker_model is None:
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return _reranker_model


def set_retriever_k(retriever, k: int) -> None:
    """Set top-k on common retriever types, including ensemble retrievers."""

    if hasattr(retriever, "search_kwargs"):
        retriever.search_kwargs["k"] = k

    if hasattr(retriever, "k"):
        retriever.k = k

    for child_retriever in getattr(retriever, "retrievers", []):
        set_retriever_k(child_retriever, k)


def rerank_documents(query: str, documents: list[Document], top_k: int = 3):
    """
    Re-score retrieved documents based on relevance to the query.

    Returns the top_k most relevant documents.
    """

    if not documents:
        return []

    pairs = [(query, doc.page_content) for doc in documents]
    scores = get_reranker_model().predict(pairs)

    scored_docs = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True,
    )

    reranked = []
    for score, doc in scored_docs[:top_k]:
        doc.metadata["rerank_score"] = round(float(score), 4)
        doc.metadata["confidence"] = (
            "High" if score > 0 else
            "Medium" if score > -5 else
            "Low"
        )
        reranked.append(doc)
    
    return reranked


def retrieve_and_rerank(query: str, retriever, top_k: int = 3):
    """
    Retrieve a larger candidate set, rerank it, and return the best documents.
    """

    set_retriever_k(retriever, 10)
    raw_docs = retriever.invoke(query)
    reranked_docs = rerank_documents(query, raw_docs, top_k=top_k)

    print(f"\nRe-ranking results for: '{query}'")
    for i, doc in enumerate(reranked_docs):
        print(f"  #{i + 1} Score: {doc.metadata['rerank_score']} -> {doc.page_content[:80]}...")

    return reranked_docs


