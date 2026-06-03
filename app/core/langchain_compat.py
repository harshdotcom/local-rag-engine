"""Compatibility imports for LangChain packages.

LangChain 1.x moved many retrievers from ``langchain`` into
``langchain_classic``. Keeping these imports in one place avoids scattering
version-specific imports across the RAG modules.
"""

from langchain_core.documents import Document

try:
    from langchain_classic.retrievers import (
        ContextualCompressionRetriever,
        EnsembleRetriever,
        ParentDocumentRetriever,
    )
    from langchain_classic.retrievers.document_compressors import LLMChainExtractor
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever
    from langchain_classic.storage import InMemoryStore
except ImportError:
    from langchain.retrievers import (  # type: ignore[no-redef]
        ContextualCompressionRetriever,
        EnsembleRetriever,
        ParentDocumentRetriever,
    )
    from langchain.retrievers.document_compressors import LLMChainExtractor  # type: ignore[no-redef]
    from langchain.retrievers.multi_query import MultiQueryRetriever  # type: ignore[no-redef]
    from langchain.storage import InMemoryStore  # type: ignore[no-redef]
