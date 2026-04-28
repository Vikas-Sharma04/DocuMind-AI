import logging
from typing import List, Optional
from langchain_core.documents import Document
from config import TOP_K_RESULTS, RETRIEVAL_TYPE

logger = logging.getLogger(__name__)


def get_retriever(vectorstore, top_k: Optional[int] = None):
    """
    Safe retriever for Chroma + LangChain (FIXED VERSION)
    """

    if vectorstore is None:
        raise ValueError("Vectorstore is None")

    k = top_k or TOP_K_RESULTS

    logger.info(f"Creating retriever | type={RETRIEVAL_TYPE} | k={k}")

    # ✅ SAFE: similarity (works in all Chroma versions)
    if RETRIEVAL_TYPE == "similarity" or RETRIEVAL_TYPE == "mmr":

        # IMPORTANT:
        # Do NOT pass fetch_k or lambda_mult manually if your stack is unstable
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    logger.info("Retriever created successfully")
    return retriever


def retrieve_documents(
    retriever,
    query: str,
    top_k: Optional[int] = None
) -> List[Document]:

    if not retriever:
        raise ValueError("Retriever is None")

    if not query or not query.strip():
        raise ValueError("Empty query")

    query = query.strip()

    logger.info(f"Retrieving docs | query={query[:50]}")

    docs = retriever.invoke(query)

    # clean results
    filtered_docs = [
        doc for doc in docs
        if doc.page_content and len(doc.page_content.strip()) > 20
    ]

    logger.info(f"Retrieved {len(filtered_docs)} docs")

    return filtered_docs


def get_retrieval_metadata(documents: List[Document]) -> list:

    return [
        {
            "rank": i + 1,
            "page": doc.metadata.get("page", "Unknown"),
            "source": doc.metadata.get("source", "Unknown"),
            "chunk_size": len(doc.page_content),
        }
        for i, doc in enumerate(documents)
    ]