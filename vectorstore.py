import logging
import time
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)


def create_vectorstore(
    chunks: List[Document],
    embeddings
) -> Chroma:
    """
    Create an in-memory vector database from chunks

    Args:
        chunks: List of chunked documents
        embeddings: Embeddings instance

    Returns:
        Chroma vectorstore instance

    Raises:
        Exception: If vectorstore creation fails
    """
    try:
        logger.info("Creating in-memory vectorstore")

        # Validate chunks
        if not chunks:
            raise ValueError("No chunks provided to create vectorstore")

        # Filter out empty chunks
        valid_chunks = [
            chunk for chunk in chunks
            if chunk.page_content and chunk.page_content.strip()
        ]

        if not valid_chunks:
            raise ValueError("All chunks are empty after filtering")

        logger.info(
            f"Creating vectorstore with {len(valid_chunks)} valid chunks "
            f"(filtered from {len(chunks)})"
        )

        # ✅ In-memory vectorstore (NO persist_directory)
        vectorstore = Chroma.from_documents(
            documents=valid_chunks,
            embedding=embeddings,
            collection_name=f"col_{int(time.time())}" # Needs 'import time'
        )

        logger.info("Vectorstore created successfully (in-memory)")
        return vectorstore

    except Exception as e:
        logger.error(f"Error creating vectorstore: {str(e)}")
        raise Exception(f"Failed to create vectorstore: {str(e)}")


def delete_vectorstore(vectorstore: Optional[Chroma]) -> bool:
    """
    Delete vectorstore from memory

    Args:
        vectorstore: Chroma instance

    Returns:
        True if successful
    """
    try:
        if vectorstore is not None:
            # In-memory → just dereference
            del vectorstore
            logger.info("Vectorstore deleted from memory")
        return True

    except Exception as e:
        logger.error(f"Error deleting vectorstore: {str(e)}")
        return False


def vectorstore_exists(vectorstore: Optional[Chroma]) -> bool:
    """
    Check if vectorstore exists in memory

    Args:
        vectorstore: Chroma instance

    Returns:
        True if exists, False otherwise
    """
    return vectorstore is not None