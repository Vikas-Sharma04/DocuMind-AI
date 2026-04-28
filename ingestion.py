import logging
import os
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf(
    file_path: str,
    max_pages: Optional[int] = None
) -> List[Document]:
    """
    Load a PDF file and extract documents

    Args:
        file_path: Path to the PDF file
        max_pages: Optional limit on number of pages to load

    Returns:
        List of Document objects with non-empty content
    """
    try:
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        logger.info(f"Loading PDF: {file_path}")

        loader = PyPDFLoader(file_path)

        # ✅ Lazy load (better for large PDFs)
        docs = []
        for i, doc in enumerate(loader.lazy_load()):
            if max_pages and i >= max_pages:
                logger.info(f"Stopped loading at {max_pages} pages")
                break

            if doc.page_content and doc.page_content.strip():
                docs.append(doc)

        if not docs:
            raise ValueError("PDF contains no readable text")

        logger.info(f"Loaded {len(docs)} valid pages")
        return docs

    except Exception as e:
        logger.error(f"PDF loading error: {str(e)}")
        raise Exception(f"Failed to load PDF: {str(e)}")


def validate_pdf(file_path: str) -> bool:
    """
    Lightweight PDF validation

    Args:
        file_path: Path to file

    Returns:
        True if valid PDF, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            return False

        # ✅ Fast check using file signature
        with open(file_path, "rb") as f:
            header = f.read(5)

        return header == b"%PDF-"

    except Exception as e:
        logger.error(f"PDF validation failed: {str(e)}")
        return False