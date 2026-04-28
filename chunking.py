"""
Document chunking module (structure-aware for PDFs like experiments)
"""

import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def split_docs(
    docs: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    max_chunks: int = 2000
) -> List[Document]:

    try:
        if not docs:
            raise ValueError("No documents provided")

        logger.info(f"Splitting {len(docs)} pages")

        valid_docs = [
            d for d in docs
            if d.page_content and d.page_content.strip()
        ]

        # 🔥 IMPORTANT: experiment-aware splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\nExperiment ",   # VERY IMPORTANT for your dataset
                "\n\n\n",
                "\n\n",
                "\n",
                ". ",
                " ",
            ]
        )

        raw_chunks = splitter.split_documents(valid_docs)

        chunks = []
        seen = set()

        for i, c in enumerate(raw_chunks):
            text = c.page_content.strip()
            if not text or text in seen:
                continue

            seen.add(text)

            chunks.append(Document(
                page_content=text,
                metadata={
                    **c.metadata,
                    "chunk_id": i,
                    "length": len(text)
                }
            ))

            if len(chunks) >= max_chunks:
                break

        logger.info(f"Created {len(chunks)} chunks")

        return chunks

    except Exception as e:
        logger.error(str(e))
        raise