"""
Embedding module using HuggingFace (local model)
"""

import logging
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


@st.cache_resource
def load_embeddings():
    """
    Load HuggingFace embedding model (cached)

    Returns:
        HuggingFaceEmbeddings instance
    """
    try:
        logger.info("Loading HuggingFace embeddings: all-MiniLM-L6-v2")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        logger.info("Embeddings loaded successfully")
        return embeddings

    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        raise Exception(f"Failed to load embeddings: {str(e)}")


def get_embeddings():
    """
    Get cached embeddings instance
    """
    return load_embeddings()