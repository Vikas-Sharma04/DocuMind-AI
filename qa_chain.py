import logging
from typing import List

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    MISTRAL_API_KEY
)

logger = logging.getLogger(__name__)


# =========================
# RAG PROMPT (STRICT)
# =========================
RAG_SYSTEM_PROMPT = """You are a precise AI assistant for document question answering.

Rules:
- Use ONLY the provided context.
- If answer is not present, say: "I couldn't find this information in the provided document."
- Do NOT hallucinate.
- Be concise and factual.
"""


# =========================
# CHAT PROMPT
# =========================
CHAT_SYSTEM_PROMPT = """You are a helpful AI assistant.
Give clear, accurate, and concise answers."""


# =========================
# LLM INIT (MISTRAL)
# =========================
def get_llm():
    try:
        if not MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY is missing")

        logger.info(f"Initializing Mistral LLM: {LLM_MODEL}")

        llm = ChatMistralAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            api_key=MISTRAL_API_KEY
        )

        return llm

    except Exception as e:
        logger.error(f"LLM init error: {str(e)}")
        raise


# =========================
# FORMAT DOCS
# =========================
def format_docs(docs: List[Document]) -> str:
    """
    Convert retrieved docs into structured context
    """

    if not docs:
        return "No relevant context found."

    return "\n\n".join(
        f"[Page: {doc.metadata.get('page', 'Unknown')}]\n{doc.page_content}"
        for doc in docs
    )


# =========================
# RAG CHAIN
# =========================
def create_rag_chain(llm, retriever):
    try:
        if llm is None:
            raise ValueError("LLM is None")

        if retriever is None:
            raise ValueError("Retriever is None")

        logger.info("Creating Mistral RAG chain")

        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}")
        ])

        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain

    except Exception as e:
        logger.error(f"RAG chain error: {str(e)}")
        raise


# =========================
# CHAT CHAIN
# =========================
def create_chat_chain(llm):
    try:
        if llm is None:
            raise ValueError("LLM is None")

        logger.info("Creating chat chain")

        prompt = ChatPromptTemplate.from_messages([
            ("system", CHAT_SYSTEM_PROMPT),
            ("human", "{question}")
        ])

        return prompt | llm | StrOutputParser()

    except Exception as e:
        logger.error(f"Chat chain error: {str(e)}")
        raise