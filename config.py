"""
Configuration settings for AskMyPDF (Hybrid RAG optimized)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ==================== LLM CONFIGURATION ====================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mistral")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-small-2506")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ==================== EMBEDDING CONFIGURATION ====================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==================== CHUNKING CONFIGURATION ====================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ==================== RETRIEVAL CONFIGURATION ====================
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "12"))
RETRIEVAL_TYPE = os.getenv("RETRIEVAL_TYPE", "hybrid")

# keyword boost (important for "Experiment 7")
ENABLE_KEYWORD_BOOST = True

# ==================== API KEYS ====================
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# ==================== UI ====================
APP_NAME = "AskMyPDF"
APP_DESCRIPTION = "Chat with your PDFs using AI"

MAX_FILE_SIZE_MB = 25
ALLOWED_FILE_TYPES = ["pdf"]

# ==================== SAFETY ====================
MAX_PDF_PAGES = 200
MAX_CHUNKS = 2000