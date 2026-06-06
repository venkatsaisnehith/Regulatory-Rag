"""
Project Configuration (config.py)
Defines model selections, file paths, database connection configurations,
and Hugging Face API keys.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- General System Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_REG_DIR = os.path.join(DATA_DIR, "sample_regulations")

# Ensure folders exist
for folder in [DATA_DIR, SAMPLE_REG_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- Model Selection Config ---
# We use Hugging Face Serverless API by default for generation (LLM)
# and local lightweight models for Embeddings and Reranking.
LLM_PROVIDER = "huggingface"  # Options: "huggingface", "openai", "azure"

# Hugging Face Settings
# Mistral-7B or Llama-3-8B are highly reliable serverless options
HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# Local Embedding Model (SentenceTransformers)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # ~120MB model size

# Local Reranking Model (Cross-Encoder)
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"  # ~200MB model size

# --- Qdrant Vector Database Config ---
# Qdrant's client supports running in-memory without Docker, 
# which keeps the system lightweight and portable.
QDRANT_LOCATION = ":memory:"  # Options: ":memory:" (in-memory RAM), or "http://localhost:6333"
QDRANT_COLLECTION_NAME = "regulatory_docs"

# --- Security and Prompt Settings ---
SYSTEM_COMPLIANCE_PROMPT = (
    "You are an expert regulatory compliance auditor for the Luxembourg financial sector.\n"
    "Your task is to answer the compliance query accurately using ONLY the provided regulatory context.\n"
    "Follow these strict guardrail rules:\n"
    "1. Quote the exact article or section number of the regulation where applicable.\n"
    "2. If the answer cannot be verified by the provided context chunks, state explicitly "
    "'I cannot verify this query using the provided regulation documents.' Do not guess or hallucinate."
)
