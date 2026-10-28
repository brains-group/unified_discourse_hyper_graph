"""
Configuration file for API keys and other settings.

[unh_v1] Upstream hard-codes OpenAI + all-MiniLM-L6-v2. Every value below is now read from
the environment, with the upstream value noted next to it. See ../run.sh for the values
used in our runs (DeepSeek + voyageai/voyage-4-nano).
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# OpenAI-compatible API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
# upstream: no base_url (api.openai.com)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

# API Rate Limiting Configuration
CALLS_PER_MINUTE = int(os.environ.get("LOGICRAG_CALLS_PER_MINUTE", "20"))  # upstream: 20
PERIOD = 60
MAX_RETRIES = 3
RETRY_DELAY = 120

# Model Configuration
DEFAULT_MODEL = os.environ.get("LOGICRAG_LLM_MODEL", "gpt-4o-mini")  # upstream: "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 250

# Embedding Configuration
# upstream: "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL = os.environ.get("LOGICRAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = 32

# Cache Configuration (upstream: relative "cache" / "result", i.e. dependent on the cwd)
CACHE_DIR = os.environ.get("LOGICRAG_CACHE_DIR", os.path.join(PROJECT_ROOT, "cache"))
RESULT_DIR = os.environ.get("LOGICRAG_RESULT_DIR", os.path.join(PROJECT_ROOT, "result"))
