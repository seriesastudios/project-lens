import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    AI_BASE_URL = os.getenv("AI_BASE_URL", "http://127.0.0.1:1234/v1")  # LM Studio default port
    AI_MODEL_NAME = os.getenv("AI_MODEL_NAME", "qwen/qwen3-4b-2507")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-nomic-embed-text-v1.5")
    DATABASE_PATH = os.getenv("DATABASE_PATH", "lens.db")

config = Config()
