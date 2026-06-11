import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    AI_BASE_URL = os.getenv("AI_BASE_URL", "http://localhost:11434/v1")
    AI_MODEL_NAME = os.getenv("AI_MODEL_NAME", "phi-4-mini-reasoning")
    DATABASE_PATH = os.getenv("DATABASE_PATH", "lens.db")

config = Config()
