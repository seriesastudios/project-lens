import os
import pathlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent


def _resolve_db_path() -> str:
    """A relative DATABASE_PATH is anchored to the repo root (not the process
    cwd) so the same DB is used however the app is launched. Absolute paths and
    explicit env overrides pass through untouched."""
    raw = os.getenv("DATABASE_PATH", "lens.db")
    path = pathlib.Path(raw)
    return str(path if path.is_absolute() else BASE_DIR / path)


class Config:
    AI_BASE_URL = os.getenv("AI_BASE_URL", "http://127.0.0.1:1234/v1")  # LM Studio default port
    AI_MODEL_NAME = os.getenv("AI_MODEL_NAME", "qwen/qwen3-4b-2507")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-nomic-embed-text-v1.5")
    DATABASE_PATH = _resolve_db_path()

config = Config()
