import pytest

from app.config import config
from app.database import models


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point every test at a fresh throwaway database."""
    db_path = str(tmp_path / "test_lens.db")
    monkeypatch.setattr(config, "DATABASE_PATH", db_path)
    models.init_db()
    yield db_path
