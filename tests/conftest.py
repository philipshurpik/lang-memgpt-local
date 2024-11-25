import os
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture(scope="session", autouse=True)
def mock_chroma_client():
    with patch("memory_langgraph.app_ctx.vectordb_client") as mock_client:
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        yield mock_client

@pytest.fixture(scope="session", autouse=True)
def set_fake_env_vars():
    os.environ["CHROMA_PERSIST_DIRECTORY"] = "./test_chroma_db"
    yield
