from __future__ import annotations

from functools import lru_cache

import langsmith
from langchain_core.runnables import RunnableConfig
from langchain_fireworks import FireworksEmbeddings
import chromadb

from lang_memgpt_local import _schemas as schemas
from lang_memgpt_local import _settings as settings

_DEFAULT_DELAY = 60  # seconds

@lru_cache
def get_chroma_client():
    return chromadb.PersistentClient(path=settings.SETTINGS.chroma_persist_directory)

@langsmith.traceable
def ensure_configurable(config: RunnableConfig) -> schemas.GraphConfig:
    """Merge the user-provided config with default values."""
    configurable = config.get("configurable", {})
    return {
        **configurable,
        **schemas.GraphConfig(
            delay=configurable.get("delay", _DEFAULT_DELAY),
            model=configurable.get("model", settings.SETTINGS.model),
            thread_id=configurable["thread_id"],
            user_id=configurable["user_id"],
        ),
    }

@lru_cache
def get_embeddings():
    return FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")

__all__ = ["ensure_configurable", "get_chroma_client", "get_embeddings"]