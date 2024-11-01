from __future__ import annotations

import os

from dotenv import load_dotenv
from functools import lru_cache
import langsmith
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from importlib import import_module
from lang_memgpt_local import _schemas as schemas
from lang_memgpt_local import _settings as settings

_DEFAULT_DELAY = 20  # seconds

load_dotenv()

@lru_cache
def get_vectordb_client():
    module_name, class_name = settings.SETTINGS.vectordb_class.rsplit('.', 1)
    module = import_module(module_name)
    VectorDBClass = getattr(module, class_name)
    return VectorDBClass(**settings.SETTINGS.vectordb_config)

# Other utility functions...

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
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


def init_agent_model(model_name: str = None):
    """Initialize the agent model."""
    return ChatOpenAI(
        model_name=model_name or settings.SETTINGS.model,
        temperature=0.7,  # adjust as needed
        streaming=True
    )


def init_response_model():
    """Initialize the response model."""
    return ChatOpenAI(
        model_name=os.environ['OPENAI_RESPONSE_MODEL'],
        temperature=1.0,
        max_tokens=256,
        timeout=45,
        streaming=True,
        frequency_penalty=os.environ['OPENAI_PENALTY'],
        presence_penalty=os.environ['OPENAI_PENALTY'],
    )


__all__ = ["ensure_configurable"]