import logging
import os
from functools import lru_cache
from importlib import import_module
from typing import Any, Dict, List, Optional

import tiktoken
from dotenv import load_dotenv
from langchain import hub
from langchain_core.messages import AnyMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import add_messages
from typing_extensions import Annotated, TypedDict

load_dotenv()


class Constants:
    PAYLOAD_KEY = "content"
    PATH_KEY = "path"
    PATCH_PATH = "user/{user_id}/core"
    INSERT_PATH = "user/{user_id}/recall/{event_id}"
    TIMESTAMP_KEY = "timestamp"
    TYPE_KEY = "type"


# Settings
class Settings:
    def __init__(self):
        self.vectordb_class: str = "memory_langgraph.vector_db.chroma.ChromaAdapter"
        self.vectordb_config: Dict[str, Any] = {"persist_directory": "./vectordb"}
        self.model: str = "gpt-4o-mini"


# Schemas
class GraphConfig(TypedDict):
    model: str
    """The model to use for the memory assistant."""
    thread_id: str
    """The thread ID of the conversation."""
    user_id: str
    """The ID of the user to remember in the conversation."""


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    """The messages in the conversation."""
    core_memories: List[str]
    """The core memories associated with the user."""
    recall_memories: List[str]
    """The recall memories retrieved for the current context."""
    final_response: Optional[str]
    """Response from the final LLM."""


# AppCtx class
class AppCtx:
    def __init__(self):
        self.logger = logging.getLogger("memory")
        self.logger.setLevel(logging.DEBUG)

        # Initialize settings and constants
        self.settings = Settings()
        self.constants = Constants()

        # Initialize prompts
        self.prompts = {
            "agent": hub.pull("langgraph-agent"),
            "response": hub.pull("langgraph-response"),
        }

        # Initialize models
        self.agent_model = self.init_agent_model(self.settings.model)
        self.response_model = self.init_response_model()

        # Initialize vector DB client
        self.vectordb_client = self.get_vectordb_client()

    @lru_cache()
    def get_vectordb_client(self):
        module_name, class_name = self.settings.vectordb_class.rsplit('.', 1)
        module = import_module(module_name)
        VectorDBClass = getattr(module, class_name)
        return VectorDBClass(**self.settings.vectordb_config)

    @lru_cache()
    def get_embeddings(self):
        return OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

    def init_agent_model(self, model_name: str = None):
        """Initialize the agent model."""
        return ChatOpenAI(
            model_name=model_name or self.settings.model,
            temperature=0.7,  # Adjust as needed
            streaming=True
        )

    def init_response_model(self):
        """Initialize the response model."""
        return ChatOpenAI(
            model_name=os.environ.get('OPENAI_RESPONSE_MODEL', 'gpt-4o'),
            temperature=1.0,
            max_tokens=256,
            timeout=45,
            streaming=True,
            frequency_penalty=float(os.environ.get('OPENAI_PENALTY', 0.0)),
            presence_penalty=float(os.environ.get('OPENAI_PENALTY', 0.0)),
        )

    def ensure_configurable(self, config: RunnableConfig) -> GraphConfig:
        """Merge the user-provided config with default values."""
        configurable = config.get("configurable", {})
        return GraphConfig(
            model=configurable.get("model", self.settings.model),
            thread_id=configurable["thread_id"],
            user_id=configurable["user_id"],
        )

    def get_tokenizer(self):
        return tiktoken.encoding_for_model(self.settings.model)


ctx = AppCtx()
