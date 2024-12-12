import logging
import os
from typing import List, Optional

import tiktoken
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import add_messages
from qdrant_client import QdrantClient
from typing_extensions import Annotated, TypedDict

from .prompts import agent_llm_prompt, get_prompt_template, response_llm_prompt
from .vector_db import ChromaAdapter, QdrantAdapter

load_dotenv()

logger = logging.getLogger("app_ctx")
logger.setLevel(logging.DEBUG)


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
        self.agent_model_name: str = os.environ.get('OPENAI_AGENT_MODEL', "gpt-4o-mini")
        self.response_model_name: str = os.environ.get('OPENAI_RESPONSE_MODEL', "gpt-4o")
        self.response_model_penalty: float = float(os.environ.get('RESPONSE_PENALTY', 0.0))
        self.response_model_temperature: float = float(os.environ.get('RESPONSE_TEMPERATURE', 1.0))
        self.qdrant_embeddings_model = os.environ.get('QDRANT_EMBEDDINGS', "text-embedding-3-small")
        self.qdrant_url: str = os.getenv("QDRANT_URL")
        self.qdrant_api_key: str = os.getenv("QDRANT_API_KEY")
        self.qdrant_wisdom_collection: str = os.getenv("QDRANT_WISDOM_COLLECTION")
        self.recall_collection: str = os.getenv("QDRANT_MEMORY_COLLECTION", "memories")
        self.core_collection: str = os.getenv("QDRANT_MEMORY_COLLECTION", "core_memories")


# Schemas
class GraphConfig(TypedDict):
    thread_id: str  # The thread ID of the conversation.
    user_id: str  # The ID of the user to remember in the conversation.


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # The messages in the conversation.
    core_memories: List[str]  # The core memories associated with the user.
    recall_memories: List[str]  # The recall memories retrieved for the current context.
    final_response: Optional[str]  # Response from the final LLM.
    to_summarize: List[BaseMessage]  # New field for tracking messages to summarize


class AppCtx:
    def __init__(self):
        self.settings = Settings()
        self.constants = Constants()

        self.qdrant_client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key
        )
        self.qdrant_vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.settings.qdrant_wisdom_collection,
            embedding=OpenAIEmbeddings(),
        )
        self.qdrant_memory_embeddings = OpenAIEmbeddings(model=self.settings.qdrant_embeddings_model)
        self.recall_memory_adapter = QdrantAdapter(
            client=self.qdrant_client,
            collection_name=self.settings.recall_collection
        )
        self.core_memory_adapter = ChromaAdapter(collection_name=self.settings.core_collection)
        self.agent_prompt = get_prompt_template(agent_llm_prompt)
        self.agent_model = ChatOpenAI(
            model_name=self.settings.agent_model_name,
            temperature=0,
            streaming=True
        )
        self.response_prompt = get_prompt_template(response_llm_prompt)
        self.response_model = ChatOpenAI(
            model_name=self.settings.response_model_name,
            temperature=self.settings.response_model_temperature,
            max_tokens=256,
            timeout=45,
            streaming=True,
            frequency_penalty=self.settings.response_model_penalty,
            presence_penalty=self.settings.response_model_penalty,
        )
        self.tokenizer = tiktoken.encoding_for_model(self.settings.agent_model_name)

    @staticmethod
    def ensure_configurable(config: RunnableConfig) -> GraphConfig:
        """Merge the user-provided config with default values."""
        configurable = config.get("configurable", {})
        return GraphConfig(
            thread_id=configurable["thread_id"],
            user_id=configurable["user_id"],
        )


ctx = AppCtx()
