import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Tuple, List

import langsmith
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from .app_ctx import ctx, Constants

load_dotenv()
logger = logging.getLogger("memory")
logger.setLevel(logging.INFO)

search_wrapper = TavilySearchResults(
    max_results=5,  # Optional: Configure number of results
    include_raw_content=True,  # Optional: Include raw content
    include_images=False,  # Optional: Don't include images
    search_depth="advanced"  # Optional: Use advanced search
)

db_adapter = ctx.get_vectordb_client()


@tool
async def save_recall_memory(memory: str) -> str:
    """Save a contextual memory to the database for later semantic retrieval.

    Args:
        memory (str): The memory to be saved.

    Returns:
        str: The saved memory.
    """
    config = ensure_config()
    configurable = ctx.ensure_configurable(config)
    embeddings = ctx.get_embeddings()
    vector = await embeddings.aembed_query(memory)

    current_time = datetime.now(tz=timezone.utc)
    event_id = str(uuid.uuid4())
    path = Constants.INSERT_PATH.format(
        user_id=configurable["user_id"],
        event_id=event_id,
    )

    metadata = {
        Constants.PAYLOAD_KEY: memory,
        Constants.PATH_KEY: path,
        Constants.TIMESTAMP_KEY: current_time.isoformat(),
        Constants.TYPE_KEY: "recall",
        "user_id": configurable["user_id"],
    }

    db_adapter.add_memory(event_id, vector, metadata, memory)
    return memory


@tool
def search_tool(query: str) -> str:
    """Search the internet for information about a query.

    Args:
        query (str): The search query

    Returns:
        str: Search results summary
    """
    try:
        results = search_wrapper.run(query)
        return results
    except Exception as e:
        logger.error(f"Error in search_tool: {str(e)}")
        return f"Error performing search: {str(e)}"


@tool
def ask_wisdom(query: str) -> str:
    """Ask wisdom library question that can be answered using a knowledge base of human wisdom, prominent thinkers, psychologists, and philosophers.

    Args:
        query (str): The search question sentence

    Returns:
        str: Search results 
    """
    try:
        qdrant_vectorstore = QdrantVectorStore(
            client=QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY")),
            collection_name=os.getenv("QDRANT_COLLECTION"),
            embedding=OpenAIEmbeddings(),
        )
        results = qdrant_vectorstore.similarity_search(query, k=5)
        formatted_results = "\n\n".join([doc.page_content.strip() for doc in results])
        return formatted_results

    except Exception as e:
        logger.error(f"Error in ask rag db: {str(e)}")
        return f"Error performing ask rag db: {str(e)}"


@tool
def search_memory(query: str, top_k: int = 5) -> List[str]:
    """Search for memories in the database based on semantic similarity.

    Args:
        query (str): The search query.
        top_k (int): The number of results to return.

    Returns:
        list[str]: A list of relevant memories.
    """
    try:
        config = ensure_config()
        configurable = ctx.ensure_configurable(config)
        embeddings = ctx.get_embeddings()
        vector = embeddings.embed_query(query)

        where_clause = {
            "$and": [
                {"user_id": {"$eq": configurable["user_id"]}},
                {Constants.TYPE_KEY: {"$eq": "recall"}}
            ]
        }

        results = db_adapter.query_memories(vector, where_clause, top_k)
        return [x[Constants.PAYLOAD_KEY] for x in results]

    except Exception as e:
        logger.error(f"Error in search_memory: {str(e)}")
        return []


@langsmith.traceable
def fetch_core_memories(user_id: str) -> Tuple[str, dict[str, str]]:
    """Fetch core memories for a specific user.

    Args:
        user_id (str): The ID of the user.

    Returns:
        Tuple[str, dict[str, str]]: The path and dictionary of core memories.
    """
    path = Constants.PATCH_PATH.format(user_id=user_id)
    collection = db_adapter.get_collection("core_memories")
    results = collection.get(ids=[path], include=["metadatas"])

    memories = {}
    if results and results['metadatas']:
        payload = results['metadatas'][0][Constants.PAYLOAD_KEY]
        memories = json.loads(payload)["memories"]
    return path, memories


@tool
def store_core_memory(key: str, value: str) -> str:
    """Store a core memory about user in key-value format.

    Args:
        key (str): The key/type of the memory (e.g., "name", "age", "preference.color")
        value (str): The value to store.

    Returns:
        str: A confirmation message.
    """
    config = ensure_config()
    configurable = ctx.ensure_configurable(config)
    path, existing_memories = fetch_core_memories(configurable["user_id"])

    existing_memories[key] = value

    db_adapter.upsert(
        "core_memories",
        [path],
        [{
            Constants.PAYLOAD_KEY: json.dumps({"memories": existing_memories}),
            Constants.PATH_KEY: path,
            Constants.TIMESTAMP_KEY: datetime.now(tz=timezone.utc).isoformat(),
            Constants.TYPE_KEY: "core",
            "user_id": configurable["user_id"],
        }],
        [json.dumps({"memories": existing_memories})]
    )
    return f"Memory stored with key: {key}"
