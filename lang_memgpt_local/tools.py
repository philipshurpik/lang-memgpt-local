import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple, List

import langsmith
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import tool

from lang_memgpt_local import _constants as constants
from lang_memgpt_local import _utils as utils

logger = logging.getLogger("memory")
logger.setLevel(logging.INFO)

# Initialize the search tool for external information retrieval
search_wrapper = TavilySearchResults(
    max_results=5,  # Optional: Configure number of results
    include_raw_content=True,  # Optional: Include raw content
    include_images=False,  # Optional: Don't include images
    search_depth="advanced"  # Optional: Use advanced search
)

# Initialize the database adapter
db_adapter = utils.get_vectordb_client()


@tool
async def save_recall_memory(memory: str) -> str:
    """Save a contextual memory to the database for later semantic retrieval.

    Args:
        memory (str): The memory to be saved.

    Returns:
        str: The saved memory.
    """
    config = ensure_config()
    configurable = utils.ensure_configurable(config)
    embeddings = utils.get_embeddings()
    vector = await embeddings.aembed_query(memory)

    current_time = datetime.now(tz=timezone.utc)
    event_id = str(uuid.uuid4())
    path = constants.INSERT_PATH.format(
        user_id=configurable["user_id"],
        event_id=event_id,
    )

    metadata = {
        constants.PAYLOAD_KEY: memory,
        constants.PATH_KEY: path,
        constants.TIMESTAMP_KEY: current_time.isoformat(),
        constants.TYPE_KEY: "recall",
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
        configurable = utils.ensure_configurable(config)
        embeddings = utils.get_embeddings()
        vector = embeddings.embed_query(query)

        where_clause = {
            "$and": [
                {"user_id": {"$eq": configurable["user_id"]}},
                {constants.TYPE_KEY: {"$eq": "recall"}}
            ]
        }

        results = db_adapter.query_memories(vector, where_clause, top_k)
        return [x[constants.PAYLOAD_KEY] for x in results]

    except Exception as e:
        logger.error(f"Error in search_memory: {str(e)}")
        return []


@langsmith.traceable
def fetch_core_memories(user_id: str) -> Tuple[str, list[str]]:
    """Fetch core memories for a specific user.

    Args:
        user_id (str): The ID of the user.

    Returns:
        Tuple[str, list[str]]: The path and list of core memories.
    """
    path = constants.PATCH_PATH.format(user_id=user_id)
    collection = db_adapter.get_collection("core_memories")
    results = collection.get(ids=[path], include=["metadatas"])

    memories = []
    if results and results['metadatas']:
        payload = results['metadatas'][0][constants.PAYLOAD_KEY]
        memories = json.loads(payload)["memories"]
    return path, memories


@tool
def store_core_memory(memory: str, index: Optional[int] = None) -> str:
    """Store a core memory about user, like his name or basic preferences in the database.

    Args:
        memory (str): The memory to store.
        index (Optional[int]): The index at which to store the memory.

    Returns:
        str: A confirmation message.
    """
    config = ensure_config()
    configurable = utils.ensure_configurable(config)
    path, existing_memories = fetch_core_memories(configurable["user_id"])

    if index is not None:
        if index < 0 or index >= len(existing_memories):
            return "Error: Index out of bounds."
        existing_memories[index] = memory
    else:
        if memory not in existing_memories:
            existing_memories.insert(0, memory)

    db_adapter.upsert(
        "core_memories",
        [path],
        [{
            constants.PAYLOAD_KEY: json.dumps({"memories": existing_memories}),
            constants.PATH_KEY: path,
            constants.TIMESTAMP_KEY: datetime.now(tz=timezone.utc).isoformat(),
            constants.TYPE_KEY: "core",
            "user_id": configurable["user_id"],
        }],
        [json.dumps({"memories": existing_memories})]
    )
    return "Memory stored."
