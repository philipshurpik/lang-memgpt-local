import logging
from typing import List

from langchain_core.runnables.config import ensure_config
from langchain_core.tools import tool

from ..app_ctx import ctx, Constants

logger = logging.getLogger("tools")
logger.setLevel(logging.INFO)


@tool
def search_recall_memory(query: str, top_k: int = 5) -> List[str]:
    """Search for memories in the database based on semantic similarity.

    Args:
        query (str): The search query.
        top_k (int): The number of results to return.

    Returns:
        List[str]: A list of relevant memories.
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

        db_adapter = ctx.get_vectordb_client()
        results = db_adapter.query_memories(vector, where_clause, top_k)
        return [x[Constants.PAYLOAD_KEY] for x in results]

    except Exception as e:
        logger.error(f"Error in search_memory: {str(e)}")
        return []
