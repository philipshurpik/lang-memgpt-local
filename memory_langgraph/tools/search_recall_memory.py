import logging
from typing import List

from langchain_core.runnables.config import ensure_config
from langchain_core.tools import tool

from ..app_ctx import ctx

logger = logging.getLogger("tools")
logger.setLevel(logging.INFO)


@tool
async def search_recall_memory(query: str, top_k: int = 5) -> List[str]:
    """Search for semantically similar memories.

    Args:
        query: The search query text.
        top_k: Number of results to return (default: 5).

    Returns:
        List of relevant memory texts.
    """
    try:
        config = ensure_config()
        user_id = ctx.ensure_configurable(config)["user_id"]
        vector = await ctx.qdrant_memory_embeddings.aembed_query(query)

        results = await ctx.recall_memory_adapter.query_memories(
            vector=vector,
            user_id=user_id,
            n_results=top_k
        )
        return [result["content"] for result in results]

    except Exception as e:
        logger.error(f"Error in search_memory: {str(e)}")
        return []
