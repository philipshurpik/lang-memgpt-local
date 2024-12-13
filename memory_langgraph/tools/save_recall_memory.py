import uuid
from datetime import UTC, datetime

from langchain_core.runnables.config import ensure_config
from langchain_core.tools import tool

from ..app_ctx import ctx


@tool
async def save_recall_memory(memory: str) -> str:
    """Save a contextual memory for later retrieval.

    Args:
        memory: The memory text to save.

    Returns:
        The saved memory text.
    """
    config = ensure_config()
    user_id = ctx.ensure_configurable(config)["user_id"]
    vector = await ctx.qdrant_memory_embeddings.aembed_query(memory)

    metadata = {
        "user_id": user_id,
        "type": "recall",
        "timestamp": datetime.now(UTC).isoformat(),
    }

    await ctx.recall_memory_adapter.add_memory(
        id=str(uuid.uuid4()),
        vector=vector,
        metadata=metadata,
        content=memory
    )
    return memory
