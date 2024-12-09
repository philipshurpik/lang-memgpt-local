import uuid
from datetime import datetime, timezone

from langchain_core.runnables.config import ensure_config
from langchain_core.tools import tool

from ..app_ctx import Constants, ctx


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
    vector = await ctx.qdrant_memory_embeddings.aembed_query(memory)

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

    ctx.recall_memory_adapter.add_memory(event_id, vector, metadata, memory)
    return memory
