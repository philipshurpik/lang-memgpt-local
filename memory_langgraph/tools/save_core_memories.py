import json
from datetime import datetime, timezone

from langchain_core.runnables.config import ensure_config
from langchain_core.tools import tool

from .load_core_memories import load_core_memories
from ..app_ctx import ctx, Constants


@tool
def save_core_memories(key: str, value: str) -> str:
    """Store a core memory about the user in key-value format.

    Args:
        key (str): The key/type of the memory (e.g., "name", "age").
        value (str): The value to store.

    Returns:
        str: A confirmation message.
    """
    config = ensure_config()
    configurable = ctx.ensure_configurable(config)
    path, existing_memories = load_core_memories(configurable["user_id"])

    existing_memories[key] = value

    ctx.core_memory_adapter.upsert(
        ctx.settings.core_collection,
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
