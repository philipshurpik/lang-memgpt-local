import json
import logging
from typing import Dict, Tuple

import langsmith

from ..app_ctx import ctx, Constants

logger = logging.getLogger("tools")
logger.setLevel(logging.INFO)


@langsmith.traceable
def load_core_memories(user_id: str) -> Tuple[str, Dict[str, str]]:
    """Fetch core memories for a specific user.

    Args:
        user_id (str): The ID of the user.

    Returns:
        Tuple[str, Dict[str, str]]: The path and dictionary of core memories.
    """
    path = Constants.PATCH_PATH.format(user_id=user_id)
    collection = ctx.core_memory_adapter.get_collection(ctx.settings.core_collection)
    results = collection.get(ids=[path], include=["metadatas"])

    memories = {}
    if results and results['metadatas']:
        payload = results['metadatas'][0].get(Constants.PAYLOAD_KEY)
        if payload:
            memories = json.loads(payload).get("memories", {})
    return path, memories
