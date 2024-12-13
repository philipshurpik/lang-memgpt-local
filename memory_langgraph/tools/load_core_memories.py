import logging
from typing import Dict

import langsmith

from ..app_ctx import ctx

logger = logging.getLogger("tools")
logger.setLevel(logging.INFO)


@langsmith.traceable
async def load_core_memories(user_id: str) -> Dict[str, str]:
    """Get user's memories."""
    return await ctx.core_memory_adapter.get_memories(user_id)
