import logging
from typing import List

from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import RunnableConfig

from ..app_ctx import State, ctx
from ..tools import search_recall_memory

logger = logging.getLogger("memory")
logger.setLevel(logging.DEBUG)


def get_recent_user_messages(messages: List[BaseMessage], n: int = 3) -> str:
    """Get the last n user messages."""
    user_messages = [msg.content for msg in messages if msg.type == "human"]
    return " ".join(user_messages[-n:])


async def load_memories(state: State, config: RunnableConfig) -> State:
    """Load core and recall memories for the current conversation."""
    configurable = ctx.ensure_configurable(config)
    user_id = configurable["user_id"]

    core_memories = await ctx.core_memory_adapter.get_memories(user_id)
    search_context = get_recent_user_messages(state["messages"])
    recall_memories = await search_recall_memory.ainvoke({"query": search_context})

    return {
        "messages": state["messages"],
        "core_memories": core_memories,
        "recall_memories": recall_memories,
        "final_response": None,
        "to_summarize": [],
    }
