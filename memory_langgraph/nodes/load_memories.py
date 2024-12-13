import logging
from langchain_core.messages.utils import get_buffer_string
from langchain_core.runnables.config import RunnableConfig, get_executor_for_config

from ..app_ctx import State, ctx
from ..tools import load_core_memories, search_recall_memory

logger = logging.getLogger("memory")
logger.setLevel(logging.DEBUG)


async def load_memories(state: State, config: RunnableConfig) -> State:
    """Load core and recall memories for the current conversation."""
    configurable = ctx.ensure_configurable(config)
    user_id = configurable["user_id"]
    convo_str = get_buffer_string(state["messages"])
    convo_str = ctx.tokenizer.decode(ctx.tokenizer.encode(convo_str)[:2048])

    core_memories = await load_core_memories(user_id)
    with get_executor_for_config(config) as executor:
        recall_memories_future = executor.submit(search_recall_memory, convo_str)  # search_recall_memory.invoke ??
        recall_memories = recall_memories_future.result()

    return {
        "messages": state["messages"],
        "core_memories": core_memories,
        "recall_memories": recall_memories,
        "final_response": None,
        "to_summarize": [],
    }