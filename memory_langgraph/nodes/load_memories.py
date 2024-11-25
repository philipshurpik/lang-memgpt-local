import logging

from langchain_core.messages.utils import get_buffer_string
from langchain_core.runnables.config import RunnableConfig, get_executor_for_config

from ..app_ctx import ctx, State
from ..tools import load_core_memories, search_recall_memory

logger = logging.getLogger("memory")
logger.setLevel(logging.DEBUG)


def load_memories(state: State, config: RunnableConfig) -> State:
    """Load core and recall memories for the current conversation."""
    configurable = ctx.ensure_configurable(config)
    user_id = configurable["user_id"]
    convo_str = get_buffer_string(state["messages"])
    convo_str = ctx.tokenizer.decode(ctx.tokenizer.encode(convo_str)[:2048])

    with get_executor_for_config(config) as executor:
        core_memories_future = executor.submit(load_core_memories, user_id)
        recall_memories_future = executor.submit(search_recall_memory, convo_str)  # search_recall_memory.invoke ??

        _, core_memories = core_memories_future.result()
        recall_memories = recall_memories_future.result()

    return {
        "messages": state["messages"],
        "core_memories": core_memories,
        "recall_memories": recall_memories,
        "final_response": None,
    }
