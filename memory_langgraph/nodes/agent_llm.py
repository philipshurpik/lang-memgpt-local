import logging
from datetime import datetime, timezone

from langchain_core.runnables.config import RunnableConfig

from ..tools import ask_wisdom, save_recall_memory, search_recall_memory, search_tavily, save_core_memories
from ..app_ctx import ctx, State

logger = logging.getLogger("graph_nodes")
logger.setLevel(logging.DEBUG)
all_tools = [ask_wisdom, save_core_memories, save_recall_memory, search_recall_memory, search_tavily]


async def agent_llm(state: State, config: RunnableConfig) -> State:
    """Process the current state and generate a response using the LLM."""
    llm = ctx.agent_model.bind_tools(all_tools, tool_choice="auto")
    bound = ctx.prompts["agent"] | llm

    core_str = "<core_memory>\n" + "\n".join(
        [f"{k}: {v}" for k, v in state["core_memories"].items()]) + "\n</core_memory>"
    recall_str = "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    logger.debug(f"Core memories: {core_str}")
    logger.debug(f"Recall memories: {recall_str}")

    response = await bound.ainvoke({
        "messages": state["messages"],
        "core_memories": core_str,
        "recall_memories": recall_str,
        "current_time": datetime.now(tz=timezone.utc).isoformat(),
    })
    return {
        "messages": response,
        "core_memories": state["core_memories"],
        "recall_memories": state["recall_memories"],
        "final_response": None,
    }
