from datetime import datetime, timezone

from langchain_core.messages.ai import AIMessage
from langchain_core.runnables.config import RunnableConfig

from ..app_ctx import ctx, State


async def response_llm(state: State, config: RunnableConfig) -> State:
    """Final LLM to generate response using memories but no tools."""
    llm = ctx.response_model
    bound = ctx.prompts["response"] | llm

    state_messages = state["messages"]
    if isinstance(state_messages[-1], AIMessage):
        state_messages = state_messages[:-1]

    response = await bound.ainvoke({
        "messages": state_messages,
        "core_memories": state["core_memories"],
        "recall_memories": state["recall_memories"],
        "current_time": datetime.now(tz=timezone.utc).isoformat()
    })
    return {
        "messages": state_messages,
        "final_response": response.content,
        "core_memories": state["core_memories"],
        "recall_memories": state["recall_memories"],
    }
