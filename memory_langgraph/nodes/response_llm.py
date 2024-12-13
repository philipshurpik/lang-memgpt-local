from datetime import datetime, timezone

from langchain_core.messages.ai import AIMessage

from ..app_ctx import State, ctx


async def response_llm(state: State) -> State:
    """Final LLM to generate response using memories but no tools."""
    chain = ctx.response_prompt | ctx.response_model

    state_messages = state["messages"]
    if isinstance(state_messages[-1], AIMessage):
        state_messages = state_messages[:-1]

    response = await chain.ainvoke({
        "messages": state_messages,
        "core_memories": state["core_memories"],
        "recall_memories": state["recall_memories"],
        "current_time": datetime.now(tz=timezone.utc).isoformat()
    })
    return {
        "messages": state_messages,
        "core_memories": state["core_memories"],
        "recall_memories": state["recall_memories"],
    }
