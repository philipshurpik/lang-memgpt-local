import logging
from typing import List
from langchain_core.messages import BaseMessage, SystemMessage, RemoveMessage
from langchain_core.messages.utils import get_buffer_string
from ..app_ctx import ctx, State

logger = logging.getLogger("summarize")


async def summarize_conversation(state: State) -> State:
    """Summarize old messages and save to recall memory."""
    messages_to_summarize: List[BaseMessage] = state.get("to_summarize", [])
    
    if not messages_to_summarize:
        return state
        
    summary_messages = [
        SystemMessage(content="Create a concise summary of this conversation segment that captures key points and context."),
        *messages_to_summarize
    ]
    
    summary = await ctx.summary_model.ainvoke(summary_messages)
    
    await ctx.tools.save_recall_memory.ainvoke(summary.content)
    
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]
    
    return {
        "messages": delete_messages,
        "core_memories": state["core_memories"],
        "recall_memories": state["recall_memories"],
        "final_response": state["final_response"],
        "to_summarize": []  # Clear the to_summarize list
    }
