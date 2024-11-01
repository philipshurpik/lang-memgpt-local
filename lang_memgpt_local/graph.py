import logging
from datetime import datetime, timezone

import tiktoken
from langchain_core.messages.utils import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig, get_executor_for_config
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import Literal

from lang_memgpt_local import _schemas as schemas
from lang_memgpt_local import _utils as utils
from lang_memgpt_local.tools import save_recall_memory, search_memory, store_core_memory, fetch_core_memories
from lang_memgpt_local.tools import search_tool

logger = logging.getLogger("memory")
logger.setLevel(logging.DEBUG)

memory_tools = [save_recall_memory, store_core_memory]
utility_tools = [search_tool, search_memory]
all_tools = memory_tools + utility_tools

# Define the prompt template for the agent
agent_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant with advanced long-term memory capabilities."
     " Powered by a stateless LLM, you must rely on external memory to store"
     " information between conversations."
     " And you have access to tool for external search (search_tool)\n\n"
     "Memory Usage Guidelines:\n"
     "1. Actively use memory tools (save_core_memory, save_recall_memory)"
     " to build a comprehensive understanding of the user.\n"
     "2. Make informed suppositions based on stored memories.\n"
     "3. Regularly reflect on past interactions to identify patterns.\n"
     "4. Update your mental model of the user with new information.\n"
     "5. Cross-reference new information with existing memories.\n"
     "6. Prioritize storing emotional context and personal values.\n"
     "7. Use memory to anticipate needs and tailor responses.\n"
     "8. Recognize changes in user's perspectives over time.\n"
     "9. Leverage memories for personalized examples.\n"
     "10. Recall past experiences to inform problem-solving.\n\n"
     "## Core Memories\n"
     "Core memories are fundamental to understanding the user, his name, basis preferences and are always available:\n"
     "{core_memories}\n\n"
     "## Recall Memories\n"
     "Recall memories are contextually retrieved based on the current conversation:\n{recall_memories}\n\n"
     "Current time: {current_time}"
     ),
    ("placeholder", "{messages}")
])

response_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant with access to memories and search results."
     " Use this context to provide personalized and informed responses.\n\n"
     "Core memories about the user:\n{core_memories}\n\n"
     "Contextual recall memories:\n{recall_memories}\n\n"
     "Search results:\n{search_results}\n\n"
     "Current time: {current_time}"
     ),
    ("placeholder", "{messages}")
])


async def agent_llm(state: schemas.State, config: RunnableConfig) -> schemas.State:
    """Process the current state and generate a response using the LLM.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        schemas.State: The updated state with the agent's response.
    """
    configurable = utils.ensure_configurable(config)
    llm = utils.init_chat_model(configurable["model"])
    bound = agent_prompt | llm.bind_tools(all_tools, tool_choice="auto")
    core_str = "<core_memory>\n" + "\n".join(state["core_memories"]) + "\n</core_memory>"
    recall_str = "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    logger.debug(f"Core memories: {core_str}")
    logger.debug(f"Recall memories: {recall_str}")
    response = await bound.ainvoke(
        {
            "messages": state["messages"],
            "core_memories": core_str,
            "recall_memories": recall_str,
            "current_time": datetime.now(tz=timezone.utc).isoformat(),
        }
    )
    return {
        "messages": response,
        "core_memories": state["core_memories"],
        "recall_memories": state["recall_memories"],
    }


async def response_llm(state: schemas.State, config: dict) -> schemas.State:
    """Final LLM to generate response using memories but no tools"""
    configurable = utils.ensure_configurable(config)
    llm = utils.init_chat_model(configurable["model"])
    bound = response_prompt | llm

    response = await bound.ainvoke({
        "messages": state["messages"],
        "core_memories": state["core_memories"],
        "recall_memories": state["recall_memories"],
        "search_results": state["search_results"],
        "current_time": datetime.now(tz=timezone.utc).isoformat()
    })

    return {
        "messages": state["messages"],
        "final_response": response.content,
        "core_memories": state["core_memories"],
        "recall_memories": state["recall_memories"],
        "search_results": state["search_results"]
    }


def load_memories(state: schemas.State, config: RunnableConfig) -> schemas.State:
    """Load core and recall memories for the current conversation.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        schemas.State: The updated state with loaded memories.
    """
    configurable = utils.ensure_configurable(config)
    user_id = configurable["user_id"]
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    convo_str = get_buffer_string(state["messages"])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])

    with get_executor_for_config(config) as executor:
        futures = [
            executor.submit(fetch_core_memories, user_id),
            executor.submit(search_memory.invoke, convo_str),
        ]
        _, core_memories = futures[0].result()
        recall_memories = futures[1].result()
    return {
        "messages": state["messages"],
        "core_memories": core_memories,
        "recall_memories": recall_memories,
    }


def route_tools(state: schemas.State) -> Literal["tools", "response_llm"]:
    """Route to tools or final LLM based on agent response"""
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"
    return "response_llm"


# Create the LangGraph StateGraph
builder = StateGraph(schemas.State, schemas.GraphConfig)
builder.add_node("load_memories", load_memories)
builder.add_node("agent_llm", agent_llm)
builder.add_node("tools", ToolNode(all_tools))
builder.add_node("response_llm", response_llm)

# Add edges to the graph
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent_llm")
builder.add_conditional_edges("agent_llm", route_tools, ["tools", "response_llm"])
builder.add_edge("tools", "response_llm")
builder.add_edge("response_llm", END)

# Compile the graph into an executable LangGraph
memory = MemorySaver()
memgraph = builder.compile(checkpointer=memory)

__all__ = ["memgraph"]

if __name__ == "__main__":
    graph_image = memgraph.get_graph().draw_mermaid_png()
    with open("../memgraph.png", "wb") as f:
        f.write(graph_image)
    print("ok")
