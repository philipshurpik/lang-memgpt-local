import logging
from datetime import datetime, timezone

import tiktoken
from langchain import hub
from dotenv import load_dotenv
from langchain_core.messages.utils import get_buffer_string
from langchain_core.runnables.config import RunnableConfig, get_executor_for_config
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import Literal

from lang_memgpt_local import _schemas as schemas
from lang_memgpt_local import _utils as utils
from lang_memgpt_local.tools import save_recall_memory, search_memory, store_core_memory, fetch_core_memories
from lang_memgpt_local.tools import search_tool, ask_wisdom

load_dotenv()
logger = logging.getLogger("memory")
logger.setLevel(logging.DEBUG)

memory_tools = [save_recall_memory, store_core_memory]
utility_tools = [search_tool, search_memory, ask_wisdom]
all_tools = memory_tools + utility_tools

prompts = {
    "agent": hub.pull("langgraph-agent"),
    "response": hub.pull("langgraph-response"),
}


async def agent_llm(state: schemas.State, config: RunnableConfig) -> schemas.State:
    """Process the current state and generate a response using the LLM.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        schemas.State: The updated state with the agent's response.
    """
    configurable = utils.ensure_configurable(config)
    llm = utils.init_agent_model(configurable["model"])
    bound = prompts["agent"] | llm.bind_tools(all_tools, tool_choice="auto")
    core_str = "<core_memory>\n" + "\n".join([f"{k}: {v}" for k, v in state["core_memories"].items()]) + "\n</core_memory>"
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
        "final_response": None,
    }


async def response_llm(state: schemas.State, config: dict) -> schemas.State:
    """Final LLM to generate response using memories but no tools"""
    llm = utils.init_response_model()
    bound = prompts["response"] | llm

    response = await bound.ainvoke({
        "messages": state["messages"][:-1] if state["messages"][-1].type == 'ai' else state["messages"],
        "core_memories": state["core_memories"],
        "recall_memories": state["recall_memories"],
        "current_time": datetime.now(tz=timezone.utc).isoformat()
    })

    return {
        "messages": state["messages"],
        "final_response": response.content,
        "core_memories": state["core_memories"],
        "recall_memories": state["recall_memories"],
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
        "final_response": None,
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
