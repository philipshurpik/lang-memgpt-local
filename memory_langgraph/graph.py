import logging
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_core.messages.utils import get_buffer_string
from langchain_core.runnables.config import RunnableConfig, get_executor_for_config
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import Literal

from memory_langgraph.app_ctx import ctx, State, GraphConfig
from memory_langgraph.tools import ask_wisdom, load_core_memories, save_recall_memory, search_recall_memory, search_tavily, \
    save_core_memories

load_dotenv()
logger = logging.getLogger("memory")
logger.setLevel(logging.DEBUG)

memory_tools = [save_recall_memory, save_core_memories]
utility_tools = [search_tavily, search_recall_memory, ask_wisdom]
all_tools = memory_tools + utility_tools


async def agent_llm(state: State, config: RunnableConfig) -> State:
    """Process the current state and generate a response using the LLM."""
    llm = ctx.agent_model.bind_tools(all_tools, tool_choice="auto")
    bound = ctx.prompts["agent"] | llm

    core_str = "<core_memory>\n" + "\n".join(
        [f"{k}: {v}" for k, v in state["core_memories"].items()]) + "\n</core_memory>"
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


async def response_llm(state: State, config: dict) -> State:
    """Final LLM to generate response using memories but no tools."""
    llm = ctx.response_model
    bound = ctx.prompts["response"] | llm

    state_messages = state["messages"][:-1] if state["messages"][-1].type == 'ai' else state["messages"]
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


def load_memories(state: State, config: RunnableConfig) -> State:
    """Load core and recall memories for the current conversation."""
    configurable = ctx.ensure_configurable(config)
    user_id = configurable["user_id"]
    tokenizer = ctx.get_tokenizer()
    convo_str = get_buffer_string(state["messages"])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])

    with get_executor_for_config(config) as executor:
        futures = [
            executor.submit(load_core_memories, user_id),
            executor.submit(search_recall_memory.invoke, convo_str),
        ]
        _, core_memories = futures[0].result()
        recall_memories = futures[1].result()
    return {
        "messages": state["messages"],
        "core_memories": core_memories,
        "recall_memories": recall_memories,
        "final_response": None,
    }


def route_tools(state: State) -> Literal["tools", "response_llm"]:
    """Route to tools or final LLM based on agent response."""
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"
    return "response_llm"


builder = StateGraph(State, GraphConfig)
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
    print("Graph image saved as memgraph.png")
