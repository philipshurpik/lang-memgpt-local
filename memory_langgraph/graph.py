import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import Literal

from memory_langgraph.app_ctx import GraphConfig, State
from memory_langgraph.nodes import (
    agent_llm,
    load_memories,
    response_llm,
    summarize_conversation,
)
from memory_langgraph.tools import (
    ask_wisdom,
    save_core_memories,
    save_recall_memory,
    search_recall_memory,
    search_tavily,
)

logger = logging.getLogger("memory")
logger.setLevel(logging.DEBUG)

all_tools = [ask_wisdom, save_core_memories, save_recall_memory, search_recall_memory, search_tavily]


def route_tools(state: State) -> Literal["tools", "response_llm"]:
    """Route to tools or final LLM based on agent response."""
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"
    return "response_llm"


def should_summarize(state: State) -> Literal["summarize", "end"]:
    """Route to summarize or end based on message count."""
    messages = state["messages"]
    if len(messages) >= 10:
        # Store all but last 2 messages for summarization
        state["to_summarize"] = messages[:-2]
        return "summarize"
    return "end"


builder = StateGraph(State, GraphConfig)
builder.add_node("load_memories", load_memories)
builder.add_node("agent_llm", agent_llm)
builder.add_node("tools", ToolNode(all_tools))
builder.add_node("response_llm", response_llm)
builder.add_node("summarize", summarize_conversation)

# Add edges to the graph
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent_llm")
builder.add_conditional_edges("agent_llm", route_tools, ["tools", "response_llm"])
builder.add_edge("tools", "response_llm")
builder.add_conditional_edges("response_llm", should_summarize, {"summarize": "summarize", "end": END})
builder.add_edge("summarize", END)

# Compile the graph into an executable LangGraph
memory = MemorySaver()
memgraph = builder.compile(checkpointer=memory)

__all__ = ["memgraph"]

if __name__ == "__main__":
    graph_image = memgraph.get_graph().draw_mermaid_png()
    with open("../memgraph.png", "wb") as f:
        f.write(graph_image)
    print("Graph image saved as memgraph.png")
