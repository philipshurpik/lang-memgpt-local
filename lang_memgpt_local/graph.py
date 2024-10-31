import logging
from datetime import datetime, timezone

import tiktoken
from langchain_core.messages.utils import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig, get_executor_for_config
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import Literal

from lang_memgpt_local import _constants as constants
from lang_memgpt_local import _schemas as schemas
from lang_memgpt_local import _utils as utils
from lang_memgpt_local.tools import save_recall_memory, search_memory, store_core_memory, fetch_core_memories
from lang_memgpt_local.tools import search_tool

logger = logging.getLogger("memory")
logger.setLevel(logging.DEBUG)

all_tools = [search_tool, save_recall_memory, search_memory, store_core_memory]

# Define the prompt template for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant with advanced long-term memory"
            " capabilities. Utilize the available memory tools to store and retrieve"
            " important details that will help you better attend to the user's"
            " needs and understand their context.\n\n"
            "## Core Memories\n"
            "Core memories are fundamental to understanding the user and are"
            " always available:\n{core_memories}\n\n"
            "## Recall Memories\n"
            "Recall memories are contextually retrieved based on the current"
            " conversation:\n{recall_memories}\n\n"
            "## Instructions\n"
            "Engage with the user naturally, as a trusted colleague or friend."
            " There's no need to explicitly mention your memory capabilities."
            " Instead, seamlessly incorporate your understanding of the user"
            " into your responses. Be attentive to subtle cues and underlying"
            " emotions. Adapt your communication style to match the user's"
            " preferences and current emotional state. Use tools to persist"
            " information you want to retain in the next conversation.\n\n"
            "Current system time: {current_time}\n\n",
        ),
        ("placeholder", "{messages}"),
    ]
)


# Main agent function
async def agent(state: schemas.State, config: RunnableConfig) -> schemas.State:
    """Process the current state and generate a response using the LLM."""
    configurable = utils.ensure_configurable(config)
    llm = utils.init_chat_model(configurable["model"])
    bound = prompt | llm.bind_tools(all_tools)
    core_str = (
            "<core_memory>\n" + "\n".join(state["core_memories"]) + "\n</core_memory>"
    )
    recall_str = (
            "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    logger.debug(f"Core memories: {core_str}")
    logger.debug(f"Recall memories: {recall_str}")
    prediction = await bound.ainvoke(
        {
            "messages": state["messages"],
            "core_memories": core_str,
            "recall_memories": recall_str,
            "current_time": datetime.now(tz=timezone.utc).isoformat(),
        }
    )
    return {
        "messages": prediction,
    }


# Function to load memories for the current conversation
def load_memories(state: schemas.State, config: RunnableConfig) -> schemas.State:
    """Load core and recall memories for the current conversation."""
    configurable = utils.ensure_configurable(config)
    user_id = configurable["user_id"]
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
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
        "core_memories": core_memories,
        "recall_memories": recall_memories,
    }


async def query_memories(state: schemas.State, config: RunnableConfig) -> schemas.State:
    """Query the user's memories."""
    configurable = utils.ensure_configurable(config)
    user_id = configurable["user_id"]
    embeddings = utils.get_embeddings()

    # Get the last few messages to use as a query
    last_messages = state["messages"][-5:]  # Adjust this number as needed
    query = " ".join([str(m.content) for m in last_messages if m.type == "human"])
    logger.debug(f"Querying memories with: {query}")

    vec = await embeddings.aembed_query(query)
    chroma_client = utils.get_vectordb_client()
    collection = chroma_client.get_or_create_collection("memories")

    # Correct the where clause format
    where_clause = {
        "$and": [
            {"user_id": {"$eq": str(user_id)}},
            {constants.TYPE_KEY: {"$eq": "recall"}}
        ]
    }

    logger.debug(f"Searching for memories with where clause: {where_clause}")

    results = collection.query(
        query_embeddings=[vec],
        where=where_clause,
        n_results=10,
    )

    # Correct handling of ChromaDB query results
    memories = []
    if results['metadatas']:
        for metadata in results['metadatas']:
            if isinstance(metadata, list):
                memories.extend([m.get(constants.PAYLOAD_KEY) for m in metadata if constants.PAYLOAD_KEY in m])
            elif isinstance(metadata, dict):
                if constants.PAYLOAD_KEY in metadata:
                    memories.append(metadata[constants.PAYLOAD_KEY])

    logger.debug(f"Retrieved memories: {memories}")

    return {
        "recall_memories": memories,
    }


# Function to determine the next step in the graph
def route_tools(state: schemas.State) -> Literal["tools", "__end__"]:
    """Determine whether to use tools or end the conversation based on the last message."""
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"
    return END


# Create the LangGraph StateGraph
builder = StateGraph(schemas.State, schemas.GraphConfig)

# Add nodes to the graph
builder.add_node("load_memories", load_memories)
builder.add_node("query_memories", query_memories)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(all_tools))

# Update the edges to include query_memories
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "query_memories")
builder.add_edge("query_memories", "agent")
builder.add_conditional_edges("agent", route_tools)
builder.add_edge("tools", "query_memories")

# Compile the graph into an executable LangGraph
memgraph = builder.compile()

__all__ = ["memgraph"]
