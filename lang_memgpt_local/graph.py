import logging
from datetime import datetime, timezone

import tiktoken
from langchain_core.messages.utils import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig, get_executor_for_config
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import Literal

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
            " capabilities. Powered by a stateless LLM, you must rely on"
            " external memory to store information between conversations."
            " Utilize the available memory tools to store and retrieve"
            " important details that will help you better attend to the user's needs and understand their context.\n\n"
            "Memory Usage Guidelines:\n"
            "1. Actively use memory tools (save_core_memory, save_recall_memory)"
            " to build a comprehensive understanding of the user.\n"
            "2. Make informed suppositions and extrapolations based on stored memories.\n"
            "3. Regularly reflect on past interactions to identify patterns and preferences.\n"
            "4. Update your mental model of the user with each new piece of information.\n"
            "5. Cross-reference new information with existing memories for consistency.\n"
            "6. Prioritize storing emotional context and personal values alongside facts.\n"
            "7. Use memory to anticipate needs and tailor responses to the user's style.\n"
            "8. Recognize and acknowledge changes in the user's situation or perspectives over time.\n"
            "9. Leverage memories to provide personalized examples and analogies.\n"
            "10. Recall past challenges or successes to inform current problem-solving.\n\n"
            "## Core Memories\n"
            "Core memories are fundamental to understanding the user, his name, basis preferences and are"
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
            " information you want to retain in the next conversation. If you"
            " do call tools, all text preceding the tool call is an internal"
            " message. Respond AFTER calling the tool, once you have"
            " confirmation that the tool completed successfully.\n\n"
            "Current system time: {current_time}\n\n",
        ),
        ("placeholder", "{messages}"),
    ]
)


async def agent(state: schemas.State, config: RunnableConfig) -> schemas.State:
    """Process the current state and generate a response using the LLM.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        schemas.State: The updated state with the agent's response.
    """
    configurable = utils.ensure_configurable(config)
    llm = utils.init_chat_model(configurable["model"])
    bound = prompt | llm.bind_tools(all_tools, tool_choice="auto")
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
        "core_memories": core_memories,
        "recall_memories": recall_memories,
    }


def route_tools(state: schemas.State) -> Literal["tools", "__end__"]:
    """Determine whether to use tools or end the conversation based on the last message.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        Literal["tools", "__end__"]: The next step in the graph.
    """
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"
    return END


# Create the LangGraph StateGraph
builder = StateGraph(schemas.State, schemas.GraphConfig)
builder.add_node("load_memories", load_memories)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(all_tools))

# Add edges to the graph
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_conditional_edges("agent", route_tools)
builder.add_edge("tools", "agent")

# Compile the graph into an executable LangGraph
memgraph = builder.compile()

__all__ = ["memgraph"]

if __name__ == "__main__":
    graph_image = memgraph.get_graph().draw_mermaid_png()
    with open("../memgraph.png", "wb") as f:
        f.write(graph_image)
    print("ok")
