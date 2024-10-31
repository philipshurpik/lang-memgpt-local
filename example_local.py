import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Union, Dict, Any

import langsmith
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint import MemorySaver
from langgraph.graph import START, StateGraph, add_messages
from pydantic.v1 import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from lang_memgpt_local import (
    _constants as constants,
    _settings as settings,
    _utils as utils,
)
from lang_memgpt_local.graph import memgraph

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# Adjust logging level for LangSmith client
logging.getLogger('langsmith.client').setLevel(logging.ERROR)


class ChatState(TypedDict):
    """The state of the chatbot."""
    messages: Annotated[List[AnyMessage], add_messages]
    user_memories: List[dict]


class ChatConfigurable(TypedDict):
    """The configurable fields for the chatbot."""
    user_id: str
    thread_id: str
    model: str
    delay: Optional[float]


def _ensure_configurable(config: RunnableConfig) -> ChatConfigurable:
    """Ensure the configuration is valid."""
    return ChatConfigurable(
        user_id=config["configurable"]["user_id"],
        thread_id=config["configurable"]["thread_id"],
        model=config["configurable"].get(
            "model", "accounts/fireworks/models/firefunction-v2"
        ),
        delay=config["configurable"].get("delay", 60),
    )


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful and friendly chatbot. Get to know the user!"
            " Ask questions! Be spontaneous!"
            "{user_info}\n\nSystem Time: {time}",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(
    time=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
)


@langsmith.traceable
def format_query(messages: List[AnyMessage]) -> str:
    """Format the query for the user's memories."""
    return " ".join([str(m.content) for m in messages if m.type == "human"][-5:])


async def query_memories(state: ChatState, config: RunnableConfig) -> ChatState:
    """Query the user's memories."""
    configurable: ChatConfigurable = config["configurable"]
    user_id = configurable["user_id"]
    embeddings = utils.get_embeddings()

    query = format_query(state["messages"])
    vec = await embeddings.aembed_query(query)
    chroma_client = utils.get_chroma_client()
    collection = chroma_client.get_or_create_collection("memories")

    with langsmith.trace(
            "chroma_query", inputs={"query": query, "user_id": user_id}
    ) as rt:
        results = collection.query(
            query_embeddings=[vec],
            where={"user_id": str(user_id)},
            n_results=10,
        )
        rt.outputs["response"] = results

    memories = [m[constants.PAYLOAD_KEY] for m in results['metadatas'][0]]
    return {
        "user_memories": memories,
    }


@langsmith.traceable
def format_memories(memories: List[dict]) -> str:
    """Format the user's memories."""
    if not memories:
        return ""
    memories = "\n".join(str(m) for m in memories)
    return f"""

## Memories

You have noted the following memorable events from previous interactions with the user.
<memories>
{memories}
</memories>
"""


async def bot(state: ChatState, config: RunnableConfig) -> ChatState:
    """Prompt the bot to respond to the user, incorporating memories (if provided)."""
    configurable = _ensure_configurable(config)
    model = init_chat_model(configurable["model"])
    chain = PROMPT | model
    memories = format_memories(state["user_memories"])
    m = await chain.ainvoke(
        {
            "messages": state["messages"],
            "user_info": memories,
        },
        config,
    )

    return {
        "messages": [m],
    }


class MemorableEvent(BaseModel):
    """A memorable event."""
    description: str
    participants: List[str] = Field(
        description="Names of participants in the event and their relationship to the user."
    )


async def post_messages(state: ChatState, config: RunnableConfig) -> ChatState:
    """Process messages and store memories."""
    configurable = _ensure_configurable(config)
    thread_id = config["configurable"]["thread_id"]
    memory_thread_id = uuid.uuid5(uuid.NAMESPACE_URL, f"memory_{thread_id}")

    # Here you would implement the logic to process messages and store memories
    # For example:
    # memories = extract_memories(state["messages"])
    # for memory in memories:
    #     await save_recall_memory(memory)

    return {
        "messages": [],
    }


builder = StateGraph(ChatState, ChatConfigurable)
builder.add_node(query_memories)
builder.add_node(bot)
builder.add_node(post_messages)
builder.add_edge(START, "query_memories")
builder.add_edge("query_memories", "bot")
builder.add_edge("bot", "post_messages")

chat_graph = builder.compile(checkpointer=MemorySaver())


# Example usage
async def main():
    user_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())

    chat = Chat(user_id, thread_id)

    response = await chat("Hi there")
    print("Bot:", response)

    response = await chat("I've been planning a surprise party for my friend Steve.")
    print("Bot:", response)

    response = await chat("Steve really likes crocheting. Maybe I can do something with that?")
    print("Bot:", response)

    response = await chat("He's also into capoeira...")
    print("Bot:", response)

    # Wait for a minute to simulate time passing
    print("Waiting for a 30 sec to simulate time passing...")
    await asyncio.sleep(30)

    # Start a new conversation
    thread_id_2 = str(uuid.uuid4())
    chat2 = Chat(user_id, thread_id_2)

    response = await chat2("Remember me?")
    print("Bot:", response)

    response = await chat2("What do you remember about Steve?")
    print("Bot:", response)


class Chat:
    def __init__(self, user_id: str, thread_id: str):
        self.thread_id = thread_id
        self.user_id = user_id

    async def __call__(self, query: str) -> str:
        logger.debug(f"Chat called with query: {query}")
        logger.debug(f"User ID: {self.user_id}, Thread ID: {self.thread_id}")

        chunks = memgraph.astream_events(
            input={
                "messages": [("human", query)],
            },
            config={
                "configurable": {
                    "user_id": self.user_id,
                    "thread_id": self.thread_id,
                    "model": settings.SETTINGS.model,
                    "delay": 4,
                }
            },
            version="v1",
        )
        res = []
        try:
            async for event in chunks:
                if event.get("event") == "on_chat_model_stream":
                    tok = event["data"]["chunk"].content
                    self.process_token(tok, res)
                elif event.get("event") == "on_tool_start":
                    logger.debug(f"Tool started: {event.get('name')}")
                elif event.get("event") == "on_tool_end":
                    logger.debug(f"Tool ended: {event.get('name')}")
                    logger.debug(f"Tool output: {event.get('data', {}).get('output')}")
        except Exception as e:
            logger.error(f"Error during chat streaming: {str(e)}")

        print()  # New line after all output
        full_response = "".join(res)
        logger.debug(f"Full response: {full_response}")
        return full_response

    def process_token(self, tok: Union[str, list, Dict[str, Any]], res: list):
        if isinstance(tok, str):
            print(tok, end="", flush=True)
            res.append(tok)
        elif isinstance(tok, list):
            for item in tok:
                self.process_token(item, res)
        elif isinstance(tok, dict):
            if 'text' in tok:
                self.process_token(tok['text'], res)
            else:
                logger.warning(f"Received dict without 'text' key: {tok}")
        else:
            logger.warning(f"Unexpected token type: {type(tok)}")


if __name__ == "__main__":
    asyncio.run(main())
