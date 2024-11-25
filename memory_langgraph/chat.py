import logging

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from .graph import memgraph

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat")
logging.getLogger('langsmith.client').setLevel(logging.ERROR)


class Chat:
    def __init__(self, user_id: str, thread_id: str):
        self.thread_id = thread_id
        self.user_id = user_id

    async def stream_response(self, query: str):
        logger.debug(f"Chat called with query: {query}")
        logger.debug(f"User ID: {self.user_id}, Thread ID: {self.thread_id}")

        config = {"configurable": {"user_id": self.user_id, "thread_id": self.thread_id}}
        input_message = HumanMessage(content=query)
        chunks = memgraph.astream_events(
            input={"messages": [input_message]},
            config=config,
            version="v2",
        )

        async for event in chunks:
            if event.get("event") == "on_chat_model_stream":
                if event.get('metadata', {}).get('langgraph_node', {}) == 'response_llm':
                    tok = event["data"]["chunk"].content
                    yield tok  # Yield the token as it's received
            elif event.get("event") == "on_tool_start":
                logger.debug(f"Tool started: {event.get('name')}")
            elif event.get("event") == "on_tool_end":
                logger.debug(f"Tool ended: {event.get('name')}")
                logger.debug(f"Tool output: {event.get('data', {}).get('output')}")

    async def __call__(self, query: str) -> str:
        res = []
        async for tok in self.stream_response(query):
            res.append(tok)
        full_response = "".join(res)
        logger.debug(f"Full response: {full_response}")
        return full_response
