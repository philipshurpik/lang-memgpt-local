import logging
from typing import Union, Dict, Any

from lang_memgpt_local.graph import memgraph
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('langsmith.client').setLevel(logging.ERROR)


class Chat:
    def __init__(self, user_id: str, thread_id: str):
        self.thread_id = thread_id
        self.user_id = user_id

    async def __call__(self, query: str) -> str:
        logger.debug(f"Chat called with query: {query}")
        logger.debug(f"User ID: {self.user_id}, Thread ID: {self.thread_id}")

        config = {"configurable": {"user_id": self.user_id, "thread_id": self.thread_id}}
        input_message = HumanMessage(content=query)
        chunks = memgraph.astream_events(
            input={"messages": [input_message]},
            config=config,
            version="v2",
        )
        res = []
        async for event in chunks:
            if event.get("event") == "on_chat_model_stream":
                if event.get('metadata', {}).get('langgraph_node', {}) == 'response_llm':
                    tok = event["data"]["chunk"].content
                    self.process_token(tok, res)
            elif event.get("event") == "on_tool_start":
                logger.debug(f"Tool started: {event.get('name')}")
            elif event.get("event") == "on_tool_end":
                logger.debug(f"Tool ended: {event.get('name')}")
                logger.debug(f"Tool output: {event.get('data', {}).get('output')}")

        print()  # New line after all output
        full_response = "".join(res)
        logger.debug(f"Full response: {full_response}")
        return full_response

    def process_token(self, tok: Union[str, list, Dict[str, Any]], res: list):
        if isinstance(tok, str):
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
