import asyncio
import logging
import uuid
from typing import Union, Dict, Any

from dotenv import load_dotenv

from lang_memgpt_local import _settings as settings
from lang_memgpt_local.graph import memgraph

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('langsmith.client').setLevel(logging.ERROR)


async def main():
    user_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())

    chat = Chat(user_id, thread_id)

    response = await chat("Hi there")
    print("Bot:", response)

    response = await chat("I enjoy belgian and german chocolate")
    print("Bot:", response)

    response = await chat("What chocolate can you recommend?")
    print("Bot:", response)

    response = await chat("I like ritter sport")
    print("Bot:", response)

    # Wait for a minute to simulate time passing
    print("Waiting for a 60 sec to simulate time passing...")
    await asyncio.sleep(60)

    # Start a new conversation
    thread_id_2 = str(uuid.uuid4())
    chat2 = Chat(user_id, thread_id_2)

    response = await chat2("Remember me?")
    print("Bot:", response)

    response = await chat2("What is my favorite chocolate?")
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
