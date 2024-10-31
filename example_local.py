import asyncio
import uuid

from dotenv import load_dotenv

from lang_memgpt_local.chat import Chat

load_dotenv()


async def main():
    user_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())

    chat = Chat(user_id, thread_id)

    response = await chat("Hi there, my name is Philip")
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


if __name__ == "__main__":
    asyncio.run(main())
