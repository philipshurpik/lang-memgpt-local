import asyncio
import uuid

from memory_langgraph.chat import Chat


async def main():
    user_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())

    chat = Chat(user_id, thread_id)

    response = await chat("Hello!  What's your name?")
    print("Bot:", response)

    response = await chat("My name is Luna. I like to eat chocolate")
    print("Bot:", response)

    response = await chat("What is the weather in Warsaw?")
    print("Bot:", response)

    response = await chat("What can you tell me about motivation?")
    print("Bot:", response)
    #
    # response = await chat("I like ritter sport")
    # print("Bot:", response)
    #
    # response = await chat("What are actual US president election polls results?")
    # print("Bot:", response)

    # Wait for a minute to simulate time passing
    print("Waiting for a 10 sec to simulate time passing...")
    await asyncio.sleep(10)

    # Start a new conversation
    thread_id_2 = str(uuid.uuid4())
    chat2 = Chat(user_id, thread_id_2)

    response = await chat2("Remember me?")
    print("Bot:", response)

    response = await chat2("Sorry, my name is Lunar!")
    print("Bot:", response)

    response = await chat2("What do I like?")
    print("Bot:", response)


if __name__ == "__main__":
    asyncio.run(main())
