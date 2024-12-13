import pytest
from langchain_core.messages import HumanMessage

from memory_langgraph.app_ctx import GraphConfig
from memory_langgraph.graph import memgraph


@pytest.mark.asyncio
async def test_core_memory(mock_app_context):
    config = GraphConfig(thread_id="test_thread_id", user_id="test_user_id")
    messages = [
        HumanMessage(content="When I was young, I had a dog named Spot. He was my favorite pup."),
    ]
    _ = await memgraph.ainvoke({"messages": messages}, {"configurable": config})

    # Verify that the core memory adapter's save_memory method was called
    assert mock_app_context.core_memory_adapter.save_memory.call_count >= 1

    # Extract the saved memory details
    save_calls = mock_app_context.core_memory_adapter.save_memory.call_args_list
    saved_memories = []
    for call in save_calls:
        args = call[0]  # args are (user_id, key, value)
        saved_memories.append((args[1], args[2]))  # (key, value) pairs

    # Ensure that the new memory about Spot was saved
    spot_memory_saved = any(
        "spot" in value.lower() for _, value in saved_memories
    )
    assert spot_memory_saved, "New memory about Spot should have been saved"