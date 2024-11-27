import json

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from memory_langgraph.app_ctx import GraphConfig, ctx
from memory_langgraph.graph import memgraph


@pytest.mark.asyncio
async def test_core_memory(mock_app_context):
    config = GraphConfig(thread_id="test_thread_id", user_id="test_user_id")
    messages = [
        HumanMessage(content="When I was young, I had a dog named Spot. He was my favorite pup."),
    ]
    output_state = await memgraph.ainvoke({"messages": messages}, {"configurable": config})

    # Verify that the core memory adapter's upsert method was called
    assert mock_app_context.core_memory_adapter.upsert.call_count >= 1

    # Extract the memories that were attempted to be saved
    upsert_calls = mock_app_context.core_memory_adapter.upsert.call_args_list
    saved_memories = []
    for call in upsert_calls:
        args, kwargs = call
        metadatas = args[2]
        payload = metadatas[0][ctx.constants.PAYLOAD_KEY]
        memories = json.loads(payload)["memories"]
        saved_memories.append(memories)

    # Ensure that the new memory was saved
    new_memory_saved = any("spot" in v.lower() for mem in saved_memories for v in mem.values())
    assert new_memory_saved, "New memory about Spot should have been saved to core memories"

    # Ensure existing memories are still accessible
    existing_memory_preserved = any(
        "spiders" in v.lower() for mem in saved_memories for v in mem.values()
    )
    assert existing_memory_preserved, "Existing memory about spiders should still be present"