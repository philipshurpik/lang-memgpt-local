import pytest
from langchain_core.messages import HumanMessage

from memory_langgraph.app_ctx import GraphConfig
from memory_langgraph.graph import memgraph


@pytest.mark.asyncio
async def test_insert_recall_memory(mock_app_context):
    config = GraphConfig(thread_id="test_thread_id", user_id="test_user_id")
    messages = [
        HumanMessage(content="I went to the beach with my friends today.")
    ]
    output_state = await memgraph.ainvoke({"messages": messages}, {"configurable": config})

    # Verify that recall memory adapter's add_memory method was called
    assert mock_app_context.recall_memory_adapter.add_memory.call_count >= 1

    # Extract the memory that was attempted to be saved
    add_memory_calls = mock_app_context.recall_memory_adapter.add_memory.call_args_list
    added_memories = []
    for call in add_memory_calls:
        args, kwargs = call
        added_memories.append(args[3])  # Assuming the 4th arg is the memory (data)

    # Ensure that the new recall memory was saved
    new_memory_saved = any("beach" in mem.lower() for mem in added_memories)
    assert new_memory_saved, "New recall memory about the beach should have been saved"
