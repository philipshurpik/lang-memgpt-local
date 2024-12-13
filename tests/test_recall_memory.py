import pytest
from langchain_core.messages import HumanMessage

from memory_langgraph.app_ctx import GraphConfig
from memory_langgraph.graph import memgraph


@pytest.mark.asyncio
async def test_recall_memory(mock_app_context):
    config = GraphConfig(thread_id="test_thread_id", user_id="test_user_id")
    messages = [
        HumanMessage(content="I went to the beach with my friends today.")
    ]

    # Use ainvoke for async execution
    _ = await memgraph.ainvoke(
        {"messages": messages},
        {"configurable": config}
    )

    # Verify memory was saved
    mock_app_context.recall_memory_adapter.add_memory.assert_awaited_once()
    call_args = mock_app_context.recall_memory_adapter.add_memory.await_args

    # Check content was saved correctly
    assert "beach" in call_args[1]["content"].lower()

    # Check metadata structure
    metadata = call_args[1]["metadata"]
    assert metadata["user_id"] == "test_user_id"
    assert metadata["type"] == "recall"
    assert "timestamp" in metadata