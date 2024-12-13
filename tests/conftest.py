from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolCall

from memory_langgraph.app_ctx import ctx


class MockAgentChain:
    async def ainvoke(self, input_dict):
        last_user_message = input_dict["messages"][-1].content
        if "Spot" in last_user_message:  # core memory tool call
            tool_calls = [ToolCall(
                name="save_core_memories",
                args={'key': 'favorite_pet', 'value': 'dog named Spot'},
                id="tool_call_123"
            )]
        elif "beach" in last_user_message:  # recall memory tool call
            tool_calls = [ToolCall(
                name="save_recall_memory",
                args={"memory": "I went to the beach with my friends today."},
                id="tool_call_123"
            )]
        else:
            tool_calls = []
        return AIMessage(
            content="Tell me more!" if len(tool_calls) == 0 else "",
            tool_calls=tool_calls
        )


class MockResponseChain:
    async def ainvoke(self, input_dict):
        return AIMessage(content="This is a mocked final response.")


class MockAgentPrompt:
    def __or__(self, other):
        return MockAgentChain()


class MockResponsePrompt:
    def __or__(self, other):
        return MockResponseChain()


@pytest.fixture(scope="function")
def mock_app_context():
    with patch('memory_langgraph.app_ctx.AppCtx.__init__', return_value=None):
        # Mock context attributes
        ctx.settings = MagicMock()

        # Mock core_memory_adapter
        ctx.core_memory_adapter = AsyncMock()
        ctx.core_memory_adapter.save_memory = AsyncMock()
        ctx.core_memory_adapter.get_memories = AsyncMock(return_value={
            "phobia": "afraid of spiders"
        })

        # Mock recall_memory_adapter
        ctx.recall_memory_adapter = AsyncMock()
        ctx.recall_memory_adapter.add_memory = AsyncMock()
        ctx.recall_memory_adapter.query_memories = AsyncMock(return_value=[
            {"content": "I like to swim.", "timestamp": "2024-01-01T00:00:00Z"}
        ])

        # Mock embeddings
        ctx.qdrant_memory_embeddings = AsyncMock()
        ctx.qdrant_memory_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

        # Mock vector store
        ctx.qdrant_vectorstore = AsyncMock()
        ctx.qdrant_vectorstore.asimilarity_search = AsyncMock(return_value=[
            MagicMock(page_content="Some wisdom content")
        ])

        # Mock prompts and chains
        ctx.agent_prompt = MockAgentPrompt()
        ctx.response_prompt = MockResponsePrompt()
        yield ctx


@pytest.fixture(scope="session", autouse=True)
def mock_mongo_client():
    with patch("motor.motor_asyncio.AsyncIOMotorClient") as mock_client:
        yield mock_client
