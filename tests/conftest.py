import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolCall

from memory_langgraph.app_ctx import Constants, ctx


class MockAgentChain:
    async def ainvoke(self, input_dict):
        last_user_message = input_dict["messages"][-1].content
        if "Spot" in last_user_message:  # core memory tool call
            tool_calls = [ToolCall(
                name="save_core_memories",
                args={'key': 'favorite_pet', 'value': 'dog named Spot'},
                id="tool_call_123"  # Unique identifier for the tool call
            )]
        elif "beach" in last_user_message:  # recall memory tool call
            tool_calls = [ToolCall(
                name="save_recall_memory",
                args={"memory": "I went to the beach with my friends today."},
                id="tool_call_123"  # Unique identifier for the tool call
            )]
        else:
            tool_calls = []
        return AIMessage(
            content="Tell me more!" if len(tool_calls) == 0 else "",
            tool_calls=tool_calls
        )


class MockResponseChain:
    async def ainvoke(self, input_dict):
        # Return an AIMessage with content
        return AIMessage(content="This is a mocked final response.")


class MockAgentPrompt:
    # Mock the '|' operator to return the appropriate chain object
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
        ctx.constants = Constants()

        # Set necessary constants
        ctx.constants.PATCH_PATH = "user/{user_id}/core"
        ctx.constants.PAYLOAD_KEY = "content"
        ctx.constants.PATH_KEY = "path"
        ctx.constants.TIMESTAMP_KEY = "timestamp"
        ctx.constants.TYPE_KEY = "type"
        ctx.settings.core_collection = "core_memories_collection"
        ctx.settings.recall_collection = "recall_memories_collection"

        # Mock core_memory_adapter
        ctx.core_memory_adapter = MagicMock()
        mock_collection = MagicMock()
        ctx.core_memory_adapter.get_collection.return_value = mock_collection

        # Mock the return value of collection.get()
        mock_collection.get.return_value = {
            'metadatas': [{
                ctx.constants.PAYLOAD_KEY: json.dumps({
                    "memories": {"phobia": "afraid of spiders"}
                }),
            }]
        }

        # Mock upsert method
        ctx.core_memory_adapter.upsert.return_value = None

        # Mock recall_memory_adapter
        ctx.recall_memory_adapter = MagicMock()
        ctx.recall_memory_adapter.add_memory.return_value = None
        ctx.recall_memory_adapter.query_memories.return_value = [{
            ctx.constants.PAYLOAD_KEY: "I like to swim."
        }]

        # Mock embeddings
        ctx.qdrant_memory_embeddings = MagicMock()
        ctx.qdrant_memory_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        ctx.qdrant_memory_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

        # Mock prompts
        ctx.agent_prompt = MockAgentPrompt()
        ctx.response_prompt = MockResponsePrompt()

        # Mock tokenizer
        ctx.tokenizer = MagicMock()
        ctx.tokenizer.encode = lambda x: x.split()  # Simple tokenizer mock
        ctx.tokenizer.decode = lambda x: ' '.join(x)
        yield ctx


@pytest.fixture(scope="session", autouse=True)
def mock_chroma_client():
    with patch("memory_langgraph.app_ctx.ctx.core_memory_adapter") as mock_client:
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        yield mock_client
