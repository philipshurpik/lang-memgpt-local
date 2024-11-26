import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from memory_langgraph.app_ctx import ctx, Constants


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
            'metadatas': [
                {
                    ctx.constants.PAYLOAD_KEY: json.dumps({
                        "memories": {"existing_memory": "I am afraid of spiders."}
                    })
                }
            ]
        }

        # Mock upsert method
        ctx.core_memory_adapter.upsert.return_value = None

        # Mock recall_memory_adapter
        ctx.recall_memory_adapter = MagicMock()
        ctx.recall_memory_adapter.save_memory.return_value = None
        ctx.recall_memory_adapter.query_memories.return_value = [
            {
                ctx.constants.PAYLOAD_KEY: "I went to the beach with my friends last summer."
            }
        ]

        # Mock embeddings
        ctx.qdrant_memory_embeddings = MagicMock()
        ctx.qdrant_memory_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        ctx.qdrant_memory_embeddings.aembed_query.return_value = [0.1, 0.2, 0.3]

        # Mock prompts
        ctx.prompts = {}

        # Create a mocked bound object for agent_llm
        class MockAgentBound:
            async def ainvoke(self, input_dict):
                last_user_message = input_dict["messages"][-1].content
                if "Spot" in last_user_message:
                    response_content = json.dumps({
                        "tool_calls": [
                            {
                                "tool": "save_core_memories",
                                "input": "favorite_pet|When I was young, I had a dog named Spot."
                            }
                        ],
                        "response": ""
                    })
                elif "beach" in last_user_message:
                    response_content = json.dumps({
                        "tool_calls": [
                            {
                                "tool": "save_recall_memory",
                                "input": "I went to the beach with my friends today."
                            }
                        ],
                        "response": ""
                    })
                else:
                    response_content = json.dumps({
                        "tool_calls": [],
                        "response": "Tell me more!"
                    })
                new_message = AIMessage(content=response_content)
                return input_dict["messages"] + [new_message]

        # Create a mocked bound object for response_llm
        class MockResponseBound:
            async def ainvoke(self, input_dict):
                # Return an AIMessage with content
                return AIMessage(content="This is a mocked final response.")

        # Mock the '|' operator to return the appropriate bound object
        class MockAgentPrompt:
            def __or__(self, other):
                return MockAgentBound()

        class MockResponsePrompt:
            def __or__(self, other):
                return MockResponseBound()

        ctx.prompts["agent"] = MockAgentPrompt()
        ctx.prompts["response"] = MockResponsePrompt()

        # Mock agent_model's bind_tools
        ctx.agent_model = MagicMock()
        ctx.agent_model.bind_tools.return_value = MagicMock()

        # Mock response_model (not used directly due to the prompt mock)
        ctx.response_model = MagicMock()

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
