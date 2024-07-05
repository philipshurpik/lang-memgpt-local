import json
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langsmith import get_current_run_tree, test

from lang_memgpt_local._constants import PATCH_PATH
from lang_memgpt_local._schemas import GraphConfig
from lang_memgpt_local.graph import memgraph

@test(output_keys=["num_mems_expected"])
@pytest.mark.parametrize(
    "messages, existing, num_mems_expected",
    [
        ([("user", "hi")], {}, 0),
        (
            [
                (
                    "user",
                    "When I was young, I had a dog named spot. He was my favorite pup. It's really one of my core memories.",
                )
            ],
            {},
            1,
        ),
        (
            [
                (
                    "user",
                    "When I was young, I had a dog named spot. It's really one of my core memories.",
                )
            ],
            {"memories": ["I am afraid of spiders."]},
            2,
        ),
    ],
)
async def test_patch_memory(
    messages: List[str],
    num_mems_expected: int,
    existing: dict,
    mock_chroma_client,
):
    user_id = "4fddb3ef-fcc9-4ef7-91b6-89e4a3efd112"
    thread_id = "e1d0b7f7-0a8b-4c5f-8c4b-8a6c9f6e5c7a"
    function_name = "CoreMemories"

    mock_collection = mock_chroma_client.return_value.get_or_create_collection.return_value

    # Set up existing memories
    if existing:
        path = PATCH_PATH.format(
            user_id=user_id,
            function_name=function_name,
        )
        mock_collection.get.return_value = {
            "metadatas": [{
                "content": json.dumps(existing)
            }]
        }
    else:
        mock_collection.get.return_value = {"metadatas": []}

    # When the memories are patched
    await memgraph.ainvoke(
        {
            "messages": messages,
        },
        {
            "configurable": GraphConfig(
                delay=0.1,
                user_id=user_id,
                thread_id=thread_id,
            ),
        },
    )

    if num_mems_expected:
        # Check if collection.upsert was called at least once
        assert mock_collection.upsert.call_count >= 1
        
        # Get the last call arguments
        last_call_args = mock_collection.upsert.call_args
        
        # Check the content of the last upsert
        upserted_metadata = last_call_args.kwargs['metadatas'][0]
        upserted_content = json.loads(upserted_metadata['content'])
        assert len(upserted_content['memories']) == num_mems_expected
        
        # Check if the new memory is in the upserted content
        new_memory = any("spot" in mem.lower() for mem in upserted_content['memories'])
        assert new_memory, "New memory about Spot should be in the upserted content"
        
        # If there was an existing memory, check if it's still there
        if existing:
            existing_memory = any("spiders" in mem.lower() for mem in upserted_content['memories'])
            assert existing_memory, "Existing memory about spiders should still be in the upserted content"

        rt = get_current_run_tree()
        rt.outputs = {"upserted": upserted_content['memories']}
    else:
        # If no memories are expected, ensure upsert wasn't called
        mock_collection.upsert.assert_not_called()

@test(output_keys=["num_events_expected"])
@pytest.mark.parametrize(
    "messages, num_events_expected",
    [
        ([("user", "hi")], 0),
        (
            [
                ("user", "I went to the beach with my friends today."),
                ("assistant", "That sounds like a fun day."),
                ("user", "You speak the truth."),
            ],
            1,
        ),
        (
            [
                ("user", "I went to the beach with my friends."),
                ("assistant", "That sounds like a fun day."),
                ("user", "I also went to the park with my family - I like the park."),
            ],
            1,
        ),
    ],
)
async def test_insert_memory(
    messages: List[str],
    num_events_expected: int,
    mock_chroma_client,
):
    user_id = "4fddb3ef-fcc9-4ef7-91b6-89e4a3efd112"
    thread_id = "e1d0b7f7-0a8b-4c5f-8c4b-8a6c9f6e5c7a"

    mock_collection = mock_chroma_client.return_value.get_or_create_collection.return_value
    mock_collection.get.return_value = {"metadatas": []}

    # When the events are inserted
    await memgraph.ainvoke(
        {
            "messages": messages,
        },
        {
            "configurable": GraphConfig(
                delay=0.1,
                user_id=user_id,
                thread_id=thread_id,
            ),
        },
    )

    if num_events_expected:
        # Check if collection.add was called at least num_events_expected times
        assert mock_collection.add.call_count >= num_events_expected