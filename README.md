# Lang-MemGPT-Local: Local Memory-Enhanced Conversational AI

## Project Overview

Lang-MemGPT-Local is a refactored version of the original lang-memgpt project, designed to run locally and leverage ChromaDB for vector storage. This project implements a long-term memory system for AI assistants using LangChain and LangGraph, enabling more contextually aware and personalized conversations.

The core  idea behind Lang-MemGPT is to create an AI assistant that can truly understand and remember its users, going beyond simple recall to develop an evolving understanding of the user's preferences, habits, and history.

## Key Adaptations

1. **Local Deployment**: Removed dependencies on LangGraph Cloud, allowing the entire system to run on a local machine.
2. **ChromaDB Integration**: Replaced Pinecone with ChromaDB for vector storage, enabling local storage of embeddings and memories.
3. **Memory Retrieval**: Implemented robust memory querying and retrieval functions to fetch relevant past interactions.
4. **Error Handling**: Enhanced error handling and logging throughout the system for better debugging and stability.
5. **Anthropic API Compatibility**: Adjusted the chat interface to handle various response formats from the Anthropic API.

## Notes for Future Developers

1. **Memory Storage**: The current implementation uses ChromaDB for storing memories. If you need to scale up or use a different vector database, focus on modifying the `_utils.py` and `graph.py` files.

2. **API Compatibility**: The chat interface in `example_local.py` is designed to handle various response formats. If integrating with a new AI model API, ensure the `process_token` method in the `Chat` class can handle the new response format.

3. **Logging and Debugging**: Extensive logging has been implemented. Adjust logging levels in `example_local.py` as needed for your development process.

4. **LangSmith Integration**: The project includes LangSmith tracking, which can be disabled if not needed. See the logging configuration in `example_local.py`.

5. **Memory Types**: The system currently uses 'core' and 'recall' memory types. Expanding on these or adding new types would involve modifying the `graph.py` file and potentially the ChromaDB schema.