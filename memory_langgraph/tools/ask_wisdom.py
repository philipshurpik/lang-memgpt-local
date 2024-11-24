import logging
import os

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

logger = logging.getLogger("tools")
logger.setLevel(logging.INFO)


@tool
def ask_wisdom(query: str) -> str:
    """Ask a question that can be answered using a knowledge base of human wisdom.

    Args:
        query (str): The search question sentence.

    Returns:
        str: Search results.
    """
    try:
        qdrant_vectorstore = QdrantVectorStore(
            client=QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY")
            ),
            collection_name=os.getenv("QDRANT_COLLECTION"),
            embedding=OpenAIEmbeddings(),
        )
        results = qdrant_vectorstore.similarity_search(query, k=5)
        formatted_results = "\n\n".join([doc.page_content.strip() for doc in results])
        return formatted_results

    except Exception as e:
        logger.error(f"Error in ask_wisdom: {str(e)}")
        return f"Error performing ask_wisdom: {str(e)}"
