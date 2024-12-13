import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.tools import tool

from ..app_ctx import ctx

logger = logging.getLogger("tools")
logger.setLevel(logging.INFO)


@tool
async def ask_wisdom(query: str) -> str:
    """Ask a question that can be answered using a knowledge base of human wisdom.

    Args:
        query (str): The search question sentence.

    Returns:
        str: Search results.
    """
    try:
        results: List[Document] = await ctx.qdrant_vectorstore.asimilarity_search(query, k=5)
        formatted_results = "\n\n".join([doc.page_content.strip() for doc in results])
        return formatted_results

    except Exception as e:
        logger.error(f"Error in ask_wisdom: {str(e)}")
        return f"Error performing ask_wisdom: {str(e)}"
