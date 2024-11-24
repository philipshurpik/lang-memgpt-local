import logging

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

logger = logging.getLogger("tools")
logger.setLevel(logging.INFO)

search_wrapper = TavilySearchResults(
    max_results=5,
    include_raw_content=True,
    include_images=False,
    search_depth="advanced"
)


@tool
def search_tavily(query: str) -> str:
    """Search the internet for information about a query.

    Args:
        query (str): The search query.

    Returns:
        str: Search results summary.
    """
    try:
        results = search_wrapper.run(query)
        return results
    except Exception as e:
        logger.error(f"Error in search_tool: {str(e)}")
        return f"Error performing search: {str(e)}"
