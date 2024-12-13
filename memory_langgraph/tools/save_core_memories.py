from langchain_core.runnables.config import ensure_config
from langchain_core.tools import tool

from ..app_ctx import ctx


@tool
async def save_core_memories(key: str, value: str) -> str:
    """Store or update a memory about the user. Use this to remember important information about the user.

    Args:
        key: The type of information to store. Use dot notation for nested information. Use singular form for keys
            Examples: 'name', 'age', 'preference.food', 'family.spouse.name', 'pet.dog.name'
        value: The value to remember.
            Examples: 'John', '25', 'pizza', 'Mary'

    Example usage:
        - save_core_memories('name', 'John')
        - save_core_memories('preference.food', 'pizza')
        - save_core_memories('family.children.count', '2')

    Returns:
        A confirmation message indicating the memory was stored.
    """
    config = ensure_config()
    user_id = ctx.ensure_configurable(config)["user_id"]
    
    await ctx.core_memory_adapter.save_memory(user_id, key, value)
    return f"Memory stored: {key} = {value}"
