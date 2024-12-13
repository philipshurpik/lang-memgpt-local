from datetime import datetime, UTC
from typing import Dict
from motor.motor_asyncio import AsyncIOMotorClient


class MongoAdapter:
    def __init__(self, connection_url: str, db_name: str, core_collection: str):
        self.client = AsyncIOMotorClient(connection_url)
        self.db = self.client[db_name]
        self.memories = self.db[core_collection]

    async def save_memory(self, user_id: str, key: str, value: str) -> None:
        """Save a single memory value for a user."""
        await self.memories.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    key: value,
                    "updated_at": datetime.now(UTC)
                }
            },
            upsert=True
        )

    async def get_memories(self, user_id: str) -> Dict[str, str]:
        """Get memories for a user."""
        result = await self.memories.find_one(
            {"user_id": user_id},
            {"_id": 0, "user_id": 0, "updated_at": 0}
        )
        return result or {}

    async def close(self):
        self.client.close()
