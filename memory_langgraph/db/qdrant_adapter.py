from typing import Any, Dict, List

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams


class QdrantAdapter:
    def __init__(self, client: AsyncQdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name
        self.initialized = False

    async def initialize(self):
        """Initialize the adapter if not already initialized."""
        if not self.initialized:
            await self._ensure_collection()
            self.initialized = True

    async def _ensure_collection(self):
        """Initialize collection if it doesn't exist."""
        collections = await self.client.get_collections()
        if self.collection_name not in [col.name for col in collections.collections]:
            await self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )

    async def add_memory(self, id: str, vector: List[float], metadata: Dict[str, Any], content: str):
        """Add a memory to the vector store."""
        await self.initialize()
        point = PointStruct(id=id, vector=vector, payload={**metadata, "content": content})
        await self.client.upsert(collection_name=self.collection_name, points=[point])

    async def query_memories(self, vector: List[float], user_id: str, n_results: int) -> List[Dict[str, Any]]:
        """Query memories by vector similarity."""
        await self.initialize()
        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=n_results,
            query_filter={
                "must": [
                    {"key": "user_id", "match": {"value": user_id}},
                    {"key": "type", "match": {"value": "recall"}}
                ]
            }
        )
        return [hit.payload for hit in results]
    
    async def close(self):
        await self.client.close()
