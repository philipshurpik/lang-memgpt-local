from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue, VectorParams, Distance
from .interface import VectorDBInterface


class QdrantAdapter(VectorDBInterface):
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name
        self.get_or_create_collection(collection_name)

    def get_or_create_collection(self, name: str):
        existing_collections = [col.name for col in self.client.get_collections().collections]
        if name not in existing_collections:
            self.client.recreate_collection(
                collection_name=name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
        return name

    def add_memory(self, id: str, vector: List[float], metadata: Dict[str, Any], content: str):
        point = PointStruct(
            id=id,
            vector=vector,
            payload=metadata
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

    def query_memories(self, vector: List[float], where: Dict[str, Any], n_results: int) -> List[Dict[str, Any]]:
        filters = self._build_filter(where)
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=n_results,
            query_filter=filters,
        )
        return [hit.payload for hit in search_result]

    def get_collection(self, name: str):
        return self.client.get_collection(collection_name=name)

    def upsert(self, collection_name: str, ids: List[str], metadatas: List[Dict[str, Any]], documents: List[str]):
        # This method can be implemented if needed
        pass

    def _build_filter(self, where_clause: Dict[str, Any]) -> Filter:
        if '$and' in where_clause:
            conditions = [self._build_filter(condition) for condition in where_clause['$and']]
            return Filter(must=conditions)
        elif '$or' in where_clause:
            conditions = [self._build_filter(condition) for condition in where_clause['$or']]
            return Filter(should=conditions)
        else:
            field, condition = next(iter(where_clause.items()))
            operator, value = next(iter(condition.items()))
            if operator == '$eq':
                return Filter(
                    must=[
                        FieldCondition(
                            key=field,
                            match=MatchValue(value=value)
                        )
                    ]
                )
            else:
                raise NotImplementedError(f"Operator {operator} not implemented in filter builder.")