from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorDBInterface(ABC):
    @abstractmethod
    def get_or_create_collection(self, name: str):
        pass

    @abstractmethod
    def add_memory(self, id: str, vector: List[float], metadata: Dict[str, Any], content: str):
        pass

    @abstractmethod
    def query_memories(self, vector: List[float], where: Dict[str, Any], n_results: int) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_collection(self, name: str):
        pass

    @abstractmethod
    def upsert(self, collection_name: str, ids: List[str], metadatas: List[Dict[str, Any]], documents: List[str]):
        pass