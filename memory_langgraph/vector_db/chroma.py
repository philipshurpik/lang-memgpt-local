from typing import List, Dict, Any

import chromadb

from .interface import VectorDBInterface


class ChromaAdapter(VectorDBInterface):
    def __init__(self, persist_directory: str = "./vectordb", memory_collection="memories"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.memory_collection = memory_collection
        self.collections = {}

    def get_or_create_collection(self, name: str):
        if name not in self.collections:
            self.collections[name] = self.client.get_or_create_collection(name)
        return self.collections[name]

    def add_memory(self, id: str, vector: List[float], metadata: Dict[str, Any], content: str):
        collection = self.get_or_create_collection(self.memory_collection)
        collection.add(ids=[id], embeddings=[vector], metadatas=[metadata], documents=[content])

    def query_memories(self, vector: List[float], where: Dict[str, Any], n_results: int) -> List[Dict[str, Any]]:
        collection = self.get_or_create_collection(self.memory_collection)
        results = collection.query(query_embeddings=[vector], where=where, n_results=n_results)
        return results['metadatas'][0] if results['metadatas'] else []

    def get_collection(self, name: str):
        return self.get_or_create_collection(name)

    def upsert(self, collection_name: str, ids: List[str], metadatas: List[Dict[str, Any]], documents: List[str]):
        collection = self.get_or_create_collection(collection_name)
        collection.upsert(ids=ids, metadatas=metadatas, documents=documents)
