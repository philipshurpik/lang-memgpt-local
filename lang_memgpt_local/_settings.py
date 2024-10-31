from pydantic_settings import BaseSettings
from typing import Dict, Any

class Settings(BaseSettings):
    vectordb_class: str = "lang_memgpt_local.adapters.chroma.ChromaAdapter"
    vectordb_config: Dict[str, Any] = {"persist_directory": "./vectordb"}
    model: str = "gpt-4o-mini"

SETTINGS = Settings()
