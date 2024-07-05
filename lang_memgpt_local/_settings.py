from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    chroma_persist_directory: str = "./chroma_db"
    model: str = "claude-3-5-sonnet-20240620"

SETTINGS = Settings()
