from pathlib import Path
from typing import List

from pydantic import AnyUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    embed_model: str = "jinaai/jina-embeddings-v2-base-es"
    openai_model: str = "gpt-3.5-turbo"
    temperature: float = 0.2
    top_k: int = 4

    vector_store_dir: Path = Path("data/vector_store")
    collection_name: str = "boe_es_v1"

    openai_api_key: SecretStr
    jina_api_key: SecretStr | None = None


    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GOBRAG_",
        case_sensitive=False,
    )


settings = Settings()
