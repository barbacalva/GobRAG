from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

DATA_DIR = Path("data")
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
COLLECTION_NAME = "boe_es_v1"

EMBED_MODEL = "jinaai/jina-embeddings-v2-base-es"
OPENAI_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.2
TOP_K = 4

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")


def _parse_origins(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [o.strip() for o in raw.split(",") if o.strip()]


CORS_ALLOW_ORIGINS: list[str] = _parse_origins(
    os.getenv("CORS_ALLOW_ORIGINS", "")
)
