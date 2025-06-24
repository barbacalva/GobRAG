from functools import lru_cache
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from gobrag.config import EMBED_MODEL


def discover_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(
        EMBED_MODEL,
        device=discover_device()
    )


def embed(text: str, model: Optional[SentenceTransformer] = None) -> List[float]:
    if model is None:
        model = get_embedder()
    return model.encode(
        [text],
        normalize_embeddings=True,
        max_seq_length=512,
    )[0].tolist()
