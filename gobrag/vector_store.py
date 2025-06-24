from functools import lru_cache

import chromadb

from gobrag.config import VECTOR_STORE_DIR, COLLECTION_NAME
from gobrag.embedding import embed


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    return client.get_collection(COLLECTION_NAME)


def query_vector_store(question: str, top_k: int):
    vec = embed(question)
    res = get_collection().query(
        query_embeddings=[vec],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    return res["documents"][0], res["metadatas"][0]
