from typing import List, Dict, Tuple, AsyncIterator

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from gobrag.config import settings
from gobrag.vector_store import query_vector_store

openai_client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())


def build_prompt(question: str, docs: list[str], metas: list[Dict]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] {meta['law_id']} – {meta['path']}\n{doc.strip()}"
        for i, (doc, meta) in enumerate(zip(docs, metas))
    )
    return (
        "Eres un asistente jurídico. Responde en español SOLO con la información del contexto. "
        "Cita siempre la disposición de origen (Ley, art. X) y enlaza al BOE.\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pregunta: {question}\n\nRespuesta:"
    )


async def rag_stream(question: str, top_k: int) -> Tuple[AsyncIterator[ChatCompletionChunk], List[Dict]]:
    docs, metas = query_vector_store(
        question=question,
        top_k=top_k,
    )
    prompt = build_prompt(question, docs, metas)
    stream = await openai_client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.temperature,
        stream=True,
    )
    return stream, metas
