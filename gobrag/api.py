#!/usr/bin/env python3

from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

from gobrag.config import settings
from gobrag.embedding import get_embedder
from gobrag.middlewares import attach_middlewares
from gobrag.rag_core import rag_stream
from gobrag.vector_store import get_collection


app = FastAPI(title="GobRAG API", version="0.1.0")
attach_middlewares(app)


class QueryIn(BaseModel):
    question: str
    top_k: int | None = settings.top_k


@app.get("/health", tags=["health"], response_class=JSONResponse)
def health():
    return {"status": "ok"}


@app.get("/ready", tags=["health"])
async def readiness():
    checks = {}

    try:
        _ = get_embedder()
        checks["embedder"] = "ok"
    except Exception as e:
        checks["embedder"] = str(e)

    try:
        col = get_collection()
        _ = col.count()
        checks["chroma"] = "ok"
    except Exception as e:
        checks["chroma"] = str(e)

    try:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY undefined")
        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        await asyncio.wait_for(client.models.list(), timeout=2.0)
        checks["openai"] = "ok"
    except Exception as e:
        checks["openai"] = str(e)

    if all(v == "ok" for v in checks.values()):
        return checks
    return JSONResponse(status_code=503, content=checks)


@app.post("/ask")
async def ask(in_payload: QueryIn):
    stream, _ = await rag_stream(in_payload.question, in_payload.top_k)

    async def generator():
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {delta}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
    )
