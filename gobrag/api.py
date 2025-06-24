#!/usr/bin/env python3

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from gobrag.config import OPENAI_API_KEY, TOP_K
from gobrag.rag_core import rag_stream

app = FastAPI(title="GobRAG API", version="0.1.0")


class QueryIn(BaseModel):
    question: str
    top_k: int | None = TOP_K


@app.get("/health", response_class=JSONResponse)
def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask(in_payload: QueryIn):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no definido")

    stream, metas = await rag_stream(in_payload.question, in_payload.top_k)

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
