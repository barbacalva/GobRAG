#!/usr/bin/env python3
"""
CLI minimalista para consultar el corpus legal indexado en ChromaDB y
obtener la respuesta de un modelo ChatGPT.

• Embedder:  carga el mismo modelo HF que usaste al generar la base
  (`jinaai/jina-embeddings-v2-base-es`) y lo ejecuta en MPS / CUDA / CPU.
• Vector store:  lee la colección persistente creada en la fase *embed*.
• Top-k:  recupera los k vecinos más cercanos y los pasa como contexto.
• Modelo de lenguaje:  llama a la API de OpenAI (usa `OPENAI_API_KEY` del
  entorno) pero se puede cambiar fácilmente a otro LLM.
• Citas:  imprime al final la fuente (ley, ruta, URL) de cada fragmento
  usado como contexto.

Uso rápido
----------
$ python -m gobrag.cli \
      "¿Qué dice la ley sobre los extranjeros en el artículo 23.2?" \
      --top-k 3
"""

from __future__ import annotations

import argparse
import sys
import textwrap

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from gobrag.config import settings
from gobrag.embedding import discover_device, embed
from gobrag.rag_core import build_prompt

WRAP = 120


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Consulta interactiva al sistema RAG sobre legislación BOE"
    )
    ap.add_argument("question", nargs="+", help="Pregunta en lenguaje natural")
    ap.add_argument(
        "--store",
        default=str(settings.vector_store_dir),
        help=f"Ruta a la base Chroma (defecto: {settings.vector_store_dir})",
    )
    ap.add_argument(
        "--collection",
        default=settings.collection_name,
        help=f"Nombre de la colección Chroma (defecto: {settings.collection_name})",
    )
    ap.add_argument(
        "--model",
        default=settings.embed_model,
        help=f"Modelo HF para embeddings (defecto: {settings.embed_model})",
    )
    ap.add_argument("--top-k", type=int, default=settings.top_k, help="Vecinos más cercanos")
    ap.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Dispositivo forzado",
    )
    ap.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature del ChatGPT"
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    question = " ".join(args.question)

    device = args.device or discover_device()
    model = SentenceTransformer(args.model, device=device)

    print(f"🔎  Embedding pregunta ({device}) …", file=sys.stderr)
    q_emb = embed(question, model)

    client = chromadb.PersistentClient(path=args.store)
    collection = client.get_collection(args.collection)

    print(f"🔍  Recuperando {args.top_k} fragmentos …", file=sys.stderr)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=args.top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    prompt = build_prompt(question, docs, metas)

    openai_key = settings.openai_api_key.get_secret_value()
    if not openai_key:
        sys.exit("ERROR: define OPENAI_API_KEY en el entorno.")

    llm = OpenAI(api_key=openai_key)
    print("💬  Consultando ChatGPT …", file=sys.stderr)
    completion = llm.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.temperature,
    )
    answer = completion.choices[0].message.content

    # Salida -----------------------------------------------------------------
    print("\n" + textwrap.fill(answer, width=WRAP) + "\n")
    print("📚  Fuentes:")
    for i, meta in enumerate(metas, 1):
        print(f" [{i}] {meta['law_id']} – {meta['path']} – {meta['url_eli']}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrumpido.")
