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
$ export OPENAI_API_KEY=sk-xxxxx
$ python rag_cli.py \
      "¿Qué dice la ley sobre los extranjeros en el artículo 23.2?" \
      --top-k 3
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from typing import List

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "jinaai/jina-embeddings-v2-base-es"
DEFAULT_STORE = "data/vector_store"
DEFAULT_COLLECTION = "boe_es_v1"
WRAP = 120


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


def embed_query(model: SentenceTransformer, text: str) -> List[float]:
    return model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        max_seq_length=512,
    )[0].tolist()


def build_prompt(question: str, docs: list[str], metas: list[dict]) -> str:
    context_lines = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        context_lines.append(
            f"[{i}] {meta['law_id']} – {meta['path']}\n{doc.strip()}"
        )

    context_str = "\n\n".join(context_lines)
    prompt = (
        "Eres un asistente jurídico. "
        "Responde en español **sólo** con la información de contexto. "
        "Cita siempre la disposición de origen (Ley, art. X) y enlaza al BOE.\n\n"
        f"Contexto:\n{context_str}\n\n"
        f"Pregunta: {question}\n\nRespuesta:"
    )
    return prompt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Consulta interactiva al sistema RAG sobre legislación BOE"
    )
    ap.add_argument("question", nargs="+", help="Pregunta en lenguaje natural")
    ap.add_argument(
        "--store",
        default=DEFAULT_STORE,
        help=f"Ruta a la base Chroma (defecto: {DEFAULT_STORE})",
    )
    ap.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Nombre de la colección Chroma (defecto: {DEFAULT_COLLECTION})",
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modelo HF para embeddings (defecto: {DEFAULT_MODEL})",
    )
    ap.add_argument("--top-k", type=int, default=4, help="Vecinos más cercanos")
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
    q_emb = embed_query(model, question)

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

    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        sys.exit("ERROR: define OPENAI_API_KEY en el entorno.")

    llm = OpenAI(api_key=openai_key)
    print("💬  Consultando ChatGPT …", file=sys.stderr)
    completion = llm.chat.completions.create(
        model="gpt-3.5-turbo",
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
