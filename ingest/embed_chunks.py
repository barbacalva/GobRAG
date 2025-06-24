#!/usr/bin/env python3
"""
Fase 3: generación de embeddings y carga en **ChromaDB** en disco.

Entrada
~~~~~~~
* ``data/chunks/*.jsonl`` generados por *process_laws_chunk.py*.

Salida
~~~~~~
* Directorio ``vector_store/`` (o el indicado con ``--store``), con la
  colección persistente de Chroma.

Características
~~~~~~~~~~~~~~~
* **Batch** configurable (por defecto 512 textos).
* Modelo **open‑source** por defecto: ``jinaai/jina-embeddings-v2-base-es``.
* Autodetección del *device* («mps» en Apple Silicon, «cuda» o «cpu»).
* Reanudable: salta IDs ya presentes en la colección (``--skip-existing``).
* Registro en ``embed_index.csv`` con SHA256 y estado.

Uso rápido
~~~~~~~~~~

```bash
python embed_chunks.py \
  --chunk-dir ../data/chunks \
  --store ../vector_store \
  --collection boe_es_v1 \
  --batch 512 --device mps --workers 0
```
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Final, List

import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from gobrag.embedding import discover_device

CSV_FIELDNAMES: Final[list[str]] = [
    "id",
    "status",
    "sha256",
    "updated_at",
    "error",
]


def process_jsonl(
    jsonl_path: Path,
    collection: chromadb.api.models.Collection.Collection,
    model: SentenceTransformer,
    batch_size: int,
    skip_existing: bool,
    idx_writer_func,
):
    with jsonl_path.open("r", encoding="utf-8") as f:
        batch: list[dict[str, Any]] = []
        for line in f:
            data = json.loads(line)
            batch.append(data)
            if len(batch) >= batch_size:
                _flush(batch, collection, model, skip_existing, idx_writer_func)
                batch.clear()
        if batch:
            _flush(batch, collection, model, skip_existing, idx_writer_func)


def _flush(
    batch: List[dict[str, Any]],
    collection: chromadb.api.models.Collection.Collection,
    model: SentenceTransformer,
    skip_existing: bool,
    idx_writer_func,
) -> None:
    ids = [item["id"] for item in batch]

    if skip_existing:
        existing = set(
            collection.get(ids=ids, include=[]).get("ids", [])
        )
        if existing:
            batch = [item for item in batch if item["id"] not in existing]
            if not batch:
                return
        ids = [item["id"] for item in batch]

    texts = [item["text"] for item in batch]
    metadatas = [
        {
            "law_id": item["law_id"],
            "path": " / ".join(item["path"]),
            "kind": item["kind"],
            "version": item["version"],
            "url_eli": item["url_eli"],
        }
        for item in batch
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        truncation=True,
        max_length=216,
    )
    collection.upsert(ids=ids,
                      embeddings=embeddings,
                      metadatas=metadatas,
                      documents=texts)

    from datetime import datetime, timezone

    for i, item in enumerate(batch):
        idx_writer_func(
            {
                "id": item["id"],
                "status": "ok",
                "sha256": hashlib.sha256(texts[i].encode("utf-8")).hexdigest(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "error": "",
            }
        )


def get_idx_writer(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    f = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    def _write(row: dict[str, Any]):
        writer.writerow(row)
        f.flush()

    return _write


def parse_args():
    base = Path(__file__).resolve().parent.parent / "data"
    ap = argparse.ArgumentParser(
        description="Genera embeddings y los inserta en ChromaDB."
    )
    ap.add_argument(
        "--chunk-dir",
        type=Path,
        default=base / "chunks",
        help="Directorio con .jsonl de chunks."
    )
    ap.add_argument(
        "--store",
        type=Path,
        default=base / "vector_store",
        help="Ruta persistente de ChromaDB."
    )
    ap.add_argument(
        "--collection",
        type=str,
        default="boe_es_v1",
        help="Nombre de la colección."
    )
    ap.add_argument(
        "--model",
        type=str,
        default="jinaai/jina-embeddings-v2-base-es",
        help="Modelo sentence-transformers."
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=64,
        help="Tamaño de lote embedding."
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo forzado (cuda|mps|cpu)."
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="No procesa IDs ya en la colección."
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Log DEBUG."
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    device = args.device or discover_device()
    logging.info("Usando dispositivo: %s", device)

    model = SentenceTransformer(args.model, device=device, model_kwargs={"attn_implementation": "eager"})
    logging.info("Modelo %s cargado", args.model)

    client = chromadb.PersistentClient(path=str(args.store))
    collection = client.get_or_create_collection(args.collection)

    writer = get_idx_writer(args.store / "embed_index.csv")

    jsonl_files = sorted(args.chunk_dir.glob("*.jsonl"))
    logging.info("%d archivos .jsonl detectados", len(jsonl_files))

    for jsonl_path in tqdm(jsonl_files, desc="Procesando archivos"):
        logging.debug("Procesando %s", jsonl_path.name)
        process_jsonl(
            jsonl_path,
            collection,
            model,
            args.batch,
            args.skip_existing,
            writer,
        )

    logging.info("Embedding completado. Total vectores: %s", collection.count())


if __name__ == "__main__":
    main()
