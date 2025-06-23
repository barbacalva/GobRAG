#!/usr/bin/env python3
"""
Fase 2 de la *pipeline* RAG‑Legal: **parsing, limpieza y chunking** de las
normas descargadas en XML.

Entrada: «data/raw/*.xml» (salida de *download_bulk_laws.py*)
Salidas:
  • «data/md/<id>.md» → Versión Markdown legible (opcional).
  • «data/chunks/<id>.jsonl» → Chunks con metadatos listos para embeddings.
  • «chunk_index.csv» → Estado del procesamiento (resume).

Cada *chunk* conserva la jerarquía jurídica (Ley → Artículo → …) y metadatos
esenciales para el sistema FAQ/RAG.

Uso rápido
~~~~~~~~~~

    python process_laws_chunk.py \
        --raw-dir ../data/raw \
        --md-dir ../data/md \
        --chunk-dir ../data/chunks \
        --workers 4
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, List, Sequence, Set
from transformers import AutoTokenizer

import xml.etree.ElementTree as ET

DEFAULT_MAX_TOKENS: Final[int] = 512
DEFAULT_OVERLAP: Final[int] = 32
CSV_FIELDNAMES: Final[Sequence[str]] = (
    "id",
    "status",
    "chunks",
    "sha256_md",
    "sha256_jsonl",
    "updated_at",
    "error",
)
tok = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-es")


def count_tokens(text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


@dataclass
class Node:
    id: str
    label: str
    kind: str
    version: str
    text: str
    children: List["Node"]


def _latest_version(bloque: ET.Element) -> ET.Element | None:
    versions = list(bloque.findall("version"))
    if not versions:
        return None

    def _key(v: ET.Element) -> str:
        return v.attrib.get("fecha_vigencia") or v.attrib.get("fecha_publicacion") or "0"

    return max(versions, key=_key)


def _clean_text(xml_element: ET.Element, exclude_tags: Set[str] = None) -> str:
    parts: list[str] = []

    def _walk(e: ET.Element):
        if exclude_tags and e.tag in exclude_tags:
            if e.tail:
                parts.append(e.tail)
            return
        if e.text:
            parts.append(e.text)
        for child in e:
            _walk(child)
        if e.tail:
            parts.append(e.tail)

    _walk(xml_element)
    text = "".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def _build_tree(root: ET.Element, meta_title: str) -> Node:
    def visit(bloque: ET.Element) -> Node:
        version = _latest_version(bloque)
        label = (
                bloque.attrib.get("titulo")
                or bloque.attrib.get("id")
                or bloque.attrib.get("tipo")
        )
        kind = bloque.attrib.get("tipo", "bloque")

        if version is None:
            return Node(id=bloque.attrib.get("id", label), label=label, kind=kind, version="", text="", children=[])

        version_id = (
                version.attrib.get("id_norma")
                or bloque.attrib.get("fecha_vigencia")
                or bloque.attrib.get("fecha_publicacion")
        )
        note_nodes: list[Node] = []
        for idx, bq in enumerate(version.findall("blockquote"), 1):
            note_text = _clean_text(bq)
            if note_text:
                note_nodes.append(
                    Node(
                        id=f"{bloque.attrib.get('id', label)}_n{idx}",
                        label=f"Nota {idx}",
                        kind="note",
                        version=version_id,
                        text=note_text,
                        children=[],
                    )
                )
        text = _clean_text(version, exclude_tags={"blockquote"})
        children = [visit(child) for child in version.findall("bloque")] + note_nodes

        return Node(
            id=bloque.attrib.get("id", label),
            label=label,
            kind=kind,
            version=version_id,
            text=text,
            children=children,
        )

    texto_elem = root.find("texto")
    if texto_elem is None:
        raise ValueError("XML sin nodo <texto>")

    top_children = [visit(b) for b in texto_elem.findall("bloque")]
    return Node(id="root", label=meta_title, kind="ley", version="", text="", children=top_children)


def _chunk_node(node: Node, path: list[str], chunks: list[dict[str, Any]],
                max_tokens: int, overlap: int) -> None:
    new_path = path + [node.label]

    if node.text.strip():
        _split_and_append(node, new_path, chunks, max_tokens, overlap)
    for child in node.children:
        _chunk_node(child, new_path, chunks, max_tokens, overlap)


def _split_and_append(node: Node, path: list[str], chunks: list[dict[str, Any]],
                      max_tokens: int, overlap: int) -> None:
    tokens = tok.encode(node.text, add_special_tokens=False)
    step = max_tokens - overlap
    for i in range(0, len(tokens), step):
        tokens_segment = tokens[i: i + max_tokens]
        text_seg = tok.decode(tokens_segment)
        if not text_seg:
            continue
        chunk_id = f"{node.id}_{i // step}" if i else node.id
        token_len = count_tokens(text_seg)
        chunks.append({
            "id": chunk_id,
            "path": path,
            "kind": node.kind,
            "version": node.version,
            "text": text_seg,
            "tokens": token_len,
        })


def process_file(path: Path, md_dir: Path, chunk_dir: Path,
                 max_tokens: int, overlap: int) -> tuple[str, str | None, str, str, int]:
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        meta = root.find("metadatos")
        if meta is None:
            raise ValueError("No se encontró <metadatos>")
        law_id = meta.findtext("identificador") or path.stem
        law_title = meta.findtext("titulo") or law_id
        url_eli = meta.findtext("url_eli")

        tree_root = _build_tree(root, law_title)

        chunks: list[dict[str, Any]] = []
        _chunk_node(tree_root, [], chunks, max_tokens=max_tokens, overlap=overlap)

        for ch in chunks:
            ch.update({
                "id": law_id + "_" + ch["id"],
                "law_id": law_id,
                "url_eli": url_eli,
            })

        chunk_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = chunk_dir / f"{law_id}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f_jsonl:
            for ch in chunks:
                f_jsonl.write(json.dumps(ch, ensure_ascii=False) + "\n")

        md_dir.mkdir(parents=True, exist_ok=True)
        md_path = md_dir / f"{law_id}.md"
        with md_path.open("w", encoding="utf-8") as fmd:
            fmd.write(f"# {law_title}\n\n")
            for ch in chunks:
                heading = " / ".join(ch["path"])
                fmd.write(f"## {heading}\n\n{ch['text']}\n\n")

        sha_md = hashlib.sha256(md_path.read_bytes()).hexdigest()
        sha_jsonl = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
        return law_id, None, sha_md, sha_jsonl, len(chunks)

    except Exception as exc:
        return path.stem, str(exc), "", "", 0


def append_index(idx_path: Path, row: dict[str, Any]) -> None:
    file_exists = idx_path.exists()
    with idx_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def process_law_files(xml_files: list[Path],
                      idx_path: Path,
                      md_dir: Path,
                      chunk_dir: Path,
                      max_tokens: int,
                      overlap: int,
                      workers: int):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_file, p, md_dir, chunk_dir, max_tokens, overlap): p
            for p in xml_files
        }
        for fut in as_completed(futures):
            law_id, error, sha_md, sha_jsonl, n_chunks = fut.result()
            status = "ok" if error is None else "error"
            append_index(idx_path, {
                "id": law_id,
                "status": status,
                "chunks": n_chunks,
                "sha256_md": sha_md,
                "sha256_jsonl": sha_jsonl,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "error": error or "",
            })
            if status == "ok":
                logging.info("Procesado %s (%s chunks)", law_id, n_chunks)
            else:
                logging.error("%s → %s", law_id, error)


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent.parent / "data"
    parser = argparse.ArgumentParser(
        description="Parsea, limpia y trocea las leyes XML en chunks."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=base / "raw",
        help="Directorio de XML sin procesar."
    )
    parser.add_argument(
        "--md-dir",
        type=Path,
        default=base / "md",
        help="Salida Markdown."
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=base / "chunks",
        help="Salida JSONL de chunks."
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=base / "chunks" / "chunk_index.csv",
        help="CSV de estado."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Procesos paralelos."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Máximo tokens por chunk."
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="Solapamiento tokens entre chunks."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Nivel DEBUG."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s: %(message)s")

    xml_files = sorted(args.raw_dir.glob("*.xml"))
    logging.info("%d archivos XML encontrados en %s", len(xml_files), args.raw_dir)

    process_law_files(xml_files,
                      args.index,
                      args.md_dir,
                      args.chunk_dir,
                      args.max_tokens,
                      args.overlap,
                      args.workers)

    logging.info("Pipeline de chunking finalizada.")


if __name__ == "__main__":
    main()
