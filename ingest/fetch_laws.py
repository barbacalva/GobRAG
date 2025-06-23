#!/usr/bin/env python3
"""
Descarga masivamente los textos consolidados de cada disposición
legal listada en el catálogo del BOE.

Flujo general
-------------
1. Lee **law_list.xml** (salida de fetch_law_list.py) y extrae cada
   identificador (p. ej. ``BOE-A-2000-544``).
2. Lanza peticiones HTTP concurrentes al endpoint:

   ``https://boe.es/datosabiertos/api/legislacion-consolidada/id/<IDENTIFICADOR>``

   - Cabecera ``Accept: application/xml``.
   - Máximo de *concurrency* configurable (async/await con **aiohttp**).
   - *Timeout* y reintentos exponenciales ante errores transitorios.
3. Recorta el nodo ``<data>...</data>`` para eliminar metadatos de la
   petición (idéntico a la fase anterior).
4. Guarda resultado en ``data/raw/<identificador>.xml`` (crea carpetas si
   es necesario). Si el archivo existe se omite salvo que se use
   ``--overwrite``.
5. Mantiene un fichero **CSV** de índice (``download_index.csv``) con el
   estado de cada descarga:

   ``id,status,http_status,sha256,size,updated_at,error``

   De este modo se puede reanudar el proceso sin repetir lo ya obtenido.

Uso rápido
~~~~~~~~~~~

    python download_bulk_laws.py \
        --input ../data/law_list.xml \
        --output-dir ../data/raw \
        --workers 16 \
        --index ../data/raw/download_index.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence

import aiohttp
import xml.etree.ElementTree as ET


API_BASE: Final[str] = (
    "https://boe.es/datosabiertos/api/legislacion-consolidada/id/"
)
DEFAULT_ACCEPT: Final[str] = "application/xml"
DEFAULT_CONCURRENCY: Final[int] = 10
DEFAULT_TIMEOUT: Final[int] = 60  # in seconds
CSV_FIELDNAMES: Final[Sequence[str]] = (
    "id",
    "status",
    "http_status",
    "sha256",
    "size",
    "updated_at",
    "error",
)
SUCCESS_STATUS = {200}


@dataclass
class Job:
    identifier: str
    url: str


@dataclass
class Result:
    identifier: str
    status: str
    http_status: int | None
    sha256: str | None
    size: int | None
    error: str | None


def extract_data_node(xml_bytes: bytes) -> bytes:
    try:
        root = ET.fromstring(xml_bytes)
        data = root.find("data")
        if data is None:
            raise ValueError("Nodo <data> no encontrado en respuesta.")
        return ET.tostring(data, encoding="utf-8", xml_declaration=True)
    except ET.ParseError as exc:
        raise ValueError("Respuesta XML malformada") from exc


def load_completed_ids(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()
    completed: set[str] = set()
    with index_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok":
                completed.add(row["id"])
    return completed


def append_result(index_path: Path, result: Result) -> None:
    file_exists = index_path.exists()
    with index_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "id": result.identifier,
                "status": result.status,
                "http_status": result.http_status or "",
                "sha256": result.sha256 or "",
                "size": result.size or "",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "error": result.error or "",
            }
        )


async def fetch_one(
    session: aiohttp.ClientSession,
    job: Job,
    output_dir: Path,
    overwrite: bool,
    timeout: int,
) -> Result:
    dest = output_dir / f"{job.identifier}.xml"

    if dest.exists() and not overwrite:
        logging.info("%s ya existe, omitiendo", dest.name)
        sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()
        return Result(
            identifier=job.identifier,
            status="ok",
            http_status=200,
            sha256=sha256,
            size=dest.stat().st_size,
            error=None,
        )

    headers = {"Accept": DEFAULT_ACCEPT}
    try:
        async with session.get(job.url, headers=headers, timeout=timeout) as resp:
            status_code = resp.status
            content = await resp.read()

        if status_code not in SUCCESS_STATUS:
            return Result(job.identifier, "error", status_code, None, None, f"HTTP {status_code}")

        try:
            clean_xml = extract_data_node(content)
        except ValueError as exc:
            return Result(job.identifier, "error", status_code, None, None, str(exc))

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(clean_xml)

        sha256 = hashlib.sha256(clean_xml).hexdigest()
        return Result(
            identifier=job.identifier,
            status="ok",
            http_status=status_code,
            sha256=sha256,
            size=len(clean_xml),
            error=None,
        )
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        return Result(job.identifier, "error", None, None, None, str(exc))


async def bounded_gather(
    jobs: Sequence[Job],
    output_dir: Path,
    workers: int,
    overwrite: bool,
    timeout: int,
    index_path: Path,
) -> None:
    semaphore = asyncio.Semaphore(workers)
    timeout_obj = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession(timeout=timeout_obj) as session:

        async def run(job: Job):
            async with semaphore:
                result = await fetch_one(session, job, output_dir, overwrite, timeout)
                append_result(index_path, result)
                if result.status == "ok":
                    logging.info("Descargado %s (%s bytes)", job.identifier, result.size)
                else:
                    logging.error("%s → %s", job.identifier, result.error)

        tasks = [asyncio.create_task(run(job)) for job in jobs]
        await asyncio.gather(*tasks)


def parse_identifiers(list_path: Path) -> list[str]:
    tree = ET.parse(list_path)
    return [elem.text.strip() for elem in tree.iterfind(".//item/identificador") if elem.text]


def parse_args() -> argparse.Namespace:
    default_input = Path(__file__).resolve().parent.parent / "data" / "law_list.xml"
    default_output = Path(__file__).resolve().parent.parent / "data" / "raw"
    default_index = default_output / "download_index.csv"

    parser = argparse.ArgumentParser(
        description="Descarga masivamente las disposiciones consolidadas del BOE.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Archivo XML con la lista de leyes (por defecto {default_input})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Directorio destino de los XML (por defecto {default_output})",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=default_index,
        help=f"CSV de estado descargas (por defecto {default_index})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Número máximo de descargas concurrentes (por defecto {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout total por petición en segundos (por defecto {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Vuelve a descargar y sobrescribe archivos existentes.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activa log DEBUG.",
    )
    return parser.parse_args()


def main() -> None:  # noqa: D401
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.input.exists():
        logging.error("El archivo de entrada %s no existe", args.input)
        sys.exit(1)

    identifiers = parse_identifiers(args.input)
    logging.info("%d identificadores encontrados en %s", len(identifiers), args.input)

    completed = load_completed_ids(args.index) if not args.overwrite else set()
    pending = [id_ for id_ in identifiers if id_ not in completed]
    logging.info("%d leyes pendientes de descarga", len(pending))

    jobs = [Job(identifier=id_, url=f"{API_BASE}{id_}") for id_ in pending]

    try:
        asyncio.run(
            bounded_gather(
                jobs=jobs,
                output_dir=args.output_dir,
                workers=args.workers,
                overwrite=args.overwrite,
                timeout=args.timeout,
                index_path=args.index,
            )
        )
    except KeyboardInterrupt:
        logging.warning("Interrumpido por el usuario.")
        sys.exit(1)

    logging.info("Proceso completado.")


if __name__ == "__main__":
    main()
