#!/usr/bin/env python3
"""
Descarga el listado de legislación consolidada del Boletín Oficial del
Estado (BOE) a través de la API de Datos Abiertos y lo guarda en un
archivo XML.

Este script está pensado para ser utilizado tanto desde la línea de
comandos (CLI) como importado desde otros módulos o tareas de un
*Makefile*.

Uso rápido::

    python download_law_list.py # descarga a ../data/law_list.xml
    python download_law_list.py -o /tmp/leyes.xml --overwrite
"""
from __future__ import annotations

import argparse
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final

import requests

__all__ = [
    "API_URL",
    "DEFAULT_PARAMS",
    "DEFAULT_ACCEPT",
    "fetch_law_list",
    "extract_data_node",
    "save_file",
    "parse_args",
    "main",
]

API_URL: Final[str] = "https://boe.es/datosabiertos/api/legislacion-consolidada"
DEFAULT_PARAMS: Final[dict[str, str]] = {"limit": "-1"}
DEFAULT_ACCEPT: Final[str] = "application/xml"
DEFAULT_TIMEOUT: Final[float] = 30.0  # in seconds


def fetch_law_list(timeout: float = DEFAULT_TIMEOUT) -> bytes:
    headers = {
        "Accept": DEFAULT_ACCEPT
    }
    logging.debug("Enviando solicitud GET a %s con params %s", API_URL, DEFAULT_PARAMS)
    with requests.get(
        API_URL,
        params=DEFAULT_PARAMS,
        headers=headers,
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        ctype = response.headers.get("Content-Type", "")
        if "xml" not in ctype:
            raise ValueError(
                f"Se esperaba XML pero se obtuvo Content-Type='{ctype}'."
            )
        logging.info("Respuesta %s %s (%s bytes)", response.status_code, response.reason,
                     response.headers.get("Content-Length", "desconocido"))
        return response.content


def extract_data_node(xml_bytes: bytes) -> bytes:
    root = ET.fromstring(xml_bytes)
    data = root.find("data")
    if data is None:
        raise ValueError("No se ha encontrado el nodo <data>.")

    return ET.tostring(
        data,
        encoding="utf-8",
        xml_declaration=True,
    )


def save_file(content: bytes, path: Path, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"El archivo {path} ya existe. Utiliza --overwrite para reemplazarlo."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    logging.info("Guardado %d bytes en %s", len(content), path)


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parent.parent / "data" / "law_list.xml"

    parser = argparse.ArgumentParser(
        description=(
            "Descarga la lista de legislación consolidada disponible en el BOE "
            "y la guarda en un archivo XML."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_output,
        help=f"Archivo de salida (por defecto: {default_output})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribe el archivo de destino si ya existe.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout de red en segundos (por defecto {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activa nivel de log DEBUG para diagnóstico detallado.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        logging.debug("Iniciando descarga...")
        xml_content = fetch_law_list(timeout=args.timeout)
        xml_data = extract_data_node(xml_content)
        save_file(xml_data, args.output, overwrite=args.overwrite)
    except (requests.RequestException, FileExistsError, OSError, ValueError) as exc:
        logging.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
