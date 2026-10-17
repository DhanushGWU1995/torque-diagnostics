#!/usr/bin/env python3
"""Write table text from the raw PDFs with pdfplumber.

The passage Parquet files stay as they are. This script reads each raw PDF
again and stores only tables whose cells are real text. A picture of a
table is left for the image script. From the repository root:

    INDEX_FAMILY=ACS880-11 MAX_FILES=20 uv run python data-preprocessing/tables/extract_tables.py

Each PDF under raw/<family>/<kind>/<document_id>.pdf is written to

    s3://dats-nlp/dhanush.mathivanan@gwu.edu/torque-diagnostics/tables/asset_family=<family>/<document_id>.parquet

A document with no ruled table still gets an empty Parquet file, so the
next run skips it. Leave MAX_FILES unset to process every remaining PDF.
INDEX_FAMILY limits the run to one equipment folder. This does not rebuild
the token files, the BM25 index, or the embedding index.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from raw_pdfs import BUCKET, REGION, discover, file_limit, write_parquet

FOLDER = "tables"
STATUS_NAME = "table_status.txt"
REMAINING_NAME = "table_remaining.txt"
FAILED_NAME = "table_failed.txt"

log = logging.getLogger("extract-tables")


def cell_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def table_to_text(table: list[list[object]]) -> str:
    lines: list[str] = []
    width = 0
    for row in table:
        cells = [cell_text(cell) for cell in row]
        width = max(width, sum(1 for cell in cells if cell))
        if any(cells):
            lines.append(" | ".join(cells))
    if len(lines) < 2 or width < 2:
        return ""
    return "\n".join(lines)


def tables_for_pdf(path: str, document_id: str, family: str) -> list[dict[str, object]]:
    import pdfplumber

    rows: list[dict[str, object]] = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            found = page.extract_tables() or []
            kept = 0
            for table in found:
                text = table_to_text(table or [])
                if not text:
                    continue
                rows.append(
                    {
                        "table_id": f"{document_id}:{page_number:04d}:{kept:03d}",
                        "document_id": document_id,
                        "asset_family": family,
                        "page_number": page_number,
                        "table_index": kept,
                        "text": text,
                    }
                )
                kept += 1
    return rows


def process_one(task) -> tuple[str, str, int, str]:
    import boto3

    pdf_path = ""
    parquet_path = ""
    try:
        s3 = boto3.client("s3", region_name=REGION)
        pdf_handle, pdf_path = tempfile.mkstemp(suffix=".pdf")
        os.close(pdf_handle)
        s3.download_file(BUCKET, task.pdf_key, pdf_path)
        rows = tables_for_pdf(pdf_path, task.document_id, task.family)
        parquet_handle, parquet_path = tempfile.mkstemp(suffix=".parquet")
        os.close(parquet_handle)
        import pyarrow as pa

        schema = pa.schema(
            [
                ("table_id", pa.string()),
                ("document_id", pa.string()),
                ("asset_family", pa.string()),
                ("page_number", pa.int32()),
                ("table_index", pa.int32()),
                ("text", pa.string()),
            ]
        )
        write_parquet(rows, parquet_path, schema)
        s3.upload_file(parquet_path, BUCKET, task.parquet_key)
        return task.pdf_key, "stored" if rows else "empty", len(rows), ""
    except Exception:
        return task.pdf_key, "failed", 0, traceback.format_exc()
    finally:
        for path in (pdf_path, parquet_path):
            if path and os.path.exists(path):
                os.remove(path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import boto3

    try:
        limit = file_limit()
    except ValueError as exc:
        log.error("%s", exc)
        return 1
    s3 = boto3.client("s3", region_name=REGION)
    pending, already = discover(s3, FOLDER)
    batch = pending if limit is None else pending[:limit]
    remaining = Path.cwd() / REMAINING_NAME
    failed_path = Path.cwd() / FAILED_NAME
    status_path = Path.cwd() / STATUS_NAME
    remaining.write_text("".join(f"{task.pdf_key}\n" for task in pending), encoding="utf-8")
    log.info("table_shards=%s remaining=%s this_run=%s", already, len(pending), len(batch))
    stored = empty = failed = 0
    for index, task in enumerate(batch, start=1):
        pdf_key, status, count, error = process_one(task)
        if status == "stored":
            stored += 1
        elif status == "empty":
            empty += 1
        else:
            failed += 1
            with failed_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{pdf_key}\n{error}\n")
        log.info("this_run=%s/%s %s tables=%s %s", index, len(batch), status, count, pdf_key)
        status_path.write_text(
            f"table_shards={already + stored + empty}\nremaining={len(pending) - index}\n"
            f"stored={stored}\nempty={empty}\nfailed={failed}\n",
            encoding="utf-8",
        )
    log.info("done stored=%s empty=%s failed=%s", stored, empty, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
