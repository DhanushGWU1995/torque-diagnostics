#!/usr/bin/env python3
"""Write table text from the raw PDFs with PyMuPDF.

The passage Parquet files stay as they are. This script reads each raw PDF
again and stores only ruled tables whose cells are real text. A picture of a
table is left for the image script. From the repository root:

    INDEX_FAMILY=ACS880-11 MAX_FILES=20 uv run python data-preprocessing/tables/extract_tables.py

Each PDF under raw/<family>/<kind>/<document_id>.pdf is written to

    s3://dats-nlp/dhanush.mathivanan@gwu.edu/torque-diagnostics/chunks/asset_family=<family>/tables/<document_id>.parquet

A document with no ruled table still gets an empty Parquet file, so the
next run skips it. Finished table files are left alone. Two worker
processes each read a different PDF. A page that raises, or that runs
longer than PAGE_SECONDS (default 30), is skipped and the rest of that PDF
continues. TABLE_WORKERS changes the pool size.
Leave MAX_FILES unset to process every remaining PDF. INDEX_FAMILY limits
the run to one equipment folder. This does not rebuild the token files, the
BM25 index, or the embedding index. After the files exist, rerun those
three with REBUILD=1 so the reply can search the tables.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import sys
import tempfile
import traceback
from contextlib import contextmanager
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from raw_pdfs import BUCKET, REGION, discover, file_limit, write_parquet

FOLDER = "tables"
STATUS_NAME = "table_status.txt"
REMAINING_NAME = "table_remaining.txt"
FAILED_NAME = "table_failed.txt"
PAGE_SECONDS = int(os.environ.get("PAGE_SECONDS", "30"))
WORKERS = int(os.environ.get("TABLE_WORKERS", "2"))

log = logging.getLogger("extract-tables")


class PageTimeout(BaseException):
    """BaseException so PyMuPDF's own except Exception cannot hide a slow page."""


def _on_alarm(signum, frame) -> None:
    raise PageTimeout()


@contextmanager
def page_limit(seconds: int):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return
    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


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


def rows_from_tables(found: list, document_id: str, family: str, page_number: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
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


def cells_on_page(page) -> list:
    """Ruled-table cells only. use_layout=False keeps line detection on older calls."""
    try:
        finder = page.find_tables(use_layout=False)
    except TypeError:
        finder = page.find_tables()
    if finder is None:
        return []
    tables = getattr(finder, "tables", None)
    if tables is None:
        tables = list(finder)
    found = []
    for table in tables:
        extracted = table.extract()
        if extracted:
            found.append(extracted)
    return found


def tables_for_pdf(path: str, document_id: str, family: str) -> tuple[list[dict[str, object]], list[int]]:
    import pymupdf

    rows: list[dict[str, object]] = []
    skipped: list[int] = []
    document = pymupdf.open(path)
    try:
        total = document.page_count
        page_number = 1
        while page_number <= total:
            try:
                with page_limit(PAGE_SECONDS):
                    found = cells_on_page(document[page_number - 1])
            except PageTimeout:
                skipped.append(page_number)
                log.info("skip page %s %s", page_number, document_id)
                document.close()
                document = pymupdf.open(path)
                page_number += 1
                continue
            except Exception:
                log.info("skip page %s %s", page_number, document_id, exc_info=True)
                skipped.append(page_number)
                page_number += 1
                continue
            rows.extend(rows_from_tables(found, document_id, family, page_number))
            page_number += 1
    finally:
        document.close()
    return rows, skipped


def process_one(task) -> tuple[str, str, int, str]:
    import boto3

    pdf_path = ""
    parquet_path = ""
    try:
        s3 = boto3.client("s3", region_name=REGION)
        pdf_handle, pdf_path = tempfile.mkstemp(suffix=".pdf")
        os.close(pdf_handle)
        s3.download_file(BUCKET, task.pdf_key, pdf_path)
        rows, skipped = tables_for_pdf(pdf_path, task.document_id, task.family)
        note = ""
        if skipped:
            note = "skipped pages " + ",".join(str(number) for number in skipped)
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
        return task.pdf_key, "stored" if rows else "empty", len(rows), note
    except Exception:
        return task.pdf_key, "failed", 0, traceback.format_exc()
    finally:
        for path in (pdf_path, parquet_path):
            if path and os.path.exists(path):
                os.remove(path)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main() -> int:
    configure_logging()
    import boto3

    if WORKERS < 1:
        log.error("TABLE_WORKERS must be a positive integer")
        return 1
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
    log.info("reader=pymupdf workers=%s page_seconds=%s", WORKERS, PAGE_SECONDS)
    stored = empty = failed = 0
    if not batch:
        log.info("done stored=%s empty=%s failed=%s", stored, empty, failed)
        return 0

    def record(pdf_key: str, status: str, count: int, error: str, index: int) -> None:
        nonlocal stored, empty, failed
        if status == "stored":
            stored += 1
        elif status == "empty":
            empty += 1
        else:
            failed += 1
        if status == "failed" or error:
            with failed_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{pdf_key}\n{error}\n")
        log.info(
            "this_run=%s/%s %s tables=%s %s %s",
            index,
            len(batch),
            status,
            count,
            pdf_key,
            error,
        )
        status_path.write_text(
            f"table_shards={already + stored + empty}\nremaining={len(pending) - index}\n"
            f"stored={stored}\nempty={empty}\nfailed={failed}\n",
            encoding="utf-8",
        )

    context = get_context("spawn")
    with context.Pool(WORKERS, initializer=configure_logging) as pool:
        for index, result in enumerate(pool.imap_unordered(process_one, batch), start=1):
            record(*result, index)
    log.info("done stored=%s empty=%s failed=%s", stored, empty, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
