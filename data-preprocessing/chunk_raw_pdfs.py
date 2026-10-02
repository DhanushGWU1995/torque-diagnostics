#!/usr/bin/env python3
"""Chunk raw ABB PDFs from S3 into Parquet shards.

Credentials come from the environment or the instance role. This file does
not contain an access key. From a clone of the repo, at the repository root:

    uv sync
    uv run python data-preprocessing/chunk_raw_pdfs.py

On the EC2 instance, after Tesseract is installed:

    sudo apt-get install -y tesseract-ocr
    python3 chunk_raw_pdfs.py

Set MAX_FILES below, or pass it in the environment. A run checks that
many raw PDFs, converts the ones that do not already have a Parquet
shard, and then stops so a long job can finish in safe batches. The
keys still left are rewritten to chunk_remaining.txt. A shard is
skipped only when its Parquet footer is intact. A listed object that
is truncated or still incomplete is converted again. Progress counts
are written to chunk_status.txt. Leave MAX_FILES unset to process every
PDF that does not already have a shard.

Each PDF under

    s3://dats-nlp/dhanush.mathivanan@gwu.edu/torque-diagnostics/raw/<family>/<document-kind>/<document_id>.pdf

is written to

    s3://dats-nlp/dhanush.mathivanan@gwu.edu/torque-diagnostics/chunks/asset_family=<family>/<document_id>.parquet

The text layer is used when a page has text. OCR runs only when that layer
is nearly empty. A passage never crosses a page. A PDF of 200 MB or more is
chunked alone. Smaller PDFs run two at a time, in separate processes.
A shard is skipped only when the object in S3 is a finished Parquet
file. An incomplete object is written again.
"""

from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import io
import logging
import re
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import NamedTuple

BUCKET = "dats-nlp"
REGION = "us-east-1"
RAW_ROOT = "dhanush.mathivanan@gwu.edu/torque-diagnostics/raw"
CHUNKS_ROOT = "dhanush.mathivanan@gwu.edu/torque-diagnostics/chunks"
WORKERS = 2
# None processes every PDF that does not already have a Parquet shard.
# A positive number is one safe batch: that many remaining PDFs are
# checked, converted, and then the run stops. The next run continues
# with whatever chunk_remaining.txt still lists. The environment
# variable MAX_FILES overrides this value.
MAX_FILES: int | None = None
LARGE_FILE_BYTES = 200 * 1024 * 1024
MIN_TEXT_CHARS = 40
MAX_PASSAGE_CHARS = 3500
OCR_DPI = 150
TASKS_PER_CHILD = 25
FAILED_NAME = "chunk_failed.txt"
STATUS_NAME = "chunk_status.txt"
REMAINING_NAME = "chunk_remaining.txt"
FAMILY_ORDER = (
    "ACS880-04XT",
    "ACS880-104",
    "ACS880-107",
    "ACS880-01",
    "ACS880-04",
    "ACS880-07",
    "ACS880-11",
    "ACS880-14",
    "ACS880-17",
    "ACS880-31",
    "ACS880-34",
    "ACS880-37",
    "firmware-YINLX",
    "firmware-AINLX",
    "library",
)

FAULT_RE = re.compile(
    r"\b(?:fault|warning|alarm|aux(?:iliary)?\s+code|event)"
    r"\s*(?:code\s*)?[:\-]?\s*(\d{3,5})\b"
    r"(?!\s*(?:v|kv|kw|kva|hz|mm|cm|a)\b)",
    re.IGNORECASE,
)
PART_RE = re.compile(
    r"\b(?:part(?:\s+number)?|ordering\s+code|type\s+designation)"
    r"\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-]{4,40})",
    re.IGNORECASE,
)
PROCEDURE_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?"
    r"((?:procedure|start-up procedure|startup procedure|"
    r"maintenance procedure|replacement procedure)[^\n]{0,80})",
    re.IGNORECASE | re.MULTILINE,
)
TRACING_RE = re.compile(
    r"\b(?:circuit diagram|wiring diagram|terminal diagram|"
    r"connection diagram|tracing)\b",
    re.IGNORECASE,
)

log = logging.getLogger("chunk-raw")


class Task(NamedTuple):
    pdf_key: str
    size: int
    family: str
    document_id: str
    parquet_key: str
    batch_id: str


class Result(NamedTuple):
    pdf_key: str
    status: str
    pages: int
    passages: int
    error: str


def runs_alone(size: int) -> bool:
    return size >= LARGE_FILE_BYTES


def family_rank(family: str) -> tuple[int, str]:
    if family == "library":
        return (10_000, family)
    try:
        return (FAMILY_ORDER.index(family), family)
    except ValueError:
        return (9_000, family)


def parse_raw_key(key: str) -> tuple[str, str] | None:
    prefix = f"{RAW_ROOT}/"
    if not key.startswith(prefix) or not key.lower().endswith(".pdf"):
        return None
    parts = key[len(prefix) :].split("/")
    if len(parts) != 3 or not parts[0] or not parts[2]:
        return None
    family, _kind, filename = parts
    document_id = filename[: -len(".pdf")] if filename.lower().endswith(".pdf") else filename
    if not document_id:
        return None
    return family, document_id


def parquet_key(family: str, document_id: str) -> str:
    return f"{CHUNKS_ROOT}/asset_family={family}/{document_id}.parquet"


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_passages(text: str) -> list[str]:
    if len(text) <= MAX_PASSAGE_CHARS:
        return [text] if text else []
    passages: list[str] = []
    buffer = ""
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > MAX_PASSAGE_CHARS:
            if buffer:
                passages.append(buffer)
                buffer = ""
            for start in range(0, len(paragraph), MAX_PASSAGE_CHARS):
                passages.append(paragraph[start : start + MAX_PASSAGE_CHARS])
            continue
        if buffer and len(buffer) + 2 + len(paragraph) > MAX_PASSAGE_CHARS:
            passages.append(buffer)
            buffer = paragraph
        elif buffer:
            buffer = f"{buffer}\n\n{paragraph}"
        else:
            buffer = paragraph
    if buffer:
        passages.append(buffer)
    return passages


def joined_codes(values: list[str], limit: int) -> str | None:
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
        if len(unique) == limit:
            break
    return "|".join(unique) or None


def fault_codes(text: str) -> str | None:
    return joined_codes(FAULT_RE.findall(text), 20)


def part_codes(text: str, document_id: str) -> str | None:
    own = re.sub(r"[\s_\-]", "", document_id).upper()
    found: list[str] = []
    for token in PART_RE.findall(text):
        if not re.search(r"\d", token):
            continue
        if re.sub(r"[\s_\-]", "", token).upper() == own:
            continue
        found.append(token.upper())
    return joined_codes(found, 10)


def procedure_name(text: str) -> str | None:
    match = PROCEDURE_RE.search(text)
    if match is None:
        return None
    return match.group(1).strip()[:120] or None


def rows_for_document(
    document_id: str,
    family: str,
    pages: list[tuple[int, str]],
    batch_id: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for page_number, raw_text in pages:
        text = clean_text(raw_text)
        tracing = bool(TRACING_RE.search(text))
        for index, passage in enumerate(split_passages(text)):
            rows.append(
                {
                    "chunk_id": f"{document_id}:{page_number:04d}:{index:03d}",
                    "document_id": document_id,
                    "asset_family": family,
                    "page_number": page_number,
                    "tracing_page": tracing,
                    "fault_code": fault_codes(passage),
                    "part": part_codes(passage, document_id),
                    "procedure": procedure_name(passage),
                    "text": passage,
                    "batch_id": batch_id,
                }
            )
    return rows


def list_objects(s3, prefix: str) -> list[dict]:
    found: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{prefix}/"):
        found.extend(page.get("Contents", []))
    return found


def file_limit() -> int | None:
    raw = os.environ.get("MAX_FILES", "").strip()
    if not raw:
        return MAX_FILES
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("MAX_FILES must be a positive integer") from exc
    if value < 1:
        raise ValueError("MAX_FILES must be a positive integer")
    return value


def ordered_tasks(large: list[Task], small: list[Task]) -> list[Task]:
    return sorted(
        [*large, *small],
        key=lambda task: (family_rank(task.family), task.size, task.document_id),
    )


def take_batch(queue: list[Task], max_files: int | None) -> list[Task]:
    if max_files is None:
        return list(queue)
    return list(queue[:max_files])


def split_for_workers(tasks: list[Task]) -> tuple[list[Task], list[Task]]:
    alone = [task for task in tasks if runs_alone(task.size)]
    paired = [task for task in tasks if not runs_alone(task.size)]
    alone.sort(key=lambda task: (family_rank(task.family), -task.size))
    paired.sort(key=lambda task: (family_rank(task.family), task.document_id))
    return alone, paired


def resume_queue(discovered: list[Task], path: Path) -> list[Task]:
    by_key = {task.pdf_key: task for task in discovered}
    if not path.is_file():
        return list(discovered)
    saved = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    queue = [by_key[key] for key in saved if key in by_key]
    seen = {task.pdf_key for task in queue}
    queue.extend(task for task in discovered if task.pdf_key not in seen)
    return queue


def write_remaining(path: Path, keys: list[str]) -> None:
    path.write_text("".join(f"{key}\n" for key in keys), encoding="utf-8")


def settle_key(keys: list[str], pdf_key: str, status: str) -> None:
    if pdf_key not in keys:
        return
    keys.remove(pdf_key)
    if status == "failed":
        keys.append(pdf_key)


def parquet_tail_complete(size: int, tail: bytes) -> bool:
    if size < 12 or len(tail) < 8 or tail[-4:] != b"PAR1":
        return False
    footer_len = int.from_bytes(tail[-8:-4], "little")
    return 8 <= footer_len <= size - 8


def finished_shard_keys(s3, items: list[dict]) -> set[str]:
    shards = [item for item in items if str(item.get("Key", "")).endswith(".parquet")]

    def one(item: dict) -> str | None:
        key = str(item["Key"])
        size = int(item.get("Size") or 0)
        if size < 12:
            log.warning("incomplete shard will be rewritten: %s", key)
            return None
        try:
            response = s3.get_object(Bucket=BUCKET, Key=key, Range="bytes=-8")
            tail = response["Body"].read()
        except Exception:
            log.warning("shard tail could not be read, will rewrite %s", key)
            return None
        if parquet_tail_complete(size, tail):
            return key
        log.warning("incomplete shard will be rewritten: %s", key)
        return None

    if not shards:
        return set()
    finished: set[str] = set()
    with ThreadPoolExecutor(max_workers=min(8, len(shards))) as pool:
        for key in pool.map(one, shards):
            if key:
                finished.add(key)
    return finished


def discover(s3, batch_id: str) -> tuple[list[Task], list[Task], int]:
    existing = finished_shard_keys(s3, list_objects(s3, CHUNKS_ROOT))
    large: list[Task] = []
    small: list[Task] = []
    seen_out: set[str] = set()
    for item in list_objects(s3, RAW_ROOT):
        parsed = parse_raw_key(item["Key"])
        if parsed is None:
            continue
        family, document_id = parsed
        output = parquet_key(family, document_id)
        if output in existing or output in seen_out:
            continue
        seen_out.add(output)
        task = Task(item["Key"], int(item["Size"]), family, document_id, output, batch_id)
        if runs_alone(task.size):
            large.append(task)
        else:
            small.append(task)
    large.sort(key=lambda task: (family_rank(task.family), -task.size))
    small.sort(key=lambda task: (family_rank(task.family), task.document_id))
    return large, small, len(existing)


def ocr_page(page) -> str:
    import pymupdf
    import pytesseract
    from PIL import Image

    matrix = pymupdf.Matrix(OCR_DPI / 72, OCR_DPI / 72)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    try:
        with Image.open(io.BytesIO(pix.tobytes("png"))) as image:
            return pytesseract.image_to_string(image, lang="eng")
    finally:
        pix = None


def page_text(page) -> str:
    text = page.get_text("text") or ""
    if len(text.strip()) >= MIN_TEXT_CHARS:
        return text
    return ocr_page(page)


def write_parquet(rows: list[dict[str, object]], path: str) -> None:
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrow_schema = pa.schema(
        [
            ("chunk_id", pa.string()),
            ("document_id", pa.string()),
            ("asset_family", pa.string()),
            ("page_number", pa.int32()),
            ("tracing_page", pa.bool_()),
            ("fault_code", pa.string()),
            ("part", pa.string()),
            ("procedure", pa.string()),
            ("text", pa.string()),
            ("batch_id", pa.string()),
        ]
    )
    polars_schema = {
        "chunk_id": pl.String,
        "document_id": pl.String,
        "asset_family": pl.String,
        "page_number": pl.Int32,
        "tracing_page": pl.Boolean,
        "fault_code": pl.String,
        "part": pl.String,
        "procedure": pl.String,
        "text": pl.String,
        "batch_id": pl.String,
    }
    if rows:
        frame = pl.DataFrame(rows, schema=polars_schema).sort(
            ["document_id", "page_number", "chunk_id"]
        )
    else:
        frame = pl.DataFrame(schema=polars_schema)
    pq.write_table(frame.to_arrow().cast(arrow_schema), path, compression="zstd")


def chunk_one(task: Task) -> Result:
    import boto3
    import pymupdf

    pdf_path = ""
    parquet_path = ""
    try:
        s3 = boto3.client("s3", region_name=REGION)
        pdf_handle, pdf_path = tempfile.mkstemp(suffix=".pdf")
        os.close(pdf_handle)
        s3.download_file(BUCKET, task.pdf_key, pdf_path)
        with open(pdf_path, "rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise ValueError("object is not a PDF")
        document = pymupdf.open(pdf_path)
        try:
            pages = [(number, page_text(page)) for number, page in enumerate(document, start=1)]
        finally:
            document.close()
        rows = rows_for_document(task.document_id, task.family, pages, task.batch_id)
        parquet_handle, parquet_path = tempfile.mkstemp(suffix=".parquet")
        os.close(parquet_handle)
        write_parquet(rows, parquet_path)
        s3.upload_file(
            parquet_path,
            BUCKET,
            task.parquet_key,
            ExtraArgs={"ContentType": "application/vnd.apache.parquet"},
        )
        status = "stored" if rows else "empty"
        return Result(task.pdf_key, status, len(pages), len(rows), "")
    except Exception:
        error = traceback.format_exc(limit=4).strip()
        return Result(task.pdf_key, "failed", 0, 0, error)
    finally:
        for path in (pdf_path, parquet_path):
            if path and os.path.exists(path):
                os.unlink(path)


def require_tesseract() -> bool:
    import pytesseract

    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        log.error("tesseract-ocr is not installed. Install it with apt, then run again.")
        return False
    return True


def record_failure(path: Path, result: Result) -> None:
    line = " ".join(result.error.split())[:500]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{result.pdf_key}\t{line}\n")


def write_status(path: Path, fields: dict[str, object]) -> None:
    body = "\n".join(f"{key}={value}" for key, value in fields.items()) + "\n"
    path.write_text(body, encoding="utf-8")


def accept(
    result: Result,
    failed_path: Path,
    status_path: Path,
    remaining_path: Path,
    remaining_keys: list[str],
    counts: dict[str, int],
    total: int,
    already_parquet: int,
) -> None:
    counts[result.status] = counts.get(result.status, 0) + 1
    done = sum(counts.values())
    settle_key(remaining_keys, result.pdf_key, result.status)
    write_remaining(remaining_path, remaining_keys)
    shards = already_parquet + counts.get("stored", 0) + counts.get("empty", 0)
    write_status(
        status_path,
        {
            "parquet_shards": shards,
            "remaining": len(remaining_keys),
            "this_run_completed": done,
            "this_run_total": total,
            "stored": counts.get("stored", 0),
            "empty": counts.get("empty", 0),
            "failed": counts.get("failed", 0),
        },
    )
    if result.status == "failed":
        record_failure(failed_path, result)
        log.error(
            "parquet_shards=%s this_run=%s/%s failed %s",
            shards,
            done,
            total,
            result.pdf_key,
        )
        log.error("%s", result.error)
        return
    log.info(
        "parquet_shards=%s this_run=%s/%s %s pages=%s passages=%s %s",
        shards,
        done,
        total,
        result.status,
        result.pages,
        result.passages,
        result.pdf_key,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not require_tesseract():
        return 1

    import boto3
    from datetime import datetime, timezone

    try:
        limit = file_limit()
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    s3 = boto3.client("s3", region_name=REGION)
    try:
        large, small, already_parquet = discover(s3, batch_id)
    except Exception as exc:
        log.error("Cannot list s3://%s/%s/: %s", BUCKET, RAW_ROOT, exc)
        return 1

    remaining_path = Path.cwd() / REMAINING_NAME
    queue = resume_queue(ordered_tasks(large, small), remaining_path)
    remaining_keys = [task.pdf_key for task in queue]
    write_remaining(remaining_path, remaining_keys)
    batch = take_batch(queue, limit)
    large, small = split_for_workers(batch)
    total = len(batch)
    log.info(
        "batch=%s parquet_shards=%s remaining=%s this_run=%s max_files=%s",
        batch_id,
        already_parquet,
        len(remaining_keys),
        total,
        "all" if limit is None else limit,
    )
    status_path = Path.cwd() / STATUS_NAME
    write_status(
        status_path,
        {
            "parquet_shards": already_parquet,
            "remaining": len(remaining_keys),
            "this_run_completed": 0,
            "this_run_total": total,
            "stored": 0,
            "empty": 0,
            "failed": 0,
        },
    )
    if total == 0:
        return 0

    failed_path = Path.cwd() / FAILED_NAME
    counts: dict[str, int] = {}
    for task in large:
        accept(
            chunk_one(task),
            failed_path,
            status_path,
            remaining_path,
            remaining_keys,
            counts,
            total,
            already_parquet,
        )
    if small:
        context = get_context("spawn")
        pool_args: dict[str, object] = {"max_workers": WORKERS, "mp_context": context}
        if sys.version_info >= (3, 11):
            pool_args["max_tasks_per_child"] = TASKS_PER_CHILD
        with ProcessPoolExecutor(**pool_args) as pool:
            futures = [pool.submit(chunk_one, task) for task in small]
            for future in as_completed(futures):
                accept(
                    future.result(),
                    failed_path,
                    status_path,
                    remaining_path,
                    remaining_keys,
                    counts,
                    total,
                    already_parquet,
                )

    log.info(
        "finished parquet_shards=%s remaining=%s stored=%s empty=%s failed=%s",
        already_parquet + counts.get("stored", 0) + counts.get("empty", 0),
        len(remaining_keys),
        counts.get("stored", 0),
        counts.get("empty", 0),
        counts.get("failed", 0),
    )
    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
