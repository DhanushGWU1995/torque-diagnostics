#!/usr/bin/env python3
"""Move raw PDFs already in S3 onto the equipment folders.

The bytes stay in S3. Each object is copied to the key asset_layout.py
would use for that CSV row, and the old key is deleted only after the
copy has the same size. ACS880 frame folders and the two firmware
folders are left where they are. Catalogues, articles, brochures, and
technical files with no series in the title stay under library.

A Parquet shard already written as asset_family=library is rewritten
with the new family name and stored under that folder. The PDF is not
read again.

From the repository root, with a current AWS session:

    DRY_RUN=1 uv run python data-ingestion/scripts/rehome_raw_pdfs.py
    uv run python data-ingestion/scripts/rehome_raw_pdfs.py

DRY_RUN=1 lists the moves and does not copy or delete anything.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def _import_layout() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "asset_layout.py").is_file():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return
    raise ImportError("asset_layout.py was not found above this script")


_import_layout()
from asset_layout import (  # noqa: E402
    RAW_ROOT,
    asset_family,
    document_filename,
    document_kind_folder,
    raw_key,
)

BUCKET = "dats-nlp"
REGION = "us-east-1"
CHUNKS_ROOT = "dhanush.mathivanan@gwu.edu/torque-diagnostics/chunks"
CSV_NAME = "ABB_100GB_Download_List.csv"
WORKERS = 8
FAILED_NAME = "rehome_failed.txt"

log = logging.getLogger("rehome-raw")


def csv_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    beside = Path(__file__).resolve().parent / CSV_NAME
    in_cwd = Path.cwd() / CSV_NAME
    if in_cwd.is_file():
        return in_cwd
    return beside


def dry_run() -> bool:
    return os.environ.get("DRY_RUN", "").strip() in {"1", "true", "TRUE", "yes"}


def load_plan(path: Path) -> dict[str, tuple[str, str]]:
    plan: dict[str, tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row.get("file_suffix") or "").lower() not in {"", "pdf"}:
                continue
            document_id = (row.get("document_id") or "").strip()
            if not document_id or not row.get("download_url"):
                continue
            family = asset_family(row.get("title", ""), document_id, row.get("document_kind", ""))
            kind = document_kind_folder(row.get("document_kind", ""))
            filename = document_filename(document_id)
            stem = filename[: -len(".pdf")]
            desired = raw_key(family, kind, document_id)
            if stem in plan and plan[stem] != (family, desired):
                raise ValueError(f"two rows share the stored name {stem}")
            plan[stem] = (family, desired)
    return plan


def list_objects(s3, prefix: str) -> list[dict]:
    found: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{prefix}/"):
        found.extend(page.get("Contents", []))
    return found


def pdf_stem(key: str) -> str | None:
    prefix = f"{RAW_ROOT}/"
    if not key.startswith(prefix) or not key.lower().endswith(".pdf"):
        return None
    parts = key[len(prefix) :].split("/")
    if len(parts) != 3 or not parts[2]:
        return None
    return parts[2][: -len(".pdf")]


def parquet_parts(key: str) -> tuple[str, str] | None:
    prefix = f"{CHUNKS_ROOT}/"
    if not key.startswith(prefix) or not key.endswith(".parquet"):
        return None
    rest = key[len(prefix) :]
    folder, _, filename = rest.partition("/")
    if not folder.startswith("asset_family=") or not filename.endswith(".parquet"):
        return None
    family = folder[len("asset_family=") :]
    document_id = filename[: -len(".parquet")]
    if not family or not document_id:
        return None
    return family, document_id


def copy_pdf(s3, source: str, dest: str, size: int) -> str:
    if source == dest:
        return "kept"
    head = None
    try:
        head = s3.head_object(Bucket=BUCKET, Key=dest)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code not in {"404", "NoSuchKey", "NotFound"}:
            raise
    if head is not None and int(head["ContentLength"]) == size:
        s3.delete_object(Bucket=BUCKET, Key=source)
        return "dropped-duplicate"
    if head is not None:
        raise RuntimeError(f"destination exists with a different size: {dest}")
    s3.copy_object(
        Bucket=BUCKET,
        Key=dest,
        CopySource={"Bucket": BUCKET, "Key": source},
        MetadataDirective="COPY",
    )
    copied = s3.head_object(Bucket=BUCKET, Key=dest)
    if int(copied["ContentLength"]) != size:
        raise RuntimeError(f"copy size mismatch for {dest}")
    s3.delete_object(Bucket=BUCKET, Key=source)
    return "moved"


def rewrite_shard(s3, source: str, dest: str, family: str) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if source == dest:
        return "kept"
    body = io.BytesIO()
    s3.download_fileobj(BUCKET, source, body)
    body.seek(0)
    table = pq.read_table(body)
    if "asset_family" not in table.column_names:
        raise RuntimeError(f"shard has no asset_family column: {source}")
    if table.num_rows:
        column = pa.array([family] * table.num_rows, type=pa.string())
        index = table.schema.get_field_index("asset_family")
        table = table.set_column(index, "asset_family", column)
    out = io.BytesIO()
    pq.write_table(table, out, compression="zstd")
    out.seek(0)
    s3.upload_fileobj(
        out,
        BUCKET,
        dest,
        ExtraArgs={"ContentType": "application/vnd.apache.parquet"},
    )
    s3.delete_object(Bucket=BUCKET, Key=source)
    return "moved"


def run_jobs(s3, jobs: list[tuple], worker) -> tuple[int, list[str]]:
    failed: list[str] = []
    done = 0
    lock = threading.Lock()
    moved = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(worker, s3, *job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            source = job[0]
            with lock:
                done += 1
            try:
                status = future.result()
            except (BotoCoreError, ClientError, OSError, RuntimeError, ValueError) as exc:
                failed.append(f"{source}\t{exc}")
                log.warning("failed %s (%s/%s) %s", source, done, len(jobs), exc)
                continue
            moved += status != "kept"
            if done % 100 == 0 or done == len(jobs):
                log.info("progress %s/%s moved=%s failed=%s", done, len(jobs), moved, len(failed))
    return moved, failed


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    path = csv_path()
    if not path.is_file():
        log.error("CSV not found at %s", path)
        return 1
    plan = load_plan(path)
    preview = dry_run()
    s3 = boto3.client("s3", region_name=REGION)
    try:
        raw_items = list_objects(s3, RAW_ROOT)
        shard_items = list_objects(s3, CHUNKS_ROOT)
    except (BotoCoreError, ClientError) as exc:
        log.error("Cannot list s3://%s/: %s", BUCKET, exc)
        return 1

    pdf_jobs: list[tuple[str, str, int]] = []
    for item in raw_items:
        stem = pdf_stem(item["Key"])
        if stem is None or stem not in plan:
            continue
        _family, dest = plan[stem]
        if item["Key"] == dest:
            continue
        pdf_jobs.append((item["Key"], dest, int(item["Size"])))

    shard_jobs: list[tuple[str, str, str]] = []
    for item in shard_items:
        parsed = parquet_parts(item["Key"])
        if parsed is None:
            continue
        family, document_id = parsed
        if document_id not in plan:
            continue
        desired_family = plan[document_id][0]
        if family == desired_family:
            continue
        dest = f"{CHUNKS_ROOT}/asset_family={desired_family}/{document_id}.parquet"
        shard_jobs.append((item["Key"], dest, desired_family))

    log.info(
        "dry_run=%s pdfs_to_move=%s shards_to_move=%s",
        preview,
        len(pdf_jobs),
        len(shard_jobs),
    )
    for source, dest, _size in pdf_jobs[:8]:
        log.info("pdf %s -> %s", source, dest)
    for source, dest, _family in shard_jobs[:8]:
        log.info("shard %s -> %s", source, dest)
    if preview:
        return 0

    moved, failed = run_jobs(s3, pdf_jobs, copy_pdf)
    shard_moved, shard_failed = run_jobs(s3, shard_jobs, rewrite_shard)
    failed.extend(shard_failed)
    if failed:
        Path(FAILED_NAME).write_text("\n".join(failed) + "\n", encoding="utf-8")
        log.error("failed=%s see %s", len(failed), FAILED_NAME)
        return 1
    log.info("done pdfs_moved=%s shards_moved=%s", moved, shard_moved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
