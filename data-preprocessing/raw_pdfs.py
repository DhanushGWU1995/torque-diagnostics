"""Shared S3 listing for the table and image extractors.

The text chunks already exist. These helpers only find raw PDFs and skip a
finished Parquet object. They do not rebuild tokens, BM25, or embeddings.
Those index scripts read text/, tables/, and images/ together.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NamedTuple


def import_layout() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "asset_layout.py").is_file():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return
    raise ImportError("asset_layout.py was not found above this script")


import_layout()
from asset_layout import family_rank

BUCKET = "dats-nlp"
REGION = "us-east-1"
RAW_ROOT = "dhanush.mathivanan@gwu.edu/torque-diagnostics/raw"
PREFIX = "dhanush.mathivanan@gwu.edu/torque-diagnostics"


class Task(NamedTuple):
    pdf_key: str
    size: int
    family: str
    document_id: str
    parquet_key: str


def parse_raw_key(key: str) -> tuple[str, str] | None:
    prefix = f"{RAW_ROOT}/"
    if not key.startswith(prefix) or not key.lower().endswith(".pdf"):
        return None
    parts = key[len(prefix) :].split("/")
    if len(parts) != 3 or not parts[0] or not parts[2]:
        return None
    family, _kind, filename = parts
    document_id = filename[: -len(".pdf")]
    if not document_id:
        return None
    return family, document_id


def output_key(kind: str, family: str, document_id: str) -> str:
    """Parquet under the family folder. The passage file keeps its own name."""
    return f"{PREFIX}/chunks/asset_family={family}/{kind}/{document_id}.parquet"


def list_objects(s3, prefix: str) -> list[dict]:
    found: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{prefix}/"):
        found.extend(page.get("Contents", []))
    return found


def file_limit() -> int | None:
    raw = os.environ.get("MAX_FILES", "").strip()
    if not raw:
        return None
    value = int(raw)
    if value < 1:
        raise ValueError("MAX_FILES must be a positive integer")
    return value


def selected_family() -> str:
    return os.environ.get("INDEX_FAMILY", "").strip()


def parquet_tail_complete(size: int, tail: bytes) -> bool:
    if size < 12 or len(tail) < 8 or tail[-4:] != b"PAR1":
        return False
    footer_len = int.from_bytes(tail[-8:-4], "little")
    return 8 <= footer_len <= size - 8


def finished_shard_keys(s3, items: list[dict]) -> set[str]:
    finished: set[str] = set()
    for item in items:
        key = str(item.get("Key", ""))
        if not key.endswith(".parquet"):
            continue
        size = int(item.get("Size") or 0)
        if size < 12:
            continue
        response = s3.get_object(Bucket=BUCKET, Key=key, Range="bytes=-8")
        tail = response["Body"].read()
        if parquet_tail_complete(size, tail):
            finished.add(key)
    return finished


def discover(s3, kind: str) -> tuple[list[Task], int]:
    family_filter = selected_family()
    marker = f"/{kind}/"
    items = [
        item
        for item in list_objects(s3, f"{PREFIX}/chunks")
        if marker in str(item.get("Key", ""))
    ]
    existing = finished_shard_keys(s3, items)
    pending: list[Task] = []
    seen: set[str] = set()
    for item in list_objects(s3, RAW_ROOT):
        parsed = parse_raw_key(item["Key"])
        if parsed is None:
            continue
        family, document_id = parsed
        if family_filter and family != family_filter:
            continue
        output = output_key(kind, family, document_id)
        if output in existing or output in seen:
            continue
        seen.add(output)
        pending.append(Task(item["Key"], int(item["Size"]), family, document_id, output))
    pending.sort(key=lambda task: (family_rank(task.family), task.document_id))
    return pending, len(existing)


def write_parquet(rows: list[dict[str, object]], path: str, schema) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd")
