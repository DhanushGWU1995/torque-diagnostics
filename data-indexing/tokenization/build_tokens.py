#!/usr/bin/env python3
"""Write BM25 and identifier tokens for one equipment folder.

This reads the finished Parquet shards. It does not open the PDFs and it
does not build the embedding index. Set INDEX_FAMILY to one folder for a
test. INDEX_FAMILY=all writes every folder, which is a full read of the
shards. A folder that already has a finished token file is skipped. library is
tokenized with three processes. Every other folder stays on one process.
The tokens and their order are the same either way.

Credentials come from the environment. This file does not contain an
access key.

    INDEX_FAMILY=ACS880-11 uv run python data-indexing/tokenization/build_tokens.py

On the EC2 instance, from ~/dl-venv/nlp_project:

    INDEX_FAMILY=ACS880-11 ~/dl-venv/bin/python data-indexing/tokenization/build_tokens.py

Each folder is written to

    s3://dats-nlp/dhanush.mathivanan@gwu.edu/torque-diagnostics/index/tokens/asset_family=<family>.parquet
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manual_tokens import bm25_tokens, identifier_tokens


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
from asset_layout import family_rank

BUCKET = "dats-nlp"
REGION = "us-east-1"
CHUNKS_ROOT = "dhanush.mathivanan@gwu.edu/torque-diagnostics/chunks"
INDEX_ROOT = "dhanush.mathivanan@gwu.edu/torque-diagnostics/index/tokens"
STATUS_NAME = "token_status.txt"
LARGE_FAMILIES = frozenset({"library"})
LARGE_WORKERS = 3
_FAMILY_NAME = re.compile(r"[A-Za-z0-9_.+-]+")


def tokenize_batch(batch: list[tuple]) -> tuple[list[dict[str, object]], int]:
    """Tokenize one fetched batch. Row order inside the batch stays put."""
    rows: list[dict[str, object]] = []
    identified = 0
    for chunk_id, document_id, asset_family_name, page_number, text in batch:
        words = bm25_tokens(text)
        found = identifier_tokens(text)
        if found:
            identified += 1
        rows.append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "asset_family": asset_family_name,
                "page_number": int(page_number),
                "bm25_tokens": " ".join(words),
                "identifiers": "|".join(f"{item.kind}:{item.text}" for item in found),
            }
        )
    return rows, identified


def token_key(family: str) -> str:
    return f"{INDEX_ROOT}/asset_family={family}.parquet"


def connect():
    import duckdb

    connection = duckdb.connect()
    connection.execute("INSTALL httpfs")
    connection.execute("LOAD httpfs")
    connection.execute(
        f"""
        CREATE OR REPLACE SECRET (
            TYPE s3,
            PROVIDER credential_chain,
            CHAIN 'env',
            REGION '{REGION}'
        )
        """
    )
    return connection


def finished(s3, key: str) -> bool:
    try:
        response = s3.get_object(Bucket=BUCKET, Key=key, Range="bytes=-8")
    except Exception:
        return False
    tail = response["Body"].read()
    if len(tail) < 8 or tail[-4:] != b"PAR1":
        return False
    length = int.from_bytes(tail[:4], "little")
    return length >= 0


def families_in_bucket(s3) -> list[str]:
    found: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{CHUNKS_ROOT}/", Delimiter="/"):
        for prefix in page.get("CommonPrefixes", []):
            name = prefix["Prefix"].rstrip("/").rsplit("asset_family=", 1)[-1]
            if name and _FAMILY_NAME.fullmatch(name):
                found.append(name)
    found.sort(key=family_rank)
    return found


def selected_families(s3) -> list[str]:
    raw = os.environ.get("INDEX_FAMILY", "").strip()
    if not raw:
        print("Set INDEX_FAMILY to one folder, for example ACS880-11, or to all.", file=sys.stderr)
        return []
    if raw == "all":
        return families_in_bucket(s3)
    if not _FAMILY_NAME.fullmatch(raw):
        print(f"INDEX_FAMILY is not a folder name: {raw}", file=sys.stderr)
        return []
    return [raw]


def write_family(connection, s3, family: str) -> tuple[int, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    key = token_key(family)
    if finished(s3, key):
        print(f"skip {family}")
        return 0, 0
    glob = f"s3://{BUCKET}/{CHUNKS_ROOT}/asset_family={family}/*.parquet"
    escaped = glob.replace("'", "''")
    cursor = connection.execute(
        f"""
        SELECT chunk_id, document_id, asset_family, page_number, text
        FROM read_parquet('{escaped}', hive_partitioning = false)
        WHERE text IS NOT NULL AND trim(text) <> ''
        """
    )
    schema = pa.schema(
        [
            ("chunk_id", pa.string()),
            ("document_id", pa.string()),
            ("asset_family", pa.string()),
            ("page_number", pa.int32()),
            ("bm25_tokens", pa.string()),
            ("identifiers", pa.string()),
        ]
    )
    handle, path = tempfile.mkstemp(suffix=".parquet")
    os.close(handle)
    writer = pq.ParquetWriter(path, schema, compression="zstd")
    passages = 0
    identified = 0
    workers = LARGE_WORKERS if family in LARGE_FAMILIES else 1

    def write_rows(rows: list[dict[str, object]], found_count: int) -> None:
        nonlocal passages, identified
        identified += found_count
        passages += len(rows)
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))

    try:
        if workers == 1:
            while True:
                batch = cursor.fetchmany(2000)
                if not batch:
                    break
                write_rows(*tokenize_batch(batch))
        else:
            from concurrent.futures import ProcessPoolExecutor
            from multiprocessing import get_context

            print(f"{family} workers={workers}", flush=True)
            context = get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                pending = []
                while True:
                    batch = cursor.fetchmany(2000)
                    if not batch:
                        break
                    pending.append(pool.submit(tokenize_batch, list(batch)))
                    if len(pending) >= workers:
                        write_rows(*pending.pop(0).result())
                for future in pending:
                    write_rows(*future.result())
        writer.close()
        writer = None
        s3.upload_file(path, BUCKET, key)
    finally:
        if writer is not None:
            writer.close()
        os.remove(path)
    print(f"stored {family} passages={passages} with_identifiers={identified}")
    return passages, identified


def main() -> int:
    import boto3

    s3 = boto3.client("s3", region_name=REGION)
    families = selected_families(s3)
    if not families:
        return 1
    connection = connect()
    total_passages = 0
    total_identified = 0
    stored = 0
    for family in families:
        passages, identified = write_family(connection, s3, family)
        if passages or identified:
            stored += 1
        total_passages += passages
        total_identified += identified
    line = (
        f"families={len(families)} stored={stored} "
        f"passages={total_passages} with_identifiers={total_identified}\n"
    )
    Path(STATUS_NAME).write_text(line, encoding="utf-8")
    sys.stdout.write(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
