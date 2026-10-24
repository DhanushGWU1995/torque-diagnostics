"""Read passage Parquet for one equipment folder.

Passage files live at

    chunks/asset_family=<family>/<document_id>.parquet

Finished token, BM25, and embedding files are left in place. A later index
build skips a finished file unless REBUILD=1.
"""

from __future__ import annotations

import os
from pathlib import Path

BUCKET = "g34305547-dats6450-s2026"
REGION = "us-east-1"
CHUNKS_ROOT = "torque-diagnostics/chunks"


def rebuild_requested() -> bool:
    return os.environ.get("REBUILD", "").strip() == "1"


def passage_glob(family: str) -> str:
    local = os.environ.get("CHUNK_PARQUET", "").strip()
    if local:
        root = Path(local)
        single = root / f"asset_family={family}.parquet"
        if single.is_file():
            return single.as_posix()
        return f"{(root / f'asset_family={family}').as_posix()}/*.parquet"
    return f"s3://{BUCKET}/{CHUNKS_ROOT}/asset_family={family}/*.parquet"


def passages_present(family: str) -> bool:
    local = os.environ.get("CHUNK_PARQUET", "").strip()
    if local:
        root = Path(local)
        single = root / f"asset_family={family}.parquet"
        folder = root / f"asset_family={family}"
        return single.is_file() or any(folder.glob("*.parquet"))
    import boto3

    client = boto3.client("s3", region_name=REGION)
    prefix = f"{CHUNKS_ROOT}/asset_family={family}/"
    response = client.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=5)
    for item in response.get("Contents", []):
        key = str(item.get("Key", ""))
        if key.endswith(".parquet") and key.count("/") == prefix.count("/"):
            return True
    return False


def passage_sql_for(family: str) -> str:
    escaped = passage_glob(family).replace("'", "''")
    return f"""
        SELECT chunk_id, document_id, asset_family, page_number, text
        FROM read_parquet('{escaped}', hive_partitioning = false)
    """
