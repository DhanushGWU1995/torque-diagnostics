"""Read passage, table, and figure Parquet as one set of rows.

The text chunker writes chunks/asset_family=<family>/text/. The table and
image scripts write tables/ and images/ beside it. A document id is reused
in each folder, and the row ids inside those files can match, so this
module prefixes table and figure ids before search or citation.
"""

from __future__ import annotations

import os
from pathlib import Path

BUCKET = "dats-nlp"
REGION = "us-east-1"
CHUNKS_ROOT = "dhanush.mathivanan@gwu.edu/torque-diagnostics/chunks"
KINDS = ("text", "tables", "images")
ID_COLUMN = {"text": "chunk_id", "tables": "table_id", "images": "image_id"}
ID_PREFIX = {"text": "", "tables": "table:", "images": "image:"}


def rebuild_requested() -> bool:
    return os.environ.get("REBUILD", "").strip() == "1"


def source_label(chunk_id: str) -> str:
    """Citation word for a row. Passage rows stay unlabeled."""
    if str(chunk_id).startswith("table:"):
        return "table"
    if str(chunk_id).startswith("image:"):
        return "figure"
    return ""


def union_sql(globs: dict[str, str]) -> str:
    """One SELECT of chunk_id, document_id, asset_family, page_number, text, source_kind."""
    parts: list[str] = []
    for kind in KINDS:
        glob = globs.get(kind)
        if not glob:
            continue
        escaped = glob.replace("'", "''")
        prefix = ID_PREFIX[kind]
        column = ID_COLUMN[kind]
        chunk_id = f"'{prefix}' || {column}" if prefix else column
        parts.append(
            f"""
            SELECT
                {chunk_id} AS chunk_id,
                document_id,
                asset_family,
                page_number,
                text,
                '{kind}' AS source_kind
            FROM read_parquet('{escaped}', hive_partitioning = false)
            """
        )
    if not parts:
        raise FileNotFoundError("no passage, table, or figure Parquet")
    return "\nUNION ALL\n".join(parts)


def passage_sql_for(family: str) -> str:
    globs = present_family_globs(family)
    if not globs:
        raise FileNotFoundError(f"no passage, table, or figure Parquet for {family}")
    return union_sql(globs)


def present_family_globs(family: str) -> dict[str, str]:
    """Parquet globs that exist for one equipment folder."""
    local = os.environ.get("CHUNK_PARQUET", "").strip()
    if local:
        return _local_globs(Path(local), family)
    return _s3_globs(family)


def _local_globs(root: Path, family: str) -> dict[str, str]:
    single = root / f"asset_family={family}.parquet"
    if single.is_file():
        return {"text": single.as_posix()}
    found: dict[str, str] = {}
    family_dir = root / f"asset_family={family}"
    for kind in KINDS:
        folder = family_dir / kind
        if folder.is_dir() and any(folder.glob("*.parquet")):
            found[kind] = f"{folder.as_posix()}/*.parquet"
    return found


def _s3_globs(family: str) -> dict[str, str]:
    import boto3

    client = boto3.client("s3", region_name=REGION)
    found: dict[str, str] = {}
    for kind in KINDS:
        prefix = f"{CHUNKS_ROOT}/asset_family={family}/{kind}/"
        response = client.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=1)
        if not response.get("Contents"):
            continue
        found[kind] = f"s3://{BUCKET}/{CHUNKS_ROOT}/asset_family={family}/{kind}/*.parquet"
    return found
