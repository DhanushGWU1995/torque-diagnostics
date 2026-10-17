#!/usr/bin/env python3
"""Count passage, table, and figure Parquet, per equipment folder.

DuckDB reads the finished shards under text/, tables/, and images/. The
report counts each kind, including a shard that stored no rows, plus the
tracing pages, fault codes, parts, and procedures on the text shards. It
does not answer a technician question. The chatbot reads the BM25 and
embedding indexes, not this report.

Credentials come from the environment. This file does not contain an
access key. From a clone of the repo, at the repository root:

    uv sync
    uv run python data-preprocessing/coverage_duckdb.py

The report is written to coverage.txt in the working directory. Run it
after chunking, and again after table or image extraction.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _import_sources() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "chunk_sources.py").is_file():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return
    raise ImportError("chunk_sources.py was not found above this script")


_import_sources()
from chunk_sources import KINDS, union_sql

BUCKET = "dats-nlp"
REGION = "us-east-1"
CHUNKS_ROOT = "dhanush.mathivanan@gwu.edu/torque-diagnostics/chunks"
REPORT_NAME = "coverage.txt"

COLUMNS = (
    "asset_family",
    "text_shards",
    "table_shards",
    "image_shards",
    "text_rows",
    "table_rows",
    "image_rows",
    "blank_text",
    "blank_tables",
    "blank_images",
    "tracing_passages",
    "fault_passages",
    "part_passages",
    "procedure_passages",
)


def shard_identity(key: str) -> tuple[str, str, str] | None:
    marker = "/asset_family="
    if marker not in key or not key.endswith(".parquet"):
        return None
    parts = key.split(marker, 1)[1].split("/")
    if len(parts) != 3 or parts[1] not in KINDS or not parts[2].endswith(".parquet"):
        return None
    family, kind, filename = parts
    if not family:
        return None
    return family, filename[: -len(".parquet")], kind


def list_s3_shards() -> list[tuple[str, str, str]]:
    import boto3

    found: list[tuple[str, str, str]] = []
    paginator = boto3.client("s3", region_name=REGION).get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{CHUNKS_ROOT}/"):
        for item in page.get("Contents", []):
            parsed = shard_identity(item["Key"])
            if parsed is not None:
                found.append(parsed)
    return found


def list_local_shards(root: Path) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for path in root.rglob("*.parquet"):
        parsed = shard_identity(path.as_posix())
        if parsed is not None:
            found.append(parsed)
    return found


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


def kind_globs(base: str, kinds: set[str]) -> dict[str, str]:
    return {
        kind: f"{base}/asset_family=*/{kind}/*.parquet"
        for kind in KINDS
        if kind in kinds
    }


def report_rows(connection, globs: dict[str, str], shards: list[tuple[str, str, str]]):
    connection.execute(
        "CREATE TEMP TABLE shard_files (asset_family VARCHAR, document_id VARCHAR, source_kind VARCHAR)"
    )
    connection.executemany("INSERT INTO shard_files VALUES (?, ?, ?)", shards)
    if globs:
        row_source = f"""
            SELECT asset_family, source_kind,
                count(*) AS row_count,
                count(*) FILTER (
                    WHERE text IS NULL OR trim(CAST(text AS VARCHAR)) = ''
                ) AS blank_count
            FROM ({union_sql(globs)}) sources
            GROUP BY asset_family, source_kind
        """
    else:
        row_source = """
            SELECT CAST(NULL AS VARCHAR) AS asset_family,
                CAST(NULL AS VARCHAR) AS source_kind,
                0 AS row_count,
                0 AS blank_count
            WHERE false
        """
    if "text" in globs:
        escaped = globs["text"].replace("'", "''")
        text_source = f"""
            SELECT asset_family,
                count(*) FILTER (WHERE tracing_page) AS tracing_passages,
                count(*) FILTER (WHERE coalesce(fault_code, '') <> '') AS fault_passages,
                count(*) FILTER (WHERE coalesce(part, '') <> '') AS part_passages,
                count(*) FILTER (WHERE coalesce(procedure, '') <> '') AS procedure_passages
            FROM read_parquet('{escaped}', hive_partitioning = false)
            GROUP BY asset_family
        """
    else:
        text_source = """
            SELECT CAST(NULL AS VARCHAR) AS asset_family,
                0 AS tracing_passages,
                0 AS fault_passages,
                0 AS part_passages,
                0 AS procedure_passages
            WHERE false
        """
    return connection.execute(
        f"""
        WITH shard_counts AS (
            SELECT
                asset_family,
                count(*) FILTER (WHERE source_kind = 'text') AS text_shards,
                count(*) FILTER (WHERE source_kind = 'tables') AS table_shards,
                count(*) FILTER (WHERE source_kind = 'images') AS image_shards
            FROM shard_files
            GROUP BY asset_family
        ),
        row_counts AS (
            {row_source}
        ),
        row_pivot AS (
            SELECT
                asset_family,
                coalesce(max(CASE WHEN source_kind = 'text' THEN row_count END), 0) AS text_rows,
                coalesce(max(CASE WHEN source_kind = 'tables' THEN row_count END), 0) AS table_rows,
                coalesce(max(CASE WHEN source_kind = 'images' THEN row_count END), 0) AS image_rows,
                coalesce(max(CASE WHEN source_kind = 'text' THEN blank_count END), 0) AS blank_text,
                coalesce(max(CASE WHEN source_kind = 'tables' THEN blank_count END), 0) AS blank_tables,
                coalesce(max(CASE WHEN source_kind = 'images' THEN blank_count END), 0) AS blank_images
            FROM row_counts
            GROUP BY asset_family
        ),
        text_counts AS (
            {text_source}
        )
        SELECT
            shard_counts.asset_family,
            shard_counts.text_shards,
            shard_counts.table_shards,
            shard_counts.image_shards,
            coalesce(row_pivot.text_rows, 0),
            coalesce(row_pivot.table_rows, 0),
            coalesce(row_pivot.image_rows, 0),
            coalesce(row_pivot.blank_text, 0),
            coalesce(row_pivot.blank_tables, 0),
            coalesce(row_pivot.blank_images, 0),
            coalesce(text_counts.tracing_passages, 0),
            coalesce(text_counts.fault_passages, 0),
            coalesce(text_counts.part_passages, 0),
            coalesce(text_counts.procedure_passages, 0)
        FROM shard_counts
        LEFT JOIN row_pivot USING (asset_family)
        LEFT JOIN text_counts USING (asset_family)
        ORDER BY shard_counts.asset_family
        """
    ).fetchall()


def write_report(rows: list[tuple]) -> None:
    lines = ["\t".join(COLUMNS)]
    totals = [0] * (len(COLUMNS) - 1)
    for row in rows:
        lines.append("\t".join(str(value) for value in row))
        for index, value in enumerate(row[1:]):
            totals[index] += int(value)
    lines.append("\t".join(["TOTAL", *(str(value) for value in totals)]))
    text = "\n".join(lines) + "\n"
    Path(REPORT_NAME).write_text(text, encoding="utf-8")
    sys.stdout.write(text)


def main() -> int:
    local = os.environ.get("COVERAGE_PARQUET", "").strip()
    if local:
        root = Path(local)
        if not root.is_dir():
            print(f"COVERAGE_PARQUET is not a directory: {root}", file=sys.stderr)
            return 1
        shards = list_local_shards(root)
        base = root.as_posix()
        import duckdb

        connection = duckdb.connect()
    else:
        shards = list_s3_shards()
        base = f"s3://{BUCKET}/{CHUNKS_ROOT}"
        connection = connect()
    if not shards:
        print("No Parquet shards found.", file=sys.stderr)
        return 1
    globs = kind_globs(base, {kind for _family, _document_id, kind in shards})
    write_report(report_rows(connection, globs, shards))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
