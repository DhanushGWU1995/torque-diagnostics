#!/usr/bin/env python3
"""Count what the Parquet shards contain, per equipment folder.

Polars writes each shard. DuckDB reads those finished shards and reports
how many passages, tracing pages, fault codes, parts, and procedures each
family actually stored. It does not answer a technician question. The
chatbot reads the BM25 and embedding indexes, not this report.

Credentials come from the environment. This file does not contain an
access key. From a clone of the repo, at the repository root:

    uv sync
    uv run python data-preprocessing/coverage_duckdb.py

The report is written to coverage.txt in the working directory. Run it
once, after chunking has finished.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BUCKET = "dats-nlp"
REGION = "us-east-1"
CHUNKS_ROOT = "dhanush.mathivanan@gwu.edu/torque-diagnostics/chunks"
REPORT_NAME = "coverage.txt"

COLUMNS = (
    "asset_family",
    "shards",
    "shards_without_passages",
    "passages",
    "blank_passages",
    "tracing_passages",
    "fault_passages",
    "part_passages",
    "procedure_passages",
)


def shard_identity(key: str) -> tuple[str, str] | None:
    marker = "/asset_family="
    if marker not in key or not key.endswith(".parquet"):
        return None
    family, _, filename = key.split(marker, 1)[1].partition("/")
    if not family or "/" in filename or not filename.endswith(".parquet"):
        return None
    return family, filename[: -len(".parquet")]


def list_s3_shards() -> list[tuple[str, str]]:
    import boto3

    found: list[tuple[str, str]] = []
    paginator = boto3.client("s3", region_name=REGION).get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{CHUNKS_ROOT}/"):
        for item in page.get("Contents", []):
            parsed = shard_identity(item["Key"])
            if parsed is not None:
                found.append(parsed)
    return found


def list_local_shards(root: Path) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
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


def coverage_sql(parquet_glob: str) -> str:
    escaped = parquet_glob.replace("'", "''")
    return f"""
        SELECT
            asset_family,
            count(DISTINCT document_id) AS documents_with_passages,
            count(*) AS passages,
            count(*) FILTER (
                WHERE text IS NULL OR trim(text) = ''
            ) AS blank_passages,
            count(*) FILTER (WHERE tracing_page) AS tracing_passages,
            count(*) FILTER (
                WHERE coalesce(fault_code, '') <> ''
            ) AS fault_passages,
            count(*) FILTER (WHERE coalesce(part, '') <> '') AS part_passages,
            count(*) FILTER (
                WHERE coalesce(procedure, '') <> ''
            ) AS procedure_passages
        FROM read_parquet('{escaped}', hive_partitioning = false)
        GROUP BY asset_family
    """


def report_rows(connection, parquet_glob: str, shards: list[tuple[str, str]]):
    connection.execute("CREATE TEMP TABLE shard_files (asset_family VARCHAR, document_id VARCHAR)")
    connection.executemany("INSERT INTO shard_files VALUES (?, ?)", shards)
    return connection.execute(
        f"""
        WITH passage_counts AS (
            {coverage_sql(parquet_glob)}
        ),
        shard_counts AS (
            SELECT asset_family, count(*) AS shards
            FROM shard_files
            GROUP BY asset_family
        )
        SELECT
            shard_counts.asset_family,
            shard_counts.shards,
            shard_counts.shards - coalesce(passage_counts.documents_with_passages, 0)
                AS shards_without_passages,
            coalesce(passage_counts.passages, 0) AS passages,
            coalesce(passage_counts.blank_passages, 0) AS blank_passages,
            coalesce(passage_counts.tracing_passages, 0) AS tracing_passages,
            coalesce(passage_counts.fault_passages, 0) AS fault_passages,
            coalesce(passage_counts.part_passages, 0) AS part_passages,
            coalesce(passage_counts.procedure_passages, 0) AS procedure_passages
        FROM shard_counts
        LEFT JOIN passage_counts USING (asset_family)
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
        parquet_glob = str(root / "**" / "*.parquet")
        import duckdb

        connection = duckdb.connect()
    else:
        shards = list_s3_shards()
        parquet_glob = f"s3://{BUCKET}/{CHUNKS_ROOT}/**/*.parquet"
        connection = connect()
    if not shards:
        print("No Parquet shards found.", file=sys.stderr)
        return 1
    write_report(report_rows(connection, parquet_glob, shards))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
