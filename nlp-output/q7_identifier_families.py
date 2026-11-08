#!/usr/bin/env python3
"""Proposal Q7 — fault / part / procedure identifiers in one vs many families.

Reads passage Parquet under the chunks prefix. Writes:

    nlp-output/q7_identifier_families.csv
    nlp-output/q7_summary.csv
    nlp-output/q7_shared_vs_unique.png

    uv run python nlp-output/q7_identifier_families.py
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
BUCKET = os.environ.get("EDA_BUCKET", "dats-nlp")
PREFIX = os.environ.get(
    "EDA_PREFIX",
    "dhanush.mathivanan@gwu.edu/torque-diagnostics/chunks/",
).strip("/") + "/"
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
WORKERS = int(os.environ.get("NLP_WORKERS", "12"))
KINDS = ("fault_code", "part", "procedure")
BLUE = "#1f4e79"
GOLD = "#c59b3a"
GRAY = "#7f8c9b"


def family_from_key(key: str) -> str:
    marker = "asset_family="
    if marker not in key:
        return ""
    return key.split(marker, 1)[1].split("/", 1)[0]


def list_parquet_keys(s3) -> list[str]:
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith(".parquet") and "asset_family=" in key:
                keys.append(key)
    return keys


def split_idents(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def scan_key(key: str) -> dict[tuple[str, str], set[str]]:
    """Return mapping (kind, ident) -> set of families for one shard."""
    import io

    s3 = boto3.client("s3", region_name=REGION)
    body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    table = pq.read_table(
        io.BytesIO(body),
        columns=["fault_code", "part", "procedure"],
    )
    family = family_from_key(key)
    found: dict[tuple[str, str], set[str]] = defaultdict(set)
    if not family:
        return found
    for kind in KINDS:
        for value in table.column(kind).to_pylist():
            for ident in split_idents(value):
                found[(kind, ident)].add(family)
    return found


def merge(
    target: dict[tuple[str, str], set[str]],
    piece: dict[tuple[str, str], set[str]],
) -> None:
    for key, families in piece.items():
        target[key].update(families)


def main() -> int:
    s3 = boto3.client("s3", region_name=REGION)
    keys = list_parquet_keys(s3)
    if not keys:
        print(f"No parquet under s3://{BUCKET}/{PREFIX}", file=sys.stderr)
        return 1
    print(f"q7: scanning {len(keys)} shards", flush=True)

    combined: dict[tuple[str, str], set[str]] = defaultdict(set)
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(scan_key, key): key for key in keys}
        for future in as_completed(futures):
            merge(combined, future.result())
            done += 1
            if done % 500 == 0 or done == len(keys):
                print(f"q7: scanned {done}/{len(keys)}", flush=True)

    detail_rows: list[dict] = []
    summary = {kind: {"unique": 0, "shared": 0, "total": 0} for kind in KINDS}
    for (kind, ident), families in sorted(combined.items(), key=lambda item: (item[0][0], -len(item[1]), item[0][1])):
        n_families = len(families)
        scope = "shared" if n_families > 1 else "unique"
        summary[kind]["total"] += 1
        summary[kind][scope] += 1
        detail_rows.append(
            {
                "kind": kind,
                "identifier": ident,
                "n_families": n_families,
                "scope": scope,
                "families": "|".join(sorted(families)),
            }
        )

    detail_path = OUT / "q7_identifier_families.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["kind", "identifier", "n_families", "scope", "families"],
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_rows = [
        {
            "kind": kind,
            "total_identifiers": summary[kind]["total"],
            "unique_to_one_family": summary[kind]["unique"],
            "shared_across_families": summary[kind]["shared"],
        }
        for kind in KINDS
    ]
    summary_path = OUT / "q7_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "kind",
                "total_identifiers",
                "unique_to_one_family",
                "shared_across_families",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = list(range(len(KINDS)))
    unique_vals = [summary[kind]["unique"] for kind in KINDS]
    shared_vals = [summary[kind]["shared"] for kind in KINDS]
    width = 0.35
    ax.bar([i - width / 2 for i in x], unique_vals, width=width, color=BLUE, label="One family only")
    ax.bar([i + width / 2 for i in x], shared_vals, width=width, color=GOLD, label="Two or more families")
    ax.set_xticks(x)
    ax.set_xticklabels(["Fault code", "Part number", "Procedure name"])
    ax.set_ylabel("Distinct identifiers")
    ax.set_title("Q7. Identifiers unique to one family vs shared across families")
    ax.legend(frameon=False)
    fig.tight_layout()
    chart = OUT / "q7_shared_vs_unique.png"
    fig.savefig(chart, dpi=160)
    plt.close(fig)

    print(f"wrote {detail_path} ({len(detail_rows)} rows)", flush=True)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {chart}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
