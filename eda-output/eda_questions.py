


import os
import io
import re
import csv
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pyarrow.parquet as pq

BUCKET = "g34305547-dats6450-s2026"
PREFIX = "torque-diagnostics/chunks/"
INVENTORY = "ABB_100GB_Download_List.csv"
OUT = "eda-output/five_questions"
WORKERS = 6

os.makedirs(OUT, exist_ok=True)

s3 = boto3.client("s3", region_name="us-east-1")

def save_csv(filename, rows, columns):
    with open(os.path.join(OUT, filename), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

def family_from_key(key):
    match = re.search(r"asset_family=([^/]+)", key)
    return match.group(1) if match else "unknown"

def document_id_from_key(key):
    return os.path.basename(key).removesuffix(".parquet")

# --------------------------------------------------
# Q1 and Q3: document kinds from original inventory
# --------------------------------------------------

kind_by_id = {}
inventory_counts = Counter()

with open(INVENTORY, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    print("Inventory columns:", reader.fieldnames, flush=True)

    for row in reader:
        doc_id = row["document_id"].strip()
        kind = row.get("document_kind", "unknown").strip().lower()
        kind_by_id[doc_id] = kind
        inventory_counts[kind] += 1

save_csv(
    "document_kinds_overall.csv",
    [
        {"document_kind": kind, "pdf_count": count}
        for kind, count in inventory_counts.most_common()
    ],
    ["document_kind", "pdf_count"]
)

# --------------------------------------------------
# List Parquet files once
# --------------------------------------------------

keys = []
paginator = s3.get_paginator("list_objects_v2")

for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
    keys.extend(
        obj["Key"]
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    )

print(f"Found {len(keys):,} Parquet files", flush=True)

family_kind = Counter()
id_families = defaultdict(set)

for key in keys:
    family = family_from_key(key)
    doc_id = document_id_from_key(key)
    kind = kind_by_id.get(doc_id, "unknown")

    family_kind[(family, kind)] += 1
    id_families[doc_id].add(family)

save_csv(
    "document_kinds_by_family.csv",
    [
        {
            "asset_family": family,
            "document_kind": kind,
            "document_count": count
        }
        for (family, kind), count in sorted(family_kind.items())
    ],
    ["asset_family", "document_kind", "document_count"]
)

# --------------------------------------------------
# Q3: family composition
# --------------------------------------------------

family_totals = Counter()
datasheet_counts = Counter()
catalogue_counts = Counter()

for (family, kind), count in family_kind.items():
    family_totals[family] += count

    if "data sheet" in kind or "datasheet" in kind:
        datasheet_counts[family] += count

    if "catalog" in kind:
        catalogue_counts[family] += count

composition = []

for family, total in family_totals.items():
    ds = datasheet_counts[family]
    cat = catalogue_counts[family]

    composition.append({
        "asset_family": family,
        "documents": total,
        "data_sheets": ds,
        "catalogues": cat,
        "data_sheet_pct": round(100 * ds / total, 2),
        "catalogue_pct": round(100 * cat / total, 2),
        "combined_pct": round(100 * (ds + cat) / total, 2)
    })

composition.sort(
    key=lambda r: (-r["combined_pct"], -r["documents"])
)

save_csv(
    "family_document_composition.csv",
    composition,
    [
        "asset_family", "documents", "data_sheets",
        "catalogues", "data_sheet_pct",
        "catalogue_pct", "combined_pct"
    ]
)

# --------------------------------------------------
# Q5: duplicate document IDs
# --------------------------------------------------

duplicates = []

for doc_id, families in id_families.items():
    if len(families) > 1:
        duplicates.append({
            "document_id": doc_id,
            "family_count": len(families),
            "families": "; ".join(sorted(families))
        })

duplicates.sort(key=lambda r: -r["family_count"])

save_csv(
    "duplicate_document_ids.csv",
    duplicates,
    ["document_id", "family_count", "families"]
)

# --------------------------------------------------
# Q2: read only text and page-number columns
# --------------------------------------------------

def analyze_parquet(key):
    family = family_from_key(key)
    doc_id = document_id_from_key(key)
    kind = kind_by_id.get(doc_id, "unknown")

    obj = s3.get_object(Bucket=BUCKET, Key=key)
    data = obj["Body"].read()
    pf = pq.ParquetFile(io.BytesIO(data))

    names = pf.schema_arrow.names

    text_col = next(
        (x for x in ["text", "passage_text", "content", "passage"]
         if x in names),
        None
    )

    page_col = next(
        (x for x in ["page_number", "page", "page_num"]
         if x in names),
        None
    )

    if text_col is None:
        return None

    cols = [text_col]
    if page_col:
        cols.append(page_col)

    table = pf.read(columns=cols)
    texts = table[text_col].to_pylist()

    lengths = [
        len(str(t))
        for t in texts
        if t is not None and str(t).strip()
    ]

    pages = set()

    if page_col:
        pages = {
            p for p in table[page_col].to_pylist()
            if p is not None
        }

    return {
        "asset_family": family,
        "document_id": doc_id,
        "document_kind": kind,
        "passages": len(lengths),
        "total_characters": sum(lengths),
        "mean_passage_chars": round(statistics.mean(lengths), 2)
            if lengths else 0,
        "median_passage_chars": statistics.median(lengths)
            if lengths else 0,
        "max_passage_chars": max(lengths) if lengths else 0,
        "distinct_text_pages": len(pages)
    }

results = []
failures = []

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {
        executor.submit(analyze_parquet, key): key
        for key in keys
    }

    for i, future in enumerate(as_completed(futures), 1):
        key = futures[future]

        try:
            result = future.result()
            if result:
                results.append(result)
            else:
                failures.append((key, "No recognized text column"))
        except Exception as e:
            failures.append((key, str(e)))

        if i % 1000 == 0:
            print(f"Analyzed {i:,}/{len(keys):,}", flush=True)

columns = [
    "asset_family", "document_id", "document_kind",
    "passages", "total_characters",
    "mean_passage_chars", "median_passage_chars",
    "max_passage_chars", "distinct_text_pages"
]

save_csv("document_passage_statistics.csv", results, columns)

# Family and document-kind weighted summaries
for grouping, filename in [
    ("asset_family", "passage_statistics_by_family.csv"),
    ("document_kind", "passage_statistics_by_kind.csv")
]:
    groups = defaultdict(list)

    for row in results:
        groups[row[grouping]].append(row)

    summaries = []

    for name, rows in groups.items():
        passages = sum(r["passages"] for r in rows)
        chars = sum(r["total_characters"] for r in rows)
        pages = [r["distinct_text_pages"] for r in rows]

        summaries.append({
            grouping: name,
            "documents": len(rows),
            "passages": passages,
            "mean_passage_chars": round(chars / passages, 2)
                if passages else 0,
            "mean_distinct_text_pages_per_document":
                round(statistics.mean(pages), 2),
            "median_distinct_text_pages_per_document":
                statistics.median(pages)
        })

    save_csv(
        filename,
        summaries,
        [
            grouping, "documents", "passages",
            "mean_passage_chars",
            "mean_distinct_text_pages_per_document",
            "median_distinct_text_pages_per_document"
        ]
    )

save_csv(
    "read_failures.csv",
    [{"key": k, "error": e} for k, e in failures],
    ["key", "error"]
)

print("\n===== RESULTS =====")
print(f"Documents inspected: {len(keys):,}")
print(f"Unique document IDs: {len(id_families):,}")
print(f"IDs in multiple families: {len(duplicates):,}")
print(f"Files successfully analyzed: {len(results):,}")
print(f"Read failures: {len(failures):,}")

print("\nTop families dominated by data sheets/catalogues:")
for row in composition[:10]:
    print(row)

print(f"\nAll CSV files saved to: {OUT}")
