#!/usr/bin/env python3
"""Answer the five EDA questions from the passage Parquet in S3.

Run from the repository root. Credentials come from the environment.

    uv run --with matplotlib python eda-output/answer_questions.py

The script reads ABB_100GB_Download_List.csv and every passage file under
the chunks prefix. It writes one chart and one table per question into
eda-output/.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asset_layout import asset_family

BUCKET = os.environ.get("EDA_BUCKET", "dats-nlp")
PREFIX = os.environ.get(
    "EDA_PREFIX",
    "dhanush.mathivanan@gwu.edu/torque-diagnostics/chunks/",
).strip("/") + "/"
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
INVENTORY = ROOT / "ABB_100GB_Download_List.csv"
OUT = ROOT / "eda-output"
CHECKPOINT = OUT / "_document_stats.jsonl"
WORKERS = int(os.environ.get("EDA_WORKERS", "12"))
COLUMNS = [
    "text",
    "page_number",
    "tracing_page",
    "fault_code",
    "part",
    "procedure",
]
BLUE = "#1f4e79"
GOLD = "#c59b3a"
GRAY = "#7f8c9b"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def present(value) -> bool:
    return value is not None and str(value).strip() != ""


def save_csv(name: str, rows: list[dict], columns: list[str]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value, digits: int) -> str:
    if digits <= 0:
        return f"{value:,.0f}"
    return f"{value:,.{digits}f}"


def horizontal(labels, values, title, xlabel, filename, color=BLUE, digits=0):
    fig, ax = plt.subplots(figsize=(10.5, max(4.2, 0.42 * len(labels) + 1.4)))
    positions = list(range(len(labels) - 1, -1, -1))
    ax.barh(positions, values, color=color, height=0.72)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    span = max(values) if values else 1
    ax.set_xlim(0, span * 1.22)
    for index, value in zip(positions, values):
        ax.text(value + span * 0.015, index, format_value(value, digits), va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=160)
    plt.close(fig)


def bars(labels, values, title, ylabel, filename, color=BLUE, digits=0):
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(labels) + 2), 5))
    plot = ax.bar(labels, values, color=color, width=0.68)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    top = max(values) if values else 1
    ax.set_ylim(0, top * 1.18)
    for bar, value in zip(plot, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            format_value(value, digits),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=160)
    plt.close(fig)


def storage_stem(document_id: str) -> str:
    """Match the S3 file name. Slashes and spaces in a document id become underscores."""
    return document_id.strip().replace("/", "_").replace(" ", "_")


def is_passage_key(key: str) -> bool:
    if "asset_family=" not in key or not key.endswith(".parquet"):
        return False
    parts = key.split("asset_family=", 1)[1].split("/")
    if len(parts) == 2:
        return True
    return len(parts) >= 3 and parts[1] == "text"


def load_inventory() -> dict[str, dict]:
    rows = {}
    with INVENTORY.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            document_id = row["document_id"].strip()
            kind = (row.get("document_kind") or "unknown").strip() or "unknown"
            family = asset_family(row.get("title", ""), document_id, kind)
            rows[document_id] = {"document_kind": kind, "layout_family": family}
    return rows


def list_keys(s3) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        keys.extend(
            item["Key"]
            for item in page.get("Contents", [])
            if is_passage_key(str(item["Key"]))
        )
    return keys


def family_from_key(key: str) -> str:
    marker = "asset_family="
    start = key.find(marker)
    if start < 0:
        return "unknown"
    return key[start + len(marker):].split("/", 1)[0]


def document_id_from_key(key: str) -> str:
    return Path(key).name.removesuffix(".parquet")


def summarize(s3, key: str) -> dict:
    body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    table = pq.read_table(io.BytesIO(body), columns=COLUMNS)
    texts = table.column("text").to_pylist()
    pages = table.column("page_number").to_pylist()
    tracing = table.column("tracing_page").to_pylist()
    faults = table.column("fault_code").to_pylist()
    parts = table.column("part").to_pylist()
    procedures = table.column("procedure").to_pylist()
    characters = 0
    passage_count = 0
    text_pages = set()
    tracing_passages = fault_passages = part_passages = procedure_passages = 0
    for text, page, is_tracing, fault, part, procedure in zip(
        texts, pages, tracing, faults, parts, procedures
    ):
        if text is None or not str(text).strip():
            continue
        passage_count += 1
        characters += len(str(text))
        if page is not None:
            text_pages.add(int(page))
        if bool(is_tracing):
            tracing_passages += 1
        if present(fault):
            fault_passages += 1
        if present(part):
            part_passages += 1
        if present(procedure):
            procedure_passages += 1
    return {
        "key": key,
        "document_id": document_id_from_key(key),
        "asset_family": family_from_key(key),
        "passages": passage_count,
        "characters": characters,
        "text_pages": len(text_pages),
        "tracing_passages": tracing_passages,
        "fault_passages": fault_passages,
        "part_passages": part_passages,
        "procedure_passages": procedure_passages,
    }


def load_checkpoint() -> dict[str, dict]:
    found = {}
    if not CHECKPOINT.is_file():
        return found
    with CHECKPOINT.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            found[row["key"]] = row
    return found


def scan(s3, keys: list[str]) -> list[dict]:
    done = load_checkpoint()
    pending = [key for key in keys if key not in done]
    print(f"parquet={len(keys):,} already_read={len(done):,} remaining={len(pending):,}", flush=True)
    if pending:
        with CHECKPOINT.open("a", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = {pool.submit(summarize, s3, key): key for key in pending}
                finished = 0
                for future in as_completed(futures):
                    key = futures[future]
                    row = future.result()
                    done[key] = row
                    handle.write(json.dumps(row) + "\n")
                    finished += 1
                    if finished % 500 == 0 or finished == len(pending):
                        handle.flush()
                        print(f"read {finished:,}/{len(pending):,}", flush=True)
    return [done[key] for key in keys if key in done]


def weighted_mean(total: float, count: float) -> float:
    return round(total / count, 2) if count else 0.0


def copies_for(by_id: dict[str, list[dict]], document_id: str) -> list[dict]:
    return by_id.get(document_id) or by_id.get(storage_stem(document_id)) or []


def build(inventory: dict[str, dict], stats: list[dict]) -> None:
    kind_files = Counter(row["document_kind"] for row in inventory.values())
    ids_for_stem: dict[str, list[str]] = defaultdict(list)
    for document_id in inventory:
        ids_for_stem[storage_stem(document_id)].append(document_id)
    by_id: dict[str, list[dict]] = defaultdict(list)
    for row in stats:
        by_id[row["document_id"]].append(row)

    def kind_for(stored_id: str) -> str:
        if stored_id in inventory:
            return inventory[stored_id]["document_kind"]
        matches = ids_for_stem.get(stored_id, [])
        if not matches:
            return "unknown"
        # Two inventory ids can share one stored file. Count that file once.
        chosen = sorted(matches)[0]
        return inventory[chosen]["document_kind"]

    # A chunked file lives in the folder of its Parquet key. An unchunked
    # inventory row keeps the folder assigned from its title.
    folder_of = {}
    for document_id, info in inventory.items():
        copies = copies_for(by_id, document_id)
        families = {row["asset_family"] for row in copies}
        if len(families) == 1:
            folder_of[document_id] = next(iter(families))
        else:
            folder_of[document_id] = info["layout_family"]

    file_folders = Counter(folder_of[document_id] for document_id in inventory)

    family_passages = Counter()
    family_chars = Counter()
    family_pages = defaultdict(list)
    kind_passages = Counter()
    kind_chars = Counter()
    kind_pages = defaultdict(list)
    kind_docs = Counter()
    family_docs_with_text = Counter()
    signals = defaultdict(lambda: Counter())
    tracing = Counter()
    passages = Counter()

    for document_id, copies in by_id.items():
        if len(copies) != 1:
            continue
        row = copies[0]
        family = row["asset_family"]
        kind = kind_for(document_id)
        family_passages[family] += row["passages"]
        family_chars[family] += row["characters"]
        family_pages[family].append(row["text_pages"])
        kind_passages[kind] += row["passages"]
        kind_chars[kind] += row["characters"]
        kind_pages[kind].append(row["text_pages"])
        kind_docs[kind] += 1
        family_docs_with_text[family] += 1
        for name in ("fault_passages", "part_passages", "procedure_passages", "tracing_passages"):
            signals[family][name] += row[name]
        tracing[family] += row["tracing_passages"]
        passages[family] += row["passages"]

    # Question 1
    kind_rows = [
        {"document_kind": kind, "pdf_count": count}
        for kind, count in kind_files.most_common()
    ]
    save_csv("q1_pdfs_by_kind.csv", kind_rows, ["document_kind", "pdf_count"])
    top_kinds = kind_rows[:12]
    other_pdfs = sum(row["pdf_count"] for row in kind_rows[12:])
    if other_pdfs:
        top_kinds = [*top_kinds, {"document_kind": "Other kinds", "pdf_count": other_pdfs}]
    top_kinds = sorted(top_kinds, key=lambda row: -row["pdf_count"])
    horizontal(
        [row["document_kind"] for row in top_kinds],
        [row["pdf_count"] for row in top_kinds],
        f"Question 1. PDF count by document kind ({sum(kind_files.values()):,} PDFs)",
        "PDFs",
        "q1_pdfs_by_kind.png",
    )
    folder_rows = [
        {
            "asset_family": family,
            "pdf_count": file_folders[family],
            "passages": family_passages[family],
        }
        for family in sorted(file_folders, key=lambda name: (-file_folders[name], name))
    ]
    save_csv("q1_folders.csv", folder_rows, ["asset_family", "pdf_count", "passages"])
    top_files = folder_rows[:12]
    horizontal(
        [row["asset_family"] for row in top_files],
        [row["pdf_count"] for row in top_files],
        f"Question 1. Folders with the most PDFs ({len(file_folders)} folders)",
        "PDFs",
        "q1_folders_by_files.png",
    )
    top_passages = sorted(folder_rows, key=lambda row: (-row["passages"], row["asset_family"]))[:12]
    horizontal(
        [row["asset_family"] for row in top_passages],
        [row["passages"] for row in top_passages],
        "Question 1. Folders with the most passages",
        "Passages",
        "q1_folders_by_passages.png",
        color=GOLD,
    )

    # Question 2
    kind_stat_rows = []
    for kind, count in kind_docs.most_common():
        pages = kind_pages[kind]
        kind_stat_rows.append({
            "document_kind": kind,
            "documents": count,
            "passages": kind_passages[kind],
            "mean_passage_chars": weighted_mean(kind_chars[kind], kind_passages[kind]),
            "mean_text_pages": weighted_mean(sum(pages), len(pages)),
        })
    save_csv(
        "q2_by_kind.csv",
        kind_stat_rows,
        ["document_kind", "documents", "passages", "mean_passage_chars", "mean_text_pages"],
    )
    shown_kinds = kind_stat_rows[:12]
    horizontal(
        [row["document_kind"] for row in shown_kinds],
        [row["mean_passage_chars"] for row in shown_kinds],
        "Question 2. Mean passage length by document kind",
        "Mean characters per passage",
        "q2_length_by_kind.png",
        digits=0,
    )
    horizontal(
        [row["document_kind"] for row in shown_kinds],
        [row["mean_text_pages"] for row in shown_kinds],
        "Question 2. Mean text pages per document, by document kind",
        "Mean pages that contain passage text",
        "q2_pages_by_kind.png",
        digits=1,
    )
    family_stat_rows = []
    for family, count in family_docs_with_text.most_common():
        pages = family_pages[family]
        family_stat_rows.append({
            "asset_family": family,
            "documents": count,
            "passages": family_passages[family],
            "mean_passage_chars": weighted_mean(family_chars[family], family_passages[family]),
            "mean_text_pages": weighted_mean(sum(pages), len(pages)),
        })
    save_csv(
        "q2_by_family.csv",
        family_stat_rows,
        ["asset_family", "documents", "passages", "mean_passage_chars", "mean_text_pages"],
    )
    shown_families = family_stat_rows[:12]
    horizontal(
        [row["asset_family"] for row in shown_families],
        [row["mean_passage_chars"] for row in shown_families],
        "Question 2. Mean passage length for the 12 folders with the most PDFs",
        "Mean characters per passage",
        "q2_length_by_family.png",
    )
    horizontal(
        [row["asset_family"] for row in shown_families],
        [row["mean_text_pages"] for row in shown_families],
        "Question 2. Mean text pages for the 12 folders with the most PDFs",
        "Mean pages that contain passage text",
        "q2_pages_by_family.png",
        color=GOLD,
        digits=1,
    )

    # Question 3
    signal_rows = []
    for family, count in family_docs_with_text.most_common():
        signal_rows.append({
            "asset_family": family,
            "documents": count,
            "passages": passages[family],
            "fault_passages": signals[family]["fault_passages"],
            "part_passages": signals[family]["part_passages"],
            "procedure_passages": signals[family]["procedure_passages"],
        })
    save_csv(
        "q3_signals_by_family.csv",
        signal_rows,
        ["asset_family", "documents", "passages", "fault_passages", "part_passages", "procedure_passages"],
    )
    ranked_signals = sorted(
        signal_rows,
        key=lambda row: -(row["fault_passages"] + row["part_passages"] + row["procedure_passages"]),
    )[:12]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    y = list(range(len(ranked_signals)))[::-1]
    labels = [row["asset_family"] for row in ranked_signals]
    fault_vals = [row["fault_passages"] for row in ranked_signals]
    part_vals = [row["part_passages"] for row in ranked_signals]
    procedure_vals = [row["procedure_passages"] for row in ranked_signals]
    ax.barh(y, fault_vals, color=BLUE, height=0.7, label="Fault code")
    ax.barh(y, part_vals, left=fault_vals, color=GOLD, height=0.7, label="Part number")
    left = [a + b for a, b in zip(fault_vals, part_vals)]
    ax.barh(y, procedure_vals, left=left, color=GRAY, height=0.7, label="Named procedure")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Passages")
    ax.set_title("Question 3. Families with the most fault, part, and procedure passages")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "q3_signals_by_family.png", dpi=160)
    plt.close(fig)

    composition = []
    kind_by_family = defaultdict(Counter)
    for document_id, family in folder_of.items():
        if document_id not in inventory:
            continue
        kind_by_family[family][inventory[document_id]["document_kind"].lower()] += 1
    for family, counts in kind_by_family.items():
        total = sum(counts.values())
        data_sheets = sum(count for kind, count in counts.items() if "data sheet" in kind or "datasheet" in kind)
        catalogues = sum(count for kind, count in counts.items() if "catalog" in kind)
        composition.append({
            "asset_family": family,
            "documents": total,
            "data_sheets": data_sheets,
            "catalogues": catalogues,
            "data_sheet_or_catalogue_pct": round(100 * (data_sheets + catalogues) / total, 2) if total else 0,
        })
    composition.sort(key=lambda row: (-row["data_sheet_or_catalogue_pct"], -row["documents"]))
    save_csv(
        "q3_datasheet_catalogue.csv",
        composition,
        ["asset_family", "documents", "data_sheets", "catalogues", "data_sheet_or_catalogue_pct"],
    )
    mostly = composition[:12]

    def share_label(row: dict) -> str:
        count = row["documents"]
        unit = "PDF" if count == 1 else "PDFs"
        return f"{row['asset_family']} ({count:,} {unit})"

    horizontal(
        [share_label(row) for row in mostly],
        [row["data_sheet_or_catalogue_pct"] for row in mostly],
        "Question 3. Data-sheet and catalogue share, highest folders",
        "Percent of PDFs",
        "q3_datasheet_catalogue.png",
        color=GRAY,
        digits=1,
    )

    # Question 4
    def group_name(family: str) -> str | None:
        if family == "library":
            return "library"
        if family == "ACS860":
            return "ACS860"
        if family == "ACS880" or family.startswith("ACS880-"):
            return "ACS880"
        return None

    groups = ["ACS880", "ACS860", "library"]
    group_tracing = Counter()
    group_passages = Counter()
    for family, count in passages.items():
        name = group_name(family)
        if name is None:
            continue
        group_tracing[name] += tracing[family]
        group_passages[name] += count
    tracing_rows = []
    for name in groups:
        total = group_passages[name]
        traced = group_tracing[name]
        tracing_rows.append({
            "group": name,
            "passages": total,
            "tracing_passages": traced,
            "tracing_pct": round(100 * traced / total, 2) if total else 0,
        })
    save_csv("q4_tracing_share.csv", tracing_rows, ["group", "passages", "tracing_passages", "tracing_pct"])
    bars(
        [row["group"] for row in tracing_rows],
        [row["tracing_pct"] for row in tracing_rows],
        "Question 4. Share of passages on a tracing page",
        "Percent of passages",
        "q4_tracing_share.png",
        digits=2,
    )

    # Question 5. Each inventory id is checked against the folder of its stored file.
    duplicates = []
    for document_id in sorted(inventory):
        families = {row["asset_family"] for row in copies_for(by_id, document_id)}
        if len(families) > 1:
            duplicates.append({
                "document_id": document_id,
                "family_count": len(families),
                "families": "; ".join(sorted(families)),
            })
    save_csv("q5_duplicate_ids.csv", duplicates, ["document_id", "family_count", "families"])
    shared_rows = []
    for stem, document_ids in sorted(ids_for_stem.items()):
        if len(document_ids) < 2:
            continue
        for document_id in sorted(document_ids):
            shared_rows.append({
                "stored_name": stem,
                "document_id": document_id,
                "document_kind": inventory[document_id]["document_kind"],
                "asset_family": folder_of[document_id],
            })
    save_csv(
        "q5_shared_filename.csv",
        shared_rows,
        ["stored_name", "document_id", "document_kind", "asset_family"],
    )
    many = len(duplicates)
    one_family = len(inventory) - many
    save_csv(
        "q5_summary.csv",
        [{
            "inventory_ids": len(inventory),
            "stored_files": len(by_id),
            "ids_in_one_family": one_family,
            "ids_in_more_than_one_family": many,
            "stored_names_shared_by_ids": len({row["stored_name"] for row in shared_rows}),
        }],
        [
            "inventory_ids",
            "stored_files",
            "ids_in_one_family",
            "ids_in_more_than_one_family",
            "stored_names_shared_by_ids",
        ],
    )
    bars(
        ["One family", "More than one family"],
        [one_family, many],
        "Question 5. Document ids stored in one folder or more than one",
        "Document ids",
        "q5_document_ids.png",
        color=GOLD,
        digits=0,
    )
    print(
        f"folders={len(file_folders)} pdfs={sum(file_folders.values())} "
        f"stored_files={len(by_id)} passages={sum(passages.values())} "
        f"ids_in_more_than_one_family={many} "
        f"shared_filenames={len({row['stored_name'] for row in shared_rows})}",
        flush=True,
    )
    for row in tracing_rows:
        print(
            f"tracing {row['group']}: {row['tracing_passages']}/{row['passages']} = {row['tracing_pct']}%",
            flush=True,
        )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = load_inventory()
    s3 = boto3.client("s3", region_name=REGION)
    keys = list_keys(s3)
    if not keys:
        print(f"No passage files at s3://{BUCKET}/{PREFIX}", file=sys.stderr)
        return 1
    stats = scan(s3, keys)
    build(inventory, stats)
    if CHECKPOINT.is_file():
        CHECKPOINT.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
