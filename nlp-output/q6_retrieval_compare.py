#!/usr/bin/env python3
"""Proposal Q6 — BM25 vs embedding vs hybrid pages for two sample questions.

Writes nlp-output/q6_retrieval_compare.csv and q6_summary.md.

    uv run python nlp-output/q6_retrieval_compare.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "data-indexing" / "query"))
sys.path.insert(0, str(ROOT / "data-indexing" / "bm25"))
sys.path.insert(0, str(ROOT / "data-indexing" / "embeddings"))
sys.path.insert(0, str(ROOT / "data-indexing" / "tokenization"))

from build_bm25 import search  # noqa: E402
from build_embeddings import encode_texts, load_encoder, search_vectors  # noqa: E402
from manual_tokens import route_families  # noqa: E402
from query_bm25 import load_payload  # noqa: E402
from query_hybrid import fuse, load_embeddings  # noqa: E402

QUESTIONS = [
    {
        "id": "flange-tools",
        "question": "What tools are needed to flange-mount an ACS880-11?",
    },
    {
        "id": "acs860-reset",
        "question": "Which faults can the ACS860 reset automatically?",
    },
]
TOP_K = int(os.environ.get("TOP_K", "10"))


def compare_one(question_id: str, question: str, encoder, k: int) -> list[dict]:
    families = route_families(question)
    rows: list[dict] = []
    for family in families:
        bm25_hits = search(load_payload(family), question, max(k, 10**9))
        matrix, embed_rows = load_embeddings(family)
        query_vector = encode_texts(encoder, [question], query=True)[0]
        embed_hits = search_vectors(matrix, embed_rows, query_vector, len(embed_rows) or k)
        hybrid = fuse(bm25_hits, embed_hits, k)

        bm25_by_id = {hit["chunk_id"]: (rank, hit) for rank, hit in enumerate(bm25_hits[:k], start=1)}
        embed_by_id = {hit["chunk_id"]: (rank, hit) for rank, hit in enumerate(embed_hits[:k], start=1)}
        hybrid_by_id = {hit["chunk_id"]: (rank, hit) for rank, hit in enumerate(hybrid, start=1)}

        chunk_ids = list(
            dict.fromkeys(
                [hit["chunk_id"] for hit in bm25_hits[:k]]
                + [hit["chunk_id"] for hit in embed_hits[:k]]
                + [hit["chunk_id"] for hit in hybrid]
            )
        )
        meta = {}
        for hit in bm25_hits + embed_hits + hybrid:
            meta.setdefault(hit["chunk_id"], hit)

        for chunk_id in chunk_ids:
            hit = meta[chunk_id]
            bm25_rank = bm25_by_id.get(chunk_id, (None, None))[0]
            embed_rank = embed_by_id.get(chunk_id, (None, None))[0]
            hybrid_rank = hybrid_by_id.get(chunk_id, (None, None))[0]
            rows.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "asset_family": family,
                    "document_id": hit["document_id"],
                    "page_number": hit["page_number"],
                    "chunk_id": chunk_id,
                    "bm25_rank": bm25_rank if bm25_rank is not None else "",
                    "embed_rank": embed_rank if embed_rank is not None else "",
                    "hybrid_rank": hybrid_rank if hybrid_rank is not None else "",
                    "hybrid_score": f"{hybrid_by_id[chunk_id][1]['score']:.4f}"
                    if chunk_id in hybrid_by_id
                    else "",
                }
            )
    return rows


def write_summary(rows: list[dict]) -> None:
    lines = [
        "# Q6 — Retrieval compare summary",
        "",
        "For each sample question, BM25, BGE embedding search, and hybrid (scaled BM25 + cosine) return ranked pages.",
        f"Top-k listed per channel: **{TOP_K}**.",
        "",
    ]
    for item in QUESTIONS:
        qid = item["id"]
        subset = [row for row in rows if row["question_id"] == qid]
        lines.append(f"## {qid}")
        lines.append("")
        lines.append(f"Question: {item['question']}")
        if not subset:
            lines.append("")
            lines.append("_No rows — run on EC2 with AWS indexes._")
            lines.append("")
            continue
        families = sorted({row["asset_family"] for row in subset})
        lines.append(f"Routed families: {', '.join(families)}")
        lines.append("")
        lines.append("| hybrid_rank | document_id | page | bm25_rank | embed_rank |")
        lines.append("|---:|---|---:|---:|---:|")
        hybrid_rows = [row for row in subset if row["hybrid_rank"] != ""]
        hybrid_rows.sort(key=lambda row: int(row["hybrid_rank"]))
        for row in hybrid_rows[:TOP_K]:
            lines.append(
                f"| {row['hybrid_rank']} | {row['document_id']} | {row['page_number']} | "
                f"{row['bm25_rank'] or '—'} | {row['embed_rank'] or '—'} |"
            )
        lines.append("")
    (OUT / "q6_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    encoder = load_encoder()
    rows: list[dict] = []
    for item in QUESTIONS:
        print(f"q6: {item['id']}", flush=True)
        rows.extend(compare_one(item["id"], item["question"], encoder, TOP_K))

    fields = [
        "question_id",
        "question",
        "asset_family",
        "document_id",
        "page_number",
        "chunk_id",
        "bm25_rank",
        "embed_rank",
        "hybrid_rank",
        "hybrid_score",
    ]
    path = OUT / "q6_retrieval_compare.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_summary(rows)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)
    print(f"wrote {OUT / 'q6_summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
