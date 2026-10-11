#!/usr/bin/env python3
"""Merge the BM25 ranking and the embedding ranking for one question.

Each index scores every passage in the folder. Those scores are scaled to
the same range and added. A page that names the exact tools stays ahead of
a page that only looks related. The printed rows are still pages, with the
document id and page. This does not call a language model.

    QUESTION='socket set and hex key set for flange-mounting ACS880-11' \\
        uv run python data-indexing/query/query_hybrid.py

INDEX_FAMILY overrides the folder router. Otherwise the question names
the folder. A question with no series reads library.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "bm25"))
sys.path.insert(0, str(_HERE.parent / "embeddings"))
sys.path.insert(0, str(_HERE.parent / "tokenization"))

from build_bm25 import BUCKET, REGION, search
from build_embeddings import embed_key, encode_texts, load_encoder, read_table, search_vectors
from manual_tokens import route_families
from query_bm25 import load_payload

def load_embeddings(family: str):
    local = os.environ.get("EMBED_LOCAL", "").strip()
    if local:
        return read_table(str(Path(local) / f"asset_family={family}.parquet"))
    import boto3

    s3 = boto3.client("s3", region_name=REGION)
    handle, path = tempfile.mkstemp(suffix=".parquet")
    os.close(handle)
    try:
        s3.download_file(BUCKET, embed_key(family), path)
        return read_table(path)
    finally:
        os.remove(path)


def _scale(hits: list[dict]) -> dict[str, float]:
    if not hits:
        return {}
    peak = max(float(hit["score"]) for hit in hits)
    if peak <= 0:
        return {}
    return {hit["chunk_id"]: float(hit["score"]) / peak for hit in hits}


def fuse(bm25_hits: list[dict], embed_hits: list[dict], k: int) -> list[dict]:
    """Add the two scores after scaling each list by its own highest score."""
    bm25_score = _scale(bm25_hits)
    embed_score = _scale(embed_hits)
    bm25_rank = {hit["chunk_id"]: rank for rank, hit in enumerate(bm25_hits, start=1)}
    embed_rank = {hit["chunk_id"]: rank for rank, hit in enumerate(embed_hits, start=1)}
    meta: dict[str, dict] = {}
    for hit in bm25_hits + embed_hits:
        meta.setdefault(hit["chunk_id"], hit)
    rows = []
    for chunk_id, hit in meta.items():
        rows.append(
            {
                "chunk_id": chunk_id,
                "document_id": hit["document_id"],
                "page_number": hit["page_number"],
                "score": bm25_score.get(chunk_id, 0.0) + embed_score.get(chunk_id, 0.0),
                "bm25_rank": bm25_rank.get(chunk_id, 0),
                "embed_rank": embed_rank.get(chunk_id, 0),
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows[:k]


def main() -> int:
    question = os.environ.get("QUESTION", "").strip()
    if not question:
        print("Set QUESTION to the technician question.", file=sys.stderr)
        return 1
    override = os.environ.get("INDEX_FAMILY", "").strip()
    families = [override] if override else route_families(question)
    k = int(os.environ.get("TOP_K", "5"))
    encoder = load_encoder()
    query_vector = encode_texts(encoder, [question], query=True)[0]
    print(f"folders: {', '.join(families)}")
    for family in families:
        print(f"\n{family}")
        try:
            bm25_hits = search(load_payload(family), question, max(k, 10**9))
            matrix, rows = load_embeddings(family)
            embed_hits = search_vectors(matrix, rows, query_vector, len(rows) or k)
        except Exception as exc:
            print(f"missing index for {family}: {exc}")
            continue
        hits = fuse(bm25_hits, embed_hits, k)
        if not hits:
            print("no matching passage")
            continue
        for rank, hit in enumerate(hits, start=1):
            print(
                f"{rank}. rrf={hit['score']:.4f} "
                f"bm25_rank={hit['bm25_rank']} embed_rank={hit['embed_rank']} "
                f"document={hit['document_id']} page={hit['page_number']} "
                f"chunk={hit['chunk_id']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
