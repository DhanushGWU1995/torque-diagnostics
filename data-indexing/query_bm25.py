#!/usr/bin/env python3
"""Return the BM25 passages for one technician question.

This reads the index written by build_bm25.py. It does not call a language
model. The printed rows are the pages a later reply step is allowed to use.

    QUESTION='necessary tools for flange mounting ACS880-11' \\
        uv run python data-indexing/query_bm25.py

INDEX_FAMILY overrides the folder router. Otherwise the question names
the folder. A question with no series reads library.
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_bm25 import BUCKET, REGION, bm25_key, search
from manual_tokens import route_families


def load_payload(family: str) -> dict:
    local = os.environ.get("BM25_LOCAL", "").strip()
    if local:
        path = Path(local) / f"asset_family={family}.pkl"
        with path.open("rb") as handle:
            return pickle.load(handle)
    import boto3

    s3 = boto3.client("s3", region_name=REGION)
    handle, path = tempfile.mkstemp(suffix=".pkl")
    os.close(handle)
    try:
        s3.download_file(BUCKET, bm25_key(family), path)
        with open(path, "rb") as stored:
            return pickle.load(stored)
    finally:
        os.remove(path)


def main() -> int:
    question = os.environ.get("QUESTION", "").strip()
    if not question:
        print("Set QUESTION to the technician question.", file=sys.stderr)
        return 1
    override = os.environ.get("INDEX_FAMILY", "").strip()
    families = [override] if override else route_families(question)
    k = int(os.environ.get("TOP_K", "5"))
    print(f"folders: {', '.join(families)}")
    for family in families:
        print(f"\n{family}")
        try:
            hits = search(load_payload(family), question, k)
        except Exception as exc:
            print(f"missing index for {family}: {exc}")
            continue
        if not hits:
            print("no matching passage")
            continue
        for rank, hit in enumerate(hits, start=1):
            print(
                f"{rank}. score={hit['score']:.3f} "
                f"document={hit['document_id']} page={hit['page_number']} "
                f"chunk={hit['chunk_id']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
