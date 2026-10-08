#!/usr/bin/env python3
"""Build one BM25 word index per equipment folder.

This reads the token Parquet written by build_tokens.py. It does not open
the PDFs and it does not embed the passages. The question is tokenized
with the same BM25 tokenizer used to build the file, so a fault code, a
document id, and a hyphenated phrase such as flange-mounting stay whole.

Set INDEX_FAMILY to one folder for a test. INDEX_FAMILY=all writes every
folder. A folder that already has a finished index is skipped. library is
large, so run it after the smaller folders have stored.

Credentials come from the environment. This file does not contain an
access key.

    INDEX_FAMILY=ACS880-11 uv run python data-indexing/bm25/build_bm25.py

On the EC2 instance, from ~/dl-venv/nlp_project:

    INDEX_FAMILY=ACS880-11 ~/dl-venv/bin/python data-indexing/bm25/build_bm25.py

Each folder is written to

    s3://dats-nlp/dhanush.mathivanan@gwu.edu/torque-diagnostics/index/bm25/asset_family=<family>.pkl
"""

from __future__ import annotations

import os
import pickle
import re
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "tokenization"))

from manual_tokens import bm25_tokens


def _import_layout() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "asset_layout.py").is_file():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return
    raise ImportError("asset_layout.py was not found above this script")


_import_layout()
from asset_layout import family_rank

BUCKET = "dats-nlp"
REGION = "us-east-1"
TOKEN_ROOT = "dhanush.mathivanan@gwu.edu/torque-diagnostics/index/tokens"
BM25_ROOT = "dhanush.mathivanan@gwu.edu/torque-diagnostics/index/bm25"
STATUS_NAME = "bm25_status.txt"
_FAMILY_NAME = re.compile(r"[A-Za-z0-9_.+-]+")


def bm25_key(family: str) -> str:
    return f"{BM25_ROOT}/asset_family={family}.pkl"


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


def finished(s3, key: str) -> bool:
    try:
        response = s3.get_object(Bucket=BUCKET, Key=key, Range="bytes=0-1")
    except Exception:
        return False
    head = response["Body"].read()
    return len(head) >= 2 and head[0] == 0x80


def families_in_bucket(s3) -> list[str]:
    found: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{TOKEN_ROOT}/", Delimiter="/"):
        for item in page.get("Contents", []):
            name = item["Key"].rsplit("asset_family=", 1)[-1]
            if name.endswith(".parquet"):
                family = name[: -len(".parquet")]
                if _FAMILY_NAME.fullmatch(family):
                    found.append(family)
    if not found:
        for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{TOKEN_ROOT}/"):
            for item in page.get("Contents", []):
                name = item["Key"].rsplit("asset_family=", 1)[-1]
                if name.endswith(".parquet"):
                    family = name[: -len(".parquet")]
                    if _FAMILY_NAME.fullmatch(family):
                        found.append(family)
    found.sort(key=family_rank)
    return found


def selected_families(s3) -> list[str]:
    raw = os.environ.get("INDEX_FAMILY", "").strip()
    if not raw:
        print("Set INDEX_FAMILY to one folder, for example ACS880-11, or to all.", file=sys.stderr)
        return []
    if raw == "all":
        return families_in_bucket(s3)
    if not _FAMILY_NAME.fullmatch(raw):
        print(f"INDEX_FAMILY is not a folder name: {raw}", file=sys.stderr)
        return []
    return [raw]


def load_token_rows(connection, family: str) -> list[tuple]:
    local = os.environ.get("TOKEN_PARQUET", "").strip()
    if local:
        glob = str(Path(local) / f"asset_family={family}.parquet")
    else:
        glob = f"s3://{BUCKET}/{TOKEN_ROOT}/asset_family={family}.parquet"
    escaped = glob.replace("'", "''")
    return connection.execute(
        f"""
        SELECT chunk_id, document_id, page_number, bm25_tokens
        FROM read_parquet('{escaped}', hive_partitioning = false)
        ORDER BY chunk_id
        """
    ).fetchall()


def build_payload(rows: list[tuple]) -> dict:
    from rank_bm25 import BM25Okapi

    chunk_ids: list[str] = []
    document_ids: list[str] = []
    pages: list[int] = []
    corpus: list[list[str]] = []
    for chunk_id, document_id, page_number, tokens in rows:
        chunk_ids.append(str(chunk_id))
        document_ids.append(str(document_id))
        pages.append(int(page_number))
        corpus.append(str(tokens or "").split())
    payload = {
        "chunk_id": chunk_ids,
        "document_id": document_ids,
        "page_number": pages,
        "bm25": BM25Okapi(corpus) if corpus else None,
    }
    return payload


def search(payload: dict, question: str, k: int = 5) -> list[dict]:
    """Highest BM25 scores for a question. The question uses the BM25 tokenizer."""
    model = payload.get("bm25")
    if model is None or not payload["chunk_id"]:
        return []
    scores = model.get_scores(bm25_tokens(question))
    order = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)
    hits = []
    for index in order[:k]:
        if float(scores[index]) <= 0:
            continue
        hits.append(
            {
                "score": float(scores[index]),
                "chunk_id": payload["chunk_id"][index],
                "document_id": payload["document_id"][index],
                "page_number": payload["page_number"][index],
            }
        )
    return hits


def write_family(connection, s3, family: str) -> int:
    key = bm25_key(family)
    if s3 is not None and finished(s3, key):
        print(f"skip {family}")
        return 0
    payload = build_payload(load_token_rows(connection, family))
    handle, path = tempfile.mkstemp(suffix=".pkl")
    os.close(handle)
    try:
        with open(path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if s3 is None:
            local_out = Path(os.environ["BM25_LOCAL"])
            local_out.mkdir(parents=True, exist_ok=True)
            target = local_out / f"asset_family={family}.pkl"
            target.write_bytes(Path(path).read_bytes())
        else:
            s3.upload_file(path, BUCKET, key)
    finally:
        os.remove(path)
    print(f"stored {family} passages={len(payload['chunk_id'])}")
    return len(payload["chunk_id"])


def main() -> int:
    local = os.environ.get("BM25_LOCAL", "").strip()
    if local:
        import duckdb

        connection = duckdb.connect()
        s3 = None
        raw = os.environ.get("INDEX_FAMILY", "").strip()
        families = [raw] if raw else []
    else:
        import boto3

        s3 = boto3.client("s3", region_name=REGION)
        families = selected_families(s3)
        connection = connect()
    if not families:
        return 1
    stored = 0
    passages = 0
    for family in families:
        count = write_family(connection, s3, family)
        if count:
            stored += 1
            passages += count
    line = f"families={len(families)} stored={stored} passages={passages}\n"
    Path(STATUS_NAME).write_text(line, encoding="utf-8")
    sys.stdout.write(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
