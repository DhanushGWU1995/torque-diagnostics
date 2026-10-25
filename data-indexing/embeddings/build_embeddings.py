#!/usr/bin/env python3
"""Build one embedding index per equipment folder.

This reads passage Parquet under chunks/asset_family=<family>/. It does
not open the PDFs and it does not change the BM25 index. The vectors come
from one English sentence-transformer, BAAI/bge-base-en-v1.5. A row is
embedded as stored. A question is embedded with the model's retrieval
prefix so the two vectors can be compared by cosine similarity. A folder
that already has a finished embedding file is skipped.

Set INDEX_FAMILY to one folder for a test. INDEX_FAMILY=all writes every
folder. A folder that already has a finished index is skipped. library is
large, so run it after the smaller folders have stored.

Credentials come from the environment. This file does not contain an
access key.

    INDEX_FAMILY=ACS880-11 uv run python data-indexing/embeddings/build_embeddings.py

On the EC2 instance, from ~/dl-venv/nlp_project:

    INDEX_FAMILY=ACS880-11 ~/dl-venv/bin/python data-indexing/embeddings/build_embeddings.py

Each folder is written to

    s3://dats-nlp/dhanush.mathivanan@gwu.edu/torque-diagnostics/index/embeddings/asset_family=<family>.parquet
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "bm25"))
sys.path.insert(0, str(_HERE.parents[2]))

from build_bm25 import BUCKET, REGION, connect, selected_families  # noqa: E402
from chunk_sources import passage_sql_for, passages_present, rebuild_requested  # noqa: E402

EMBED_ROOT = "g34305547-dats6450-s2026/torque-diagnostics/index/embeddings"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBED_DIM = 768
STATUS_NAME = "embeddings_status.txt"
BATCH = 32


def embed_key(family: str) -> str:
    return f"{EMBED_ROOT}/asset_family={family}.parquet"


def finished(s3, key: str) -> bool:
    try:
        response = s3.get_object(Bucket=BUCKET, Key=key, Range="bytes=0-3")
    except Exception:
        return False
    return response["Body"].read() == b"PAR1"


def load_passages(connection, family: str) -> list[tuple]:
    if not passages_present(family):
        return []
    return connection.execute(
        f"""
        SELECT chunk_id, document_id, page_number, text
        FROM ({passage_sql_for(family)}) sources
        WHERE text IS NOT NULL AND trim(text) <> ''
        ORDER BY chunk_id
        """
    ).fetchall()


def load_encoder():
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model={MODEL_NAME} device={device}", flush=True)
    return SentenceTransformer(MODEL_NAME, device=device)


def encode_texts(encoder, texts: list[str], query: bool) -> np.ndarray:
    prepared = [QUERY_PREFIX + text if query else text for text in texts]
    vectors = encoder.encode(
        prepared,
        batch_size=BATCH,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def search_vectors(matrix: np.ndarray, rows: list[dict], query_vector: np.ndarray, k: int) -> list[dict]:
    """Highest cosine scores. The matrix and the query vector are both unit length."""
    if matrix.size == 0:
        return []
    scores = matrix @ query_vector.astype(np.float32)
    order = np.argsort(-scores)[:k]
    hits = []
    for index in order:
        score = float(scores[index])
        if score <= 0:
            continue
        row = dict(rows[int(index)])
        row["score"] = score
        hits.append(row)
    return hits


def embed_passages(encoder, passages: list[tuple]) -> tuple[np.ndarray, list[dict]]:
    rows: list[dict] = []
    blocks: list[np.ndarray] = []
    for start in range(0, len(passages), BATCH):
        batch = passages[start : start + BATCH]
        texts = [str(text) for _, _, _, text in batch]
        blocks.append(encode_texts(encoder, texts, query=False))
        for chunk_id, document_id, page_number, _text in batch:
            rows.append(
                {
                    "chunk_id": str(chunk_id),
                    "document_id": str(document_id),
                    "page_number": int(page_number),
                }
            )
        print(f"encoded {min(start + BATCH, len(passages))}/{len(passages)}", flush=True)
    if not blocks:
        return np.zeros((0, EMBED_DIM), dtype=np.float32), rows
    return np.vstack(blocks), rows


def write_table(path: str, matrix: np.ndarray, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    embedding = pa.FixedSizeListArray.from_arrays(
        pa.array(matrix.reshape(-1), type=pa.float32()),
        EMBED_DIM,
    )
    table = pa.table(
        {
            "chunk_id": [row["chunk_id"] for row in rows],
            "document_id": [row["document_id"] for row in rows],
            "page_number": pa.array([row["page_number"] for row in rows], type=pa.int32()),
            "embedding": embedding,
        }
    )
    pq.write_table(table, path, compression="zstd")


def read_table(path: str) -> tuple[np.ndarray, list[dict]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["chunk_id", "document_id", "page_number", "embedding"])
    rows = [
        {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "page_number": int(page_number),
        }
        for chunk_id, document_id, page_number in zip(
            table.column("chunk_id").to_pylist(),
            table.column("document_id").to_pylist(),
            table.column("page_number").to_pylist(),
        )
    ]
    values = table.column("embedding").combine_chunks().values.to_numpy()
    matrix = np.asarray(values, dtype=np.float32).reshape(len(rows), EMBED_DIM)
    return matrix, rows


def write_family(connection, s3, encoder, family: str) -> int:
    key = embed_key(family)
    if s3 is not None and finished(s3, key) and not rebuild_requested():
        print(f"skip {family}", flush=True)
        return 0
    passages = load_passages(connection, family)
    matrix, rows = embed_passages(encoder, passages)
    handle, path = tempfile.mkstemp(suffix=".parquet")
    os.close(handle)
    try:
        write_table(path, matrix, rows)
        if s3 is None:
            local_out = Path(os.environ["EMBED_LOCAL"])
            local_out.mkdir(parents=True, exist_ok=True)
            target = local_out / f"asset_family={family}.parquet"
            target.write_bytes(Path(path).read_bytes())
        else:
            s3.upload_file(path, BUCKET, key)
    finally:
        os.remove(path)
    print(f"stored {family} passages={len(rows)}", flush=True)
    return len(rows)


def main() -> int:
    local = os.environ.get("EMBED_LOCAL", "").strip()
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
    encoder = load_encoder()
    stored = 0
    passages = 0
    for family in families:
        count = write_family(connection, s3, encoder, family)
        if count:
            stored += 1
            passages += count
    line = f"families={len(families)} stored={stored} passages={passages}\n"
    Path(STATUS_NAME).write_text(line, encoding="utf-8")
    sys.stdout.write(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
