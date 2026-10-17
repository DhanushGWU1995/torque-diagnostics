#!/usr/bin/env python3
"""Write image text from the raw PDFs with Gemini.

The passage Parquet files stay as they are. This script sends a cropped
figure to Gemini and stores the words Gemini reads. It does not use
Tesseract. From the repository root, with a Gemini API key in the
environment:

    export GEMINI_API_KEY='...'
    INDEX_FAMILY=ACS880-11 MAX_FILES=5 uv run python data-preprocessing/images/extract_image_text.py

Each PDF under raw/<family>/<kind>/<document_id>.pdf is written to

    s3://dats-nlp/dhanush.mathivanan@gwu.edu/torque-diagnostics/images/asset_family=<family>/<document_id>.parquet

A figure smaller than MIN_IMAGE_PX (default 120) is skipped. A figure that
is repeated on more than two pages is skipped, because that is usually a
logo. A document with no remaining figure still gets an empty Parquet
file, so the next run skips it. Leave MAX_FILES unset only when you mean
to send every remaining PDF to Gemini. INDEX_FAMILY limits the run to one
equipment folder. GEMINI_MODEL overrides gemini-3.8-flash. This does not
rebuild the token files, the BM25 index, or the embedding index.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from raw_pdfs import BUCKET, REGION, discover, file_limit, write_parquet

FOLDER = "images"
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"
MIN_IMAGE_PX = int(os.environ.get("MIN_IMAGE_PX", "120") or "120")
STATUS_NAME = "image_status.txt"
REMAINING_NAME = "image_remaining.txt"
FAILED_NAME = "image_failed.txt"
PROMPT = (
    "Transcribe every word, label, and table cell visible in this ABB manual figure. "
    "Keep the reading order. Do not add a step, a value, or a tool that is not visible. "
    "If the figure has no words, reply with an empty string."
)
EMPTY_REPLIES = {
    "",
    "empty",
    "empty string",
    "no text",
    "no words",
    "none",
}

log = logging.getLogger("extract-images")


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY before running the image script.")
    return key


def repeated_xrefs(document) -> set[int]:
    counts: dict[int, int] = {}
    for page in document:
        seen: set[int] = set()
        for item in page.get_images(full=True):
            xref = int(item[0])
            if xref in seen:
                continue
            seen.add(xref)
            counts[xref] = counts.get(xref, 0) + 1
    return {xref for xref, count in counts.items() if count > 2}


def figure_png(page, xref: int) -> bytes | None:
    import pymupdf

    rects = page.get_image_rects(xref)
    if not rects:
        return None
    rect = max(rects, key=lambda item: item.width * item.height)
    if rect.width < MIN_IMAGE_PX or rect.height < MIN_IMAGE_PX:
        return None
    longest = max(rect.width, rect.height)
    zoom = min(2.0, 1600 / longest) if longest else 1.0
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=rect, alpha=False)
    png = pixmap.tobytes("png")
    if len(png) < 2000 or len(png) > 15 * 1024 * 1024:
        return None
    return png


def transcribe(client, png: bytes) -> str:
    from google.genai import types

    delay = 5
    last_error = ""
    for _attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(data=png, mime_type="image/png"),
                    PROMPT,
                ],
                config=types.GenerateContentConfig(temperature=0),
            )
            text = (response.text or "").strip()
            if text.lower() in EMPTY_REPLIES:
                return ""
            return text
        except Exception as exc:
            last_error = str(exc)
            time.sleep(delay)
            delay *= 3
    raise RuntimeError(last_error or "Gemini request failed")


def images_for_pdf(path: str, document_id: str, family: str, client) -> list[dict[str, object]]:
    import pymupdf

    rows: list[dict[str, object]] = []
    document = pymupdf.open(path)
    try:
        skip = repeated_xrefs(document)
        for page_number, page in enumerate(document, start=1):
            kept = 0
            seen: set[int] = set()
            for item in page.get_images(full=True):
                xref = int(item[0])
                if xref in seen or xref in skip:
                    continue
                seen.add(xref)
                png = figure_png(page, xref)
                if png is None:
                    continue
                text = transcribe(client, png)
                if not text:
                    continue
                rows.append(
                    {
                        "image_id": f"{document_id}:{page_number:04d}:{kept:03d}",
                        "document_id": document_id,
                        "asset_family": family,
                        "page_number": page_number,
                        "image_index": kept,
                        "model": MODEL,
                        "text": text,
                    }
                )
                kept += 1
    finally:
        document.close()
    return rows


def process_one(task, client) -> tuple[str, str, int, str]:
    import boto3

    pdf_path = ""
    parquet_path = ""
    try:
        s3 = boto3.client("s3", region_name=REGION)
        pdf_handle, pdf_path = tempfile.mkstemp(suffix=".pdf")
        os.close(pdf_handle)
        s3.download_file(BUCKET, task.pdf_key, pdf_path)
        rows = images_for_pdf(pdf_path, task.document_id, task.family, client)
        parquet_handle, parquet_path = tempfile.mkstemp(suffix=".parquet")
        os.close(parquet_handle)
        import pyarrow as pa

        schema = pa.schema(
            [
                ("image_id", pa.string()),
                ("document_id", pa.string()),
                ("asset_family", pa.string()),
                ("page_number", pa.int32()),
                ("image_index", pa.int32()),
                ("model", pa.string()),
                ("text", pa.string()),
            ]
        )
        write_parquet(rows, parquet_path, schema)
        s3.upload_file(parquet_path, BUCKET, task.parquet_key)
        return task.pdf_key, "stored" if rows else "empty", len(rows), ""
    except Exception:
        return task.pdf_key, "failed", 0, traceback.format_exc()
    finally:
        for path in (pdf_path, parquet_path):
            if path and os.path.exists(path):
                os.remove(path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import boto3
    from google import genai

    try:
        limit = file_limit()
        key = api_key()
    except (ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1
    client = genai.Client(api_key=key)
    s3 = boto3.client("s3", region_name=REGION)
    pending, already = discover(s3, FOLDER)
    batch = pending if limit is None else pending[:limit]
    Path.cwd().joinpath(REMAINING_NAME).write_text(
        "".join(f"{task.pdf_key}\n" for task in pending),
        encoding="utf-8",
    )
    failed_path = Path.cwd() / FAILED_NAME
    status_path = Path.cwd() / STATUS_NAME
    log.info("model=%s image_shards=%s remaining=%s this_run=%s", MODEL, already, len(pending), len(batch))
    stored = empty = failed = 0
    for index, task in enumerate(batch, start=1):
        pdf_key, status, count, error = process_one(task, client)
        if status == "stored":
            stored += 1
        elif status == "empty":
            empty += 1
        else:
            failed += 1
            with failed_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{pdf_key}\n{error}\n")
        log.info("this_run=%s/%s %s images=%s %s", index, len(batch), status, count, pdf_key)
        status_path.write_text(
            f"image_shards={already + stored + empty}\nremaining={len(pending) - index}\n"
            f"stored={stored}\nempty={empty}\nfailed={failed}\nmodel={MODEL}\n",
            encoding="utf-8",
        )
    log.info("done stored=%s empty=%s failed=%s", stored, empty, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
