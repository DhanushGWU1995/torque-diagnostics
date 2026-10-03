#!/usr/bin/env python3
"""Download the ABB raw PDFs and store them in the course bucket.

Credentials come from the environment or the instance role. This file does
not contain an access key. From a clone of the repo, at the repository root:

    uv sync
    uv run python data-ingestion/scripts/upload_library_pdfs.py

On a machine that already has the packages installed, `python3 upload_library_pdfs.py`
from this directory is enough.

The CSV ABB_100GB_Download_List.csv is read from the current directory, or
from the same directory as this script. A PDF already stored at its key is
skipped. Each new file lands at:

    s3://dats-nlp/dhanush.mathivanan@gwu.edu/torque-diagnostics/raw/<family>/<document-kind>/<document_id>.pdf

Family comes from asset_layout.py. An ACS880 frame or a firmware manual
keeps its own folder. A technical file whose title names another series,
such as COM600 or REC615, uses that series. Catalogues, articles, and
brochures stay in library, as do technical files with no series in the
title. A _COMPLETE marker is written in each folder when every row has a
PDF in S3.
"""

from __future__ import annotations

import csv
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import BinaryIO, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import boto3
from botocore.exceptions import BotoCoreError, ClientError


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
from asset_layout import RAW_ROOT, asset_family as family_for, document_kind_folder, raw_key

BUCKET = "dats-nlp"
REGION = "us-east-1"
CSV_NAME = "ABB_100GB_Download_List.csv"
WORKERS = 4
TIMEOUT = 180
ATTEMPTS = 4
MARKER = "_COMPLETE"
PDF_MAGIC = b"%PDF-"
USER_AGENT = "TorqueDiagnostics/1.0"

log = logging.getLogger("library-upload")


class HeadThenBody:
    """S3 upload stream: the checked PDF header, then the rest of the response."""

    def __init__(self, head: bytes, body: BinaryIO) -> None:
        self._head = head
        self._used = 0
        self._body = body

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            rest = self._head[self._used :]
            self._used = len(self._head)
            return rest + self._body.read()
        chunk = b""
        if self._used < len(self._head):
            take = min(size, len(self._head) - self._used)
            chunk = self._head[self._used : self._used + take]
            self._used += take
            size -= take
        if size:
            chunk += self._body.read(size)
        return chunk

    def close(self) -> None:
        self._body.close()


def csv_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    beside_script = Path(__file__).resolve().parent / CSV_NAME
    in_cwd = Path.cwd() / CSV_NAME
    if in_cwd.is_file():
        return in_cwd
    return beside_script


def object_key(row: dict[str, str]) -> str:
    family = family_for(row.get("title", ""), row.get("document_id", ""), row.get("document_kind", ""))
    kind = document_kind_folder(row.get("document_kind", ""))
    return raw_key(family, kind, row["document_id"])


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            if (row.get("file_suffix") or "").lower() not in {"", "pdf"}:
                continue
            if row.get("document_id") and row.get("download_url"):
                rows.append(row)
    return rows


def existing_keys(s3) -> set[str]:
    found: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{RAW_ROOT}/"):
        for item in page.get("Contents", []):
            found.add(item["Key"])
    return found


def download_urls(url: str) -> list[str]:
    url = url.replace(" ", "%20")
    urls = [url]
    if "DocumentPartId=1" in url:
        alternate = url.replace("DocumentPartId=1", "DocumentPartId=")
        if alternate not in urls:
            urls.append(alternate)
    return urls


def fetch(url: str) -> HeadThenBody:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    response = urlopen(request, timeout=TIMEOUT)
    head = response.read(len(PDF_MAGIC))
    if head != PDF_MAGIC:
        response.close()
        raise ValueError("response is not a PDF")
    return HeadThenBody(head, response)


def store_one(s3, key: str, url: str, seen: set[str], lock: threading.Lock) -> str:
    with lock:
        if key in seen:
            return "skipped"
    last_error = "upload failed"
    for candidate in download_urls(url):
        for attempt in range(1, ATTEMPTS + 1):
            body: HeadThenBody | None = None
            try:
                body = fetch(candidate)
                s3.upload_fileobj(
                    body,
                    BUCKET,
                    key,
                    ExtraArgs={"ContentType": "application/pdf"},
                )
                with lock:
                    seen.add(key)
                return "stored"
            except ValueError as exc:
                last_error = str(exc)
                break
            except HTTPError as exc:
                last_error = str(exc)
                if exc.code == 404:
                    break
                if exc.code in {400, 401, 403}:
                    raise RuntimeError(last_error) from exc
                time.sleep(min(2 ** attempt, 30))
            except (URLError, TimeoutError, BotoCoreError, ClientError) as exc:
                last_error = str(exc)
                time.sleep(min(2 ** attempt, 30))
            finally:
                if body is not None:
                    body.close()
    raise RuntimeError(last_error)


def write_markers(s3, keys: Iterable[str]) -> int:
    groups: dict[str, list[str]] = {}
    for key in keys:
        groups.setdefault(key.rsplit("/", 1)[0], []).append(key)
    for folder, folder_keys in groups.items():
        s3.put_object(
            Bucket=BUCKET,
            Key=f"{folder}/{MARKER}",
            Body="\n".join(sorted(folder_keys)).encode("utf-8"),
            ContentType="text/plain",
        )
    return len(groups)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    path = csv_path()
    if not path.is_file():
        log.error("CSV not found at %s", path)
        return 1
    rows = load_rows(path)
    if not rows:
        log.error("No PDF rows in %s", path)
        return 1

    s3 = boto3.client("s3", region_name=REGION)
    try:
        seen = existing_keys(s3)
    except ClientError as exc:
        log.error("Cannot list s3://%s/%s/: %s", BUCKET, RAW_ROOT, exc)
        return 1

    lock = threading.Lock()
    failed: list[str] = []
    stored = skipped = 0
    jobs = [(object_key(row), row) for row in rows]
    log.info("bucket=%s rows=%s already_in_s3=%s", BUCKET, len(jobs), len(seen))

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(store_one, s3, key, row["download_url"], seen, lock): key
            for key, row in jobs
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            key = futures[future]
            try:
                status = future.result()
            except Exception as exc:
                failed.append(f"{key}\t{exc}")
                log.warning("failed %s (%s/%s) %s", key, done, len(jobs), exc)
                continue
            stored += status == "stored"
            skipped += status == "skipped"
            if done % 25 == 0 or done == len(jobs):
                log.info(
                    "progress %s/%s stored=%s skipped=%s failed=%s",
                    done, len(jobs), stored, skipped, len(failed),
                )

    if failed:
        report = Path("library_upload_failed.txt")
        report.write_text("\n".join(failed) + "\n", encoding="utf-8")
        log.error("%s failed. Marker not written. See %s", len(failed), report)
        return 1

    folders = write_markers(s3, (key for key, _row in jobs))
    log.info("done stored=%s skipped=%s folders=%s", stored, skipped, folders)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
