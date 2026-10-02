# torque-diagnostics
Technician assistant for ABB drive manuals. Answers from indexed pages and withholds a procedure the manual does not support.

## Python

Install [uv](https://docs.astral.sh/uv/), clone the repo, and from the repository root run:

```bash
uv sync
```

That creates `.venv` and installs the libraries for `data-ingestion/scripts/upload_library_pdfs.py` and `data-preprocessing/chunk_raw_pdfs.py`. The chunking script also needs the Tesseract program (`brew install tesseract` on macOS, or `sudo apt-get install -y tesseract-ocr` on Ubuntu).

```bash
uv run python data-ingestion/scripts/upload_library_pdfs.py
uv run python data-preprocessing/chunk_raw_pdfs.py
```
