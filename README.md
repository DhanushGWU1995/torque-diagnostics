# torque-diagnostics
Technician assistant for ABB drive manuals. Answers from indexed pages and withholds a procedure the manual does not support.

## Python

Install [uv](https://docs.astral.sh/uv/), clone the repo, and from the repository root run:

```bash
uv sync
```

That creates `.venv` and installs boto3, Polars, PyArrow, PyMuPDF, Pillow, and pytesseract for both scripts.

### After uv sync

`uv sync` does not install the Tesseract program. The chunking script needs it for pages whose text layer is nearly empty. Install it once on the machine:

```bash
brew install tesseract          # macOS
sudo apt-get install -y tesseract-ocr   # Ubuntu
```

Export a current AWS session in the same shell. The scripts do not contain access keys.

```bash
export AWS_ACCESS_KEY_ID='...'
export AWS_SECRET_ACCESS_KEY='...'
export AWS_SESSION_TOKEN='...'
export AWS_DEFAULT_REGION=us-east-1
```

Then run from the repository root:

```bash
uv run python data-ingestion/scripts/upload_library_pdfs.py
uv run python data-preprocessing/chunk_raw_pdfs.py
```

The upload script reads `ABB_100GB_Download_List.csv` and stores PDFs under `s3://dats-nlp/dhanush.mathivanan@gwu.edu/torque-diagnostics/raw/`. A PDF already at its key is skipped.

The chunking script reads that raw prefix and writes Parquet under `chunks/asset_family=<family>/`. A shard is skipped only when its Parquet footer is intact. Set `MAX_FILES` to convert one batch and then stop. `MAX_FILES=100` is a safe batch on the project EC2. Leave it unset to process every remaining PDF.

```bash
MAX_FILES=100 uv run python data-preprocessing/chunk_raw_pdfs.py
```

Progress is written to `chunk_status.txt` in the working directory. Keys that still need a shard are listed in `chunk_remaining.txt`. Failures are appended to `chunk_failed.txt`. Run only one copy of the chunking script at a time. Files of 200 MB or more run alone. Smaller files run two at a time.
