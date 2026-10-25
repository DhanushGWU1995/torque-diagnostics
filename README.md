# torque-diagnostics

Technician assistant for ABB drive manuals. Answers from indexed pages and withholds a procedure the manual does not support.

## Install

`pyproject.toml` lists the Python libraries, the same role as `package.json`. `uv.lock` pins the exact versions. From the repository root:

```bash
uv sync
```

That is the only install step. It creates `.venv` and installs every library named in `pyproject.toml`: boto3, DuckDB, Polars, PyArrow, PyMuPDF, NumPy, rank-bm25, sentence-transformers, PyTorch, Transformers, and Accelerate.

## Documentation

The write-ups live in `docs/`. This file stays at the repository root.

- [docs/full-pipeline.md](docs/full-pipeline.md) is the run from a new EC2 through the PDF download, chunking, every index, the answer, and validation.
- [docs/milestone-1-data-and-questions.docx](docs/milestone-1-data-and-questions.docx) is the project proposal, with the exploratory charts and the questions.
