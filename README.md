# torque-diagnostics

Technician assistant for ABB drive manuals. Answers only from indexed English manual pages, cites `document_id` and `page_number`, and withholds a procedure the pages do not support.

**Text RAG path (current):** PDF download → text passages → coverage / EDA → tokens → BM25 → embeddings → hybrid search → Qwen reply → DeBERTa confidence gate → evaluation + Streamlit chat UI.

Tables and figure/image corpus extraction are **not** in this run. The index is the PDF text layer only.

## Install

`pyproject.toml` lists the Python libraries. `uv.lock` pins exact versions. From the repository root:

```bash
uv sync
```

That installs boto3, DuckDB, Polars, PyArrow, PyMuPDF, NumPy, rank-bm25, sentence-transformers, PyTorch, Transformers, Accelerate, and Streamlit.

On the course EC2 Deep Learning image you can also use the existing venv:

```bash
source ~/dl-venv/bin/activate
pip install 'streamlit>=1.37'   # if Streamlit is not already installed
```

## AWS session

Scripts read temporary keys from the environment. Do not commit keys.

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."
export AWS_DEFAULT_REGION=us-east-1
```

On EC2 you can keep the same exports in `~/.aws_session.env` and `set -a; . ~/.aws_session.env; set +a` before a long job or Streamlit.

## Documentation

- [docs/full-pipeline.md](docs/full-pipeline.md) — full EC2 run: download, chunking, indexes, answer, validation, chat UI.
- [docs/milestone-1-data-and-questions.docx](docs/milestone-1-data-and-questions.docx) — project proposal with EDA charts.

## Reply and evaluation (CLI)

```bash
QUESTION='What tools are needed to flange-mount an ACS880-11?' \
  uv run python reply/answer_question.py

ANSWER=1 uv run python reply/evaluate_answers.py
```

Hand-checked samples live in `reply/questions.jsonl` (four lines today):

| id | Purpose |
|---|---|
| `flange-tools` | Copied-sentence path; gold `3AXD50000349838` p.17 |
| `acs860-reset` | ACS860 autoreset; gold p.68 / p.27 |
| `acs860-ov-controller-protect` | DeBERTa / weak-confidence path; gold p.62 |
| `unrelated-stock` | Expected withhold |

Scores: precision, recall, F1, MRR on **cited** pages. Confidence: `copied-sentence`, `deberta`, or `withheld` (plus `deberta_calls`).

## Chat UI (Streamlit on GPU EC2)

Run the UI on the **same g5 EC2** that already has S3 indexes and GPU models. Do **not** host this on GitHub Pages, free GitHub Actions, or a public Hugging Face Space (no room for Qwen + private ABB indexes + course AWS credentials).

### Start (foreground)

```bash
cd ~/dl-venv/nlp_project          # or the repo root after git clone / uv sync
source ~/dl-venv/bin/activate     # or: source .venv/bin/activate
set -a; . ~/.aws_session.env; set +a   # or export AWS_* in this shell
streamlit run ui/chat_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

### Start (background)

```bash
cd ~/dl-venv/nlp_project && source ~/dl-venv/bin/activate
set -a; . ~/.aws_session.env; set +a
nohup streamlit run ui/chat_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true \
  > /tmp/streamlit.log 2>&1 &
echo $! > /tmp/streamlit.pid
```

### Stop

```bash
pkill -f "streamlit run ui/chat_app.py"
# or: kill "$(cat /tmp/streamlit.pid)"
```

### Open from your laptop (GWU VPN)

```bash
ssh -i /Users/dhanushmathivanan/dhanush_gwu/aws_config_keys/AWS_NLP_Config_keys.pem \
  -L 8501:127.0.0.1:8501 ubuntu@THE_PUBLIC_DNS
```

Then open [http://127.0.0.1:8501](http://127.0.0.1:8501). Keep Streamlit on `127.0.0.1` so the course security group does not need a public UI port.

### What the UI shows

- Sample questions from `reply/questions.jsonl`, or your own question
- Live stage progress on each **Answer** (model reuse vs disk→GPU, retrieve, Qwen write, DeBERTa, gate, metrics)
- Answer with citations; confidence; DeBERTa triggered / calls / keeps; withheld yes/no
- Top 10 hybrid retrieval passages
- Citation metrics for hand-checked samples (precision, recall, F1, MRR, citation accuracy = precision)
- **Score all sample questions** — confidence matrix across the sample set

Shared turn logic: `reply/chat_service.py`.

## Model cache (reuse, not re-download)

| Layer | Behavior |
|---|---|
| Disk | Weights live in `models/huggingface`. First use may download once; later EC2 stop/start **reuses disk** (no Hugging Face re-download). |
| GPU | While Streamlit stays up, BGE + Qwen stay in memory (`@st.cache_resource`). DeBERTa loads on the first partial paraphrase, then stays loaded. |
| After Streamlit stop | GPU RAM is freed. Next start loads from disk once. |

Do **not** delete `models/huggingface` unless you intend to re-download. Optional cleanup of a duplicate default cache only: `rm -rf ~/.cache/huggingface` (keep the project `models/huggingface` folder).

## Layout (main pieces)

```text
data-ingestion/          PDF upload to S3
data-preprocessing/      chunking, coverage
data-indexing/           tokens, BM25, embeddings, hybrid query
eda-output/              five EDA questions + charts
reply/                   answer, evaluate, chat_service, questions.jsonl
ui/chat_app.py           Streamlit chat UI
docs/full-pipeline.md    end-to-end EC2 runbook
```
