# NLP output (proposal questions 6–11)

Artifacts that answer the **Natural language processing** and **Statistics / ML** questions from the Milestone 2 proposal. Exploratory Q1–Q5 stay under `eda-output/`.

| Proposal Q | What it asks | How we answer it | Artifact |
|---|---|---|---|
| **6** | BM25 vs embedding vs hybrid pages for flange + ACS860 reset | Script over family indexes | `q6_retrieval_compare.csv`, `q6_summary.md` |
| **7** | Fault / part / procedure ids in one family vs many | Script over passage Parquet | `q7_identifier_families.csv`, `q7_summary.csv`, charts |
| **8** | Cite when page states claim; drop when not | Gate demo + sample eval path | `q8_gate_cite_vs_drop.csv`, `q8_summary.md` |
| **9** | Withhold: wrong family vs unsupported procedure | Heuristic reason labels on sample asks | `q9_withhold_reasons.csv`, `q9_summary.md` |
| **10** | How confidence is derived (hybrid + gate) | Written method + UI evidence | `q10_q11_confidence_method.md`, `ui-evidence/` |
| **11** | Copied-sentence vs DeBERTa entailment | Same method doc + gate demo rows | (same as Q10/Q8) |

## What belongs in a script vs Streamlit screenshot

- **Scripts (this folder):** Q6 ranking tables, Q7 identifier census, Q8/Q9 deterministic gate tables, Q10/Q11 method write-up. These are reproducible from indexes / passages / gate code.
- **Streamlit screenshots (`ui-evidence/`):** Best for Q8–Q11 *as the technician sees them* — confidence label, DeBERTa counts, withheld domain refusal, top-10 passages. Capture on the GPU EC2 UI after syncing this repo (see `ui-evidence/CAPTURE.md`).

## Run on EC2 (GPU + AWS session)

```bash
cd ~/dl-venv/nlp_project   # or repo root
set -a; . ~/.aws_session.env; set +a

# All NLP questions that need S3 / GPU pieces
uv run python nlp-output/answer_nlp_questions.py

# Or one at a time
uv run python nlp-output/q6_retrieval_compare.py
uv run python nlp-output/q7_identifier_families.py
uv run python nlp-output/q8_q11_confidence_gate.py          # local demos always; LIVE=1 for Qwen path
LIVE=1 uv run python nlp-output/q8_q11_confidence_gate.py
uv run python nlp-output/q9_withhold_reason.py               # uses retrieve when AWS available
```

Q8 local demos and the Q10/Q11 method markdown do **not** need the GPU. Q6 and live Q8/Q9 need indexes on S3 (and BGE on GPU for embeddings). Q7 needs S3 passage Parquet (DuckDB/PyArrow; no GPU).
