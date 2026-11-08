# Streamlit UI evidence (Q8–Q11)

Screenshots belong here after you capture them on the GPU EC2 UI. Scripts in the parent folder already answer the questions with CSVs/markdown; screenshots are the **technician-facing** proof for the report/PPT.

## Start UI

```bash
cd ~/dl-venv/nlp_project
set -a; . ~/.aws_session.env; set +a
streamlit run ui/chat_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

SSH tunnel from the laptop, open `http://127.0.0.1:8501`.

## Captures to save in this folder

| File name (suggested) | Sample / action | Proves |
|---|---|---|
| `q8_flange_tools_answer.png` | Sample `flange-tools` → Answer | Q8 cited page when sentence is on the manual |
| `q8_unrelated_stock_withhold.png` | Sample `unrelated-stock` → Answer | Q8/Q9 domain-restricted withhold |
| `q11_ov_controller_deberta.png` | Sample `acs860-ov-controller-protect` → Answer | Q11 DeBERTa path (calls/keeps & `deberta` confidence) |
| `q10_confidence_metrics.png` | Same as above, metrics row | Q10 confidence label + citation metrics |
| `q6_top10_passages.png` | Any answered sample, Top 10 panel | Retrieval context shown in UI (hybrid list) |

Commit the PNGs once captured (`git add nlp-output/ui-evidence/*.png`).
