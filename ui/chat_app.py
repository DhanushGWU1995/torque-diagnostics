#!/usr/bin/env python3
"""Torque Diagnostics chat UI (Streamlit).

Run on the EC2 that already holds the indexes and GPU models:

    cd ~/dl-venv/nlp_project   # or the repo root
    source ~/dl-venv/bin/activate
    streamlit run ui/chat_app.py --server.address 127.0.0.1 --server.port 8501

From your laptop (VPN), forward the port:

    ssh -i .../AWS_NLP_Config_keys.pem -L 8501:127.0.0.1:8501 ubuntu@EC2_IP

Then open http://127.0.0.1:8501
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reply"))

import streamlit as st

from chat_service import confidence_matrix, run_sample, run_turn, sample_questions

st.set_page_config(page_title="Torque Diagnostics", layout="wide")
st.title("Torque Diagnostics")
st.caption(
    "ABB drive manuals · hybrid BM25 + embedding retrieval · Qwen reply · DeBERTa confidence gate"
)


@st.cache_resource(show_spinner="Loading sample questions…")
def _samples() -> list[dict]:
    return sample_questions()


samples = _samples()
labels = {
    item["id"]: (
        f"{item['id']}: {item['question']}"
        + ("  [expect withhold]" if item.get("withhold") else "")
    )
    for item in samples
}

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Ask")
    pick = st.selectbox(
        "Sample questions",
        options=["(type your own)"] + list(labels.keys()),
        format_func=lambda key: labels.get(key, key),
    )
    default_q = ""
    selected = None
    if pick != "(type your own)":
        selected = next(item for item in samples if item["id"] == pick)
        default_q = selected["question"]

    question = st.text_area("Question", value=default_q, height=100)
    run_clicked = st.button("Answer", type="primary")
    run_all = st.button("Score all sample questions")

with right:
    st.subheader("What this panel shows")
    st.markdown(
        """
- **Answer** with document id and page citations  
- **Confidence**: `copied-sentence`, `deberta`, or `withheld`  
- **DeBERTa**: whether the gate called it, call count, keep count  
- **Top 10** hybrid retrieval passages  
- **Citation metrics** when the question is one of the hand-checked samples  
  (precision, recall, F1, MRR, citation accuracy = precision)
"""
    )

if "history" not in st.session_state:
    st.session_state.history = []
if "batch" not in st.session_state:
    st.session_state.batch = []


def _render_result(result: dict) -> None:
    st.markdown("### Answer")
    st.write(result["answer"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Confidence", result["confidence"])
    c2.metric("Withheld", "yes" if result["withheld"] else "no")
    c3.metric("DeBERTa triggered", "yes" if result["deberta_triggered"] else "no")
    c4.metric("DeBERTa calls", result["deberta_calls"])
    c5.metric("DeBERTa keeps", result["deberta_keeps"])

    st.caption(f"Folders: {', '.join(result['folders'])}")

    if result.get("metrics") is not None:
        m = result["metrics"]
        st.markdown("### Citation metrics (hand-checked pages)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Precision", f"{m['precision']:.3f}")
        m2.metric("Recall", f"{m['recall']:.3f}")
        m3.metric("F1", f"{m['f1']:.3f}")
        m4.metric("MRR", f"{m['mrr']:.3f}")
        m5.metric("Citation accuracy", f"{m['citation_accuracy']:.3f}")
        if result.get("gold"):
            gold = ", ".join(
                f"{page['document_id']} p.{page['page_number']}" for page in result["gold"]
            )
            st.caption(f"Gold pages: {gold}")
        elif result.get("expect_withhold"):
            st.caption("Gold: answer should be withheld")

    st.markdown("### Top 10 retrieval passages")
    for row in result["passages"]:
        with st.expander(
            f"#{row['rank']}  score={row['score']:.4f}  "
            f"{row['document_id']} p.{row['page_number']}  [{row['asset_family']}]"
        ):
            st.write(row["snippet"] or row["text"] or "(no text loaded)")


if run_clicked and question.strip():
    with st.spinner("Retrieving, writing, and checking…"):
        if selected is not None and question.strip() == selected["question"].strip():
            result = run_sample(selected)
        else:
            result = run_turn(question)
    st.session_state.history.insert(0, result)
    _render_result(result)
elif st.session_state.history and not run_all:
    _render_result(st.session_state.history[0])

if run_all:
    batch = []
    progress = st.progress(0.0, text="Scoring samples…")
    for index, item in enumerate(samples):
        progress.progress((index) / max(len(samples), 1), text=f"Scoring {item['id']}…")
        batch.append(run_sample(item))
    progress.progress(1.0, text="Done")
    st.session_state.batch = batch

if st.session_state.batch:
    st.markdown("---")
    st.markdown("### Sample-set confidence matrix")
    counts = confidence_matrix(st.session_state.batch)
    matrix_cols = st.columns(3)
    matrix_cols[0].metric("copied-sentence", counts["copied-sentence"])
    matrix_cols[1].metric("deberta", counts["deberta"])
    matrix_cols[2].metric("withheld", counts["withheld"])

    rows = []
    for result, item in zip(st.session_state.batch, samples):
        metrics = result.get("metrics") or {}
        rows.append(
            {
                "id": item["id"],
                "confidence": result["confidence"],
                "deberta_calls": result["deberta_calls"],
                "withheld": result["withheld"],
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "citation_accuracy": metrics.get("citation_accuracy"),
                "mrr": metrics.get("mrr"),
            }
        )
    st.dataframe(rows, use_container_width=True)
    for result in st.session_state.batch:
        with st.expander(f"{result['confidence']} · {result['question'][:80]}"):
            _render_result(result)
