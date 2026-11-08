#!/usr/bin/env python3
"""One chat turn: hybrid retrieve, write, gate, and optional citation scores.

Used by the Streamlit UI. Keeps the same path as reply/evaluate_answers.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reply"))

from answer_question import load_texts, retrieve  # noqa: E402
from evaluate_answers import (  # noqa: E402
    DEFAULT_QUESTIONS,
    answer_one,
    cited_pages,
    load_questions,
    prediction_scores,
)
from manual_tokens import route_families  # noqa: E402

SHOW_PASSAGES = 10
RETRIEVE_K = 15


def sample_questions(path: Path | None = None) -> list[dict]:
    return load_questions(path or DEFAULT_QUESTIONS)


def _passage_rows(hits: list[dict], passages: list[dict], limit: int) -> list[dict]:
    text_by_id = {row["chunk_id"]: row for row in passages}
    rows: list[dict] = []
    for rank, hit in enumerate(hits[:limit], start=1):
        chunk_id = str(hit.get("chunk_id", ""))
        body = text_by_id.get(chunk_id, {})
        text = str(body.get("text") or hit.get("text") or "")
        rows.append(
            {
                "rank": rank,
                "score": float(hit.get("score") or 0.0),
                "document_id": str(body.get("document_id") or hit.get("document_id") or ""),
                "page_number": int(body.get("page_number") or hit.get("page_number") or 0),
                "asset_family": str(hit.get("asset_family") or ""),
                "text": text,
                "snippet": text[:400] + ("…" if len(text) > 400 else ""),
            }
        )
    return rows


def run_turn(
    question: str,
    *,
    gold: list[dict] | None = None,
    expect_withhold: bool = False,
    retrieve_k: int = RETRIEVE_K,
    show_passages: int = SHOW_PASSAGES,
) -> dict:
    """Answer one technician question and return UI fields."""
    question = question.strip()
    if not question:
        raise ValueError("question is empty")

    families = route_families(question)
    hits = retrieve(question, families, retrieve_k)
    passages = load_texts(hits)
    released, confidence, deberta_calls, deberta_keeps = answer_one(question, hits)
    withheld = confidence == "withheld"
    cited = cited_pages(released)

    metrics = None
    if gold is not None or expect_withhold:
        scores = prediction_scores(cited, gold or [], withheld, expect_withhold)
        # Citation accuracy: share of cited pages that match the hand-checked set.
        metrics = {
            "precision": float(scores["precision"]),
            "recall": float(scores["recall"]),
            "f1": float(scores["f1"]),
            "mrr": float(scores["mrr"]),
            "rank": int(scores["rank"]),
            "citation_accuracy": float(scores["precision"]),
        }

    return {
        "question": question,
        "folders": families,
        "answer": released,
        "confidence": confidence,
        "withheld": withheld,
        "deberta_triggered": deberta_calls > 0,
        "deberta_calls": int(deberta_calls),
        "deberta_keeps": int(deberta_keeps),
        "cited_pages": [{"document_id": doc, "page_number": page} for doc, page in cited],
        "passages": _passage_rows(hits, passages, show_passages),
        "metrics": metrics,
        "gold": gold or [],
        "expect_withhold": expect_withhold,
    }


def run_sample(item: dict, **kwargs) -> dict:
    return run_turn(
        item["question"],
        gold=item.get("relevant") or [],
        expect_withhold=bool(item.get("withhold")),
        **kwargs,
    )


def confidence_matrix(rows: list[dict]) -> dict[str, int]:
    counts = {"copied-sentence": 0, "deberta": 0, "withheld": 0}
    for row in rows:
        label = row.get("confidence", "")
        if label in counts:
            counts[label] += 1
    return counts
