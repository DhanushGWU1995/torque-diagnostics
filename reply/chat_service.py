#!/usr/bin/env python3
"""One chat turn: hybrid retrieve, write, gate, optional citation scores.

Used by the Streamlit UI. Same answer path as reply/evaluate_answers.py
so the UI and the CLI eval stay in sync.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reply"))
sys.path.insert(0, str(ROOT / "data-indexing" / "embeddings"))
sys.path.insert(0, str(ROOT / "data-indexing" / "tokenization"))

import answer_question as aq  # noqa: E402
import build_embeddings as be  # noqa: E402
from answer_question import (  # noqa: E402
    draft,
    entailment_checker,
    gate,
    load_texts,
    load_writer,
    retrieve,
)
from build_embeddings import load_encoder  # noqa: E402
from evaluate_answers import (  # noqa: E402
    DEFAULT_QUESTIONS,
    WITHHOLD,
    cited_pages,
    confidence_label,
    load_questions,
    prediction_scores,
)
from manual_tokens import route_families  # noqa: E402

SHOW_PASSAGES = 10
RETRIEVE_K = 15

ProgressFn = Callable[[str, str], None]


def _note(on_progress: ProgressFn | None, stage: str, detail: str) -> None:
    if on_progress is not None:
        on_progress(stage, detail)


def warm_models(on_progress: ProgressFn | None = None) -> dict[str, str]:
    """Load BGE + Qwen once in this process. DeBERTa loads on first paraphrase."""
    if be._encoder is None:
        _note(
            on_progress,
            "model load",
            "loading BGE (BAAI/bge-base-en-v1.5) from disk cache to GPU...",
        )
    else:
        _note(on_progress, "model load", "BGE already in GPU, reusing")
    load_encoder()
    _note(on_progress, "model load", "BGE ready")

    if aq._writer is None:
        _note(
            on_progress,
            "model load",
            "loading Qwen2.5-7B-Instruct from disk cache to GPU (slow first time)...",
        )
    else:
        _note(on_progress, "model load", "Qwen already in GPU, reusing")
    load_writer()
    _note(on_progress, "model load", "Qwen ready")

    entailment_checker()
    if aq._entails is None:
        _note(
            on_progress,
            "model load",
            "DeBERTa not loaded yet — loads from disk on first partial paraphrase",
        )
    else:
        _note(on_progress, "model load", "DeBERTa already in GPU, reusing")
    return {
        "encoder": "ready",
        "writer": "ready",
        "deberta": "ready" if aq._entails is not None else "lazy",
    }


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
    force_family: str | None = None,
    retrieve_k: int = RETRIEVE_K,
    show_passages: int = SHOW_PASSAGES,
    on_progress: ProgressFn | None = None,
) -> dict:
    """Run one question and pack the fields the UI expects."""
    question = question.strip()
    if not question:
        raise ValueError("question is empty")

    warm_models(on_progress)

    if force_family:
        families = [force_family.strip()]
        _note(
            on_progress,
            "route",
            f"demo override: forcing folder {families[0]} (wrong-family probe)",
        )
    else:
        _note(on_progress, "route", "picking folder(s) from the question text")
        families = route_families(question)
    _note(on_progress, "route", f"folders: {', '.join(families)}")

    _note(
        on_progress,
        "retrieve",
        f"hybrid BM25 + embedding search, top {retrieve_k}",
    )
    hits = retrieve(question, families, retrieve_k)
    _note(on_progress, "retrieve", f"got {len(hits)} hits")

    _note(on_progress, "passages", "loading passage text for the hits")
    passages = load_texts(hits)
    _note(on_progress, "passages", f"loaded text for {len(passages)} passages")

    if not passages:
        released, confidence, deberta_calls, deberta_keeps = WITHHOLD, "withheld", 0, 0
        _note(on_progress, "answer", "no passage text, withholding")
    else:
        _note(on_progress, "write", "Qwen drafting from retrieved pages")
        draft_text = draft(question, passages)
        _note(on_progress, "write", f"draft ready ({len(draft_text)} chars)")

        calls = {"count": 0, "keeps": 0}
        base_entails = entailment_checker()

        def counting_entails(premise: str, hypothesis: str) -> bool:
            if aq._entails is None:
                _note(
                    on_progress,
                    "deberta",
                    "loading DeBERTa-v3 MNLI from cache to GPU (first paraphrase)",
                )
            calls["count"] += 1
            kept = base_entails(premise, hypothesis)
            if kept:
                calls["keeps"] += 1
            _note(
                on_progress,
                "deberta",
                f"call #{calls['count']}: "
                f"{'entailed, keep' if kept else 'not entailed, drop'}",
            )
            return kept

        _note(on_progress, "gate", "running confidence gate (copy / DeBERTa / withhold)")
        released = gate(draft_text, passages, counting_entails, question)
        confidence = confidence_label(released, calls["keeps"])
        deberta_calls = calls["count"]
        deberta_keeps = calls["keeps"]
        if deberta_calls == 0:
            _note(
                on_progress,
                "gate",
                f"done, confidence={confidence} (no DeBERTa needed)",
            )
        else:
            _note(
                on_progress,
                "gate",
                f"done, confidence={confidence}, "
                f"deberta_calls={deberta_calls}, keeps={deberta_keeps}",
            )

    withheld = confidence == "withheld"
    cited = cited_pages(released)

    metrics = None
    if gold is not None or expect_withhold:
        _note(on_progress, "metrics", "scoring citations vs hand-checked gold")
        scores = prediction_scores(cited, gold or [], withheld, expect_withhold)
        metrics = {
            "precision": float(scores["precision"]),
            "recall": float(scores["recall"]),
            "f1": float(scores["f1"]),
            "mrr": float(scores["mrr"]),
            "rank": int(scores["rank"]),
            "citation_accuracy": float(scores["precision"]),
        }
        _note(
            on_progress,
            "metrics",
            f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
            f"f1={metrics['f1']:.3f}",
        )

    _note(on_progress, "done", "finished")
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
    forced = str(item.get("force_family") or "").strip() or None
    return run_turn(
        item["question"],
        gold=item.get("relevant") or [],
        expect_withhold=bool(item.get("withhold")),
        force_family=forced,
        **kwargs,
    )


def confidence_matrix(rows: list[dict]) -> dict[str, int]:
    counts = {"copied-sentence": 0, "deberta": 0, "withheld": 0}
    for row in rows:
        label = row.get("confidence", "")
        if label in counts:
            counts[label] += 1
    return counts
