#!/usr/bin/env python3
"""Score retrieval and the confidence check on hand-checked questions.

Each line of the question file is one JSON object with an id, a question,
and the pages that support it. A page is relevant when its document id and
page number are in that list. The top passages come from the same hybrid
search the reply uses. Precision, recall, and F1 use the distinct pages in
those passages. MRR uses the rank of the first relevant page.

Set ANSWER=1 to also run Qwen and the confidence gate. That step is skipped
otherwise, so the retrieval scores do not load the language model. The
indexes are not rebuilt.

    uv run python reply/evaluate_answers.py

    ANSWER=1 uv run python reply/evaluate_answers.py

QUESTIONS overrides the question file. TOP_K overrides the passage cutoff.
The report is written to evaluation_report.txt in the working directory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reply"))

from answer_question import (  # noqa: E402
    WITHHOLD,
    citations,
    draft,
    entailment_checker,
    gate,
    is_refusal,
    load_texts,
    retrieve,
)
from manual_tokens import route_families  # noqa: E402

REPORT_NAME = "evaluation_report.txt"
DEFAULT_QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"


def page_key(document_id: object, page_number: object) -> tuple[str, int]:
    return str(document_id), int(page_number)


def ranked_pages(hits: list[dict]) -> list[tuple[str, int]]:
    """Distinct pages in the order of their first retrieved passage."""
    ordered: list[tuple[str, int]] = []
    for hit in hits:
        key = page_key(hit["document_id"], hit["page_number"])
        if key not in ordered:
            ordered.append(key)
    return ordered


def retrieval_scores(hits: list[dict], relevant: list[dict], k: int) -> dict[str, float | int]:
    gold = {page_key(item["document_id"], item["page_number"]) for item in relevant}
    pages = ranked_pages(hits[:k])
    matched = [page for page in pages if page in gold]
    precision = len(matched) / len(pages) if pages else 0.0
    recall = len(matched) / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rank = next((index for index, page in enumerate(pages, start=1) if page in gold), 0)
    mrr = 1 / rank if rank else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": mrr,
        "rank": rank,
        "matched": len(matched),
        "retrieved_pages": len(pages),
        "relevant_pages": len(gold),
    }


def confidence_label(released: str, deberta_keeps: int) -> str:
    if WITHHOLD in released or is_refusal(released):
        return "withheld"
    if deberta_keeps:
        return "deberta"
    return "copied-sentence"


def load_questions(path: Path) -> list[dict]:
    questions = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not str(item.get("question", "")).strip():
            raise ValueError(f"line {line_number} has no question")
        if "relevant" not in item and not item.get("withhold"):
            raise ValueError(f"line {line_number} needs relevant pages or withhold")
        item.setdefault("relevant", [])
        item.setdefault("id", f"q{line_number}")
        questions.append(item)
    if not questions:
        raise ValueError(f"no questions in {path}")
    return questions


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def answer_one(question: str, hits: list[dict]) -> tuple[str, str, int]:
    """Run the writer and the gate. Returns the released text, the check, and DeBERTa calls."""
    passages = load_texts(hits)
    if not passages:
        return WITHHOLD, "withheld", 0
    calls = {"count": 0, "keeps": 0}
    entails = entailment_checker()

    def counting_entails(premise: str, hypothesis: str) -> bool:
        calls["count"] += 1
        kept = entails(premise, hypothesis)
        if kept:
            calls["keeps"] += 1
        return kept

    released = gate(draft(question, passages), passages, counting_entails, question)
    return released, confidence_label(released, calls["keeps"]), calls["count"]


def cited_pages(released: str) -> list[tuple[str, int]]:
    return [page_key(document_id, page_number) for document_id, page_number in citations(released)]


def format_row(item: dict, scores: dict[str, float | int], released: str, check: str) -> str:
    pages = ", ".join(
        f"{page['document_id']} p.{page['page_number']}" for page in item.get("relevant", [])
    )
    line = (
        f"{item['id']}\tprecision={scores['precision']:.3f}\trecall={scores['recall']:.3f}\t"
        f"f1={scores['f1']:.3f}\tmrr={scores['mrr']:.3f}\trank={scores['rank']}\t"
        f"relevant={pages or 'withhold'}"
    )
    if check:
        line += f"\tconfidence={check}"
    if released:
        line += f"\n  answer: {released}"
    return line


def evaluate(questions: list[dict], k: int, run_answer: bool) -> str:
    lines = [f"k={k}", "id\tprecision\trecall\tf1\tmrr\trank\trelevant\tconfidence"]
    retrieval_rows = []
    for item in questions:
        families = route_families(item["question"])
        hits = retrieve(item["question"], families, k)
        scores = retrieval_scores(hits, item.get("relevant", []), k)
        if item.get("relevant"):
            retrieval_rows.append(scores)
        released = ""
        check = ""
        if run_answer:
            released, check, calls = answer_one(item["question"], hits)
            if item.get("withhold"):
                check = f"{check}; correct={str(check == 'withheld').lower()}"
            elif item.get("relevant"):
                gold = {page_key(page["document_id"], page["page_number"]) for page in item["relevant"]}
                cited = set(cited_pages(released))
                supported = check != "withheld" and bool(cited & gold)
                check = f"{check}; deberta_calls={calls}; cited_relevant={str(supported).lower()}"
        lines.append(format_row(item, scores, released, check))
    if retrieval_rows:
        lines.append(
            "macro\t"
            f"precision={mean([float(row['precision']) for row in retrieval_rows]):.3f}\t"
            f"recall={mean([float(row['recall']) for row in retrieval_rows]):.3f}\t"
            f"f1={mean([float(row['f1']) for row in retrieval_rows]):.3f}\t"
            f"mrr={mean([float(row['mrr']) for row in retrieval_rows]):.3f}"
        )
    return "\n".join(lines) + "\n"


def check_metrics() -> None:
    hits = [
        {"document_id": "A", "page_number": 2},
        {"document_id": "A", "page_number": 2},
        {"document_id": "B", "page_number": 9},
        {"document_id": "C", "page_number": 1},
    ]
    relevant = [{"document_id": "B", "page_number": 9}, {"document_id": "D", "page_number": 4}]
    scores = retrieval_scores(hits, relevant, 4)
    assert scores["rank"] == 2
    assert scores["mrr"] == 0.5
    assert scores["matched"] == 1
    assert scores["retrieved_pages"] == 3
    assert abs(float(scores["precision"]) - 1 / 3) < 1e-9
    assert abs(float(scores["recall"]) - 0.5) < 1e-9
    assert confidence_label(WITHHOLD, 0) == "withheld"
    assert confidence_label("Use a torque wrench (document A, page 2).", 0) == "copied-sentence"
    assert confidence_label("Use a torque wrench (document A, page 2).", 1) == "deberta"
    assert cited_pages("Use a torque wrench (document A, page 2).") == [("A", 2)]


def main() -> int:
    check_metrics()
    path = Path(os.environ.get("QUESTIONS", str(DEFAULT_QUESTIONS)))
    if not path.is_file():
        print(f"Question file not found: {path}", file=sys.stderr)
        return 1
    k = int(os.environ.get("TOP_K", "15"))
    run_answer = os.environ.get("ANSWER", "").strip() == "1"
    report = evaluate(load_questions(path), k, run_answer)
    Path(REPORT_NAME).write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
