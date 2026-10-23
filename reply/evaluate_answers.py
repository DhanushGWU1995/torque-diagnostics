#!/usr/bin/env python3
"""Score the answer's cited pages against the hand-checked pages.

Each line of the question file is one JSON object with an id, a question,
and the pages that support it. Precision, recall, F1, and MRR use the
pages the answer cites, in the order they appear. A cited page is correct
when its document id and page number are in that list. MRR uses the
position of the first correct citation. The indexes are not rebuilt.

Set ANSWER=1 to run Qwen and the confidence gate. Without it there is no
prediction to score.

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


def prediction_scores(
    predicted: list[tuple[str, int]],
    relevant: list[dict],
    withheld: bool,
    expect_withhold: bool,
) -> dict[str, float | int]:
    """Score the pages the answer cites against the hand-checked pages."""
    ordered: list[tuple[str, int]] = []
    for page in predicted:
        if page not in ordered:
            ordered.append(page)
    gold = {page_key(item["document_id"], item["page_number"]) for item in relevant}
    if expect_withhold:
        correct = withheld
        value = 1.0 if correct else 0.0
        return {
            "precision": value,
            "recall": value,
            "f1": value,
            "mrr": value,
            "rank": 1 if correct else 0,
            "matched": 1 if correct else 0,
        }
    matched = [page for page in ordered if page in gold]
    precision = len(matched) / len(ordered) if ordered else 0.0
    recall = len(matched) / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rank = next((index for index, page in enumerate(ordered, start=1) if page in gold), 0)
    mrr = 1 / rank if rank else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": mrr,
        "rank": rank,
        "matched": len(matched),
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


def evaluate(questions: list[dict], k: int) -> str:
    lines = [
        f"k={k}",
        "precision, recall, f1, and mrr score the pages cited in the answer",
        "id\tprecision\trecall\tf1\tmrr\trank\trelevant\tconfidence",
    ]
    scored = []
    for item in questions:
        families = route_families(item["question"])
        hits = retrieve(item["question"], families, k)
        released, check, calls = answer_one(item["question"], hits)
        withheld = check == "withheld"
        scores = prediction_scores(
            cited_pages(released),
            item.get("relevant", []),
            withheld,
            bool(item.get("withhold")),
        )
        scored.append(scores)
        check = f"{check}; deberta_calls={calls}"
        lines.append(format_row(item, scores, released, check))
    lines.append(
        "macro\t"
        f"precision={mean([float(row['precision']) for row in scored]):.3f}\t"
        f"recall={mean([float(row['recall']) for row in scored]):.3f}\t"
        f"f1={mean([float(row['f1']) for row in scored]):.3f}\t"
        f"mrr={mean([float(row['mrr']) for row in scored]):.3f}"
    )
    return "\n".join(lines) + "\n"


def check_metrics() -> None:
    one = [{"document_id": "3AXD50000349838", "page_number": 17}]
    flange = prediction_scores([("3AXD50000349838", 17)], one, False, False)
    assert flange["precision"] == 1
    assert flange["recall"] == 1
    assert flange["f1"] == 1
    assert flange["mrr"] == 1
    acs_gold = [
        {"document_id": "3AXD50000034052", "page_number": 68},
        {"document_id": "3ABD00069490", "page_number": 27},
    ]
    acs = prediction_scores(
        [("3AXD50000034052", 68), ("3AXD50000034052", 213)],
        acs_gold,
        False,
        False,
    )
    assert abs(float(acs["precision"]) - 0.5) < 1e-9
    assert abs(float(acs["recall"]) - 0.5) < 1e-9
    assert abs(float(acs["f1"]) - 0.5) < 1e-9
    assert acs["rank"] == 1
    assert acs["mrr"] == 1
    missed = prediction_scores([], one, True, False)
    assert missed["f1"] == 0
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
    if os.environ.get("ANSWER", "").strip() != "1":
        print("Set ANSWER=1. Precision, recall, F1, and MRR score the pages the answer cites.", file=sys.stderr)
        return 1
    k = int(os.environ.get("TOP_K", "15"))
    report = evaluate(load_questions(path), k)
    Path(REPORT_NAME).write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
