#!/usr/bin/env python3
"""Proposal Q8 + Q11 — cite vs drop, and copied-sentence vs DeBERTa.

Always writes local gate demos (no GPU). With LIVE=1, also runs the four
hand-checked samples through chat_service (needs GPU + S3 indexes).

    uv run python nlp-output/q8_q11_confidence_gate.py
    LIVE=1 uv run python nlp-output/q8_q11_confidence_gate.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reply"))

from answer_question import WITHHOLD, gate  # noqa: E402


def _always_entail(_premise: str, _hypothesis: str) -> bool:
    return True


def _never_entail(_premise: str, _hypothesis: str) -> bool:
    return False


def local_demos() -> list[dict]:
    """Deterministic gate cases that answer Q8/Q11 without calling models."""
    page = {
        "chunk_id": "demo:0017:000",
        "document_id": "3AXD50000349838",
        "page_number": 17,
        "text": (
            "You need the following tools for flange-mounting.\n"
            "Torque wrench.\n"
            "Hex key set.\n"
            "Socket set.\n"
        ),
    }
    cases = [
        {
            "case_id": "copied-sentence",
            "question": "What tools are needed to flange-mount an ACS880-11?",
            "draft": "You need the following tools for flange-mounting. Torque wrench.",
            "entails": _never_entail,
            "expect_path": "copied-sentence",
            "note": "Draft words already on the page → quote page sentence and cite.",
        },
        {
            "case_id": "deberta-keep",
            "question": "How do I protect the DC bus?",
            "draft": "The overvoltage controller limits high DC bus voltage during braking.",
            "passage": {
                "chunk_id": "demo:0062:000",
                "document_id": "3AXD50000034052",
                "page_number": 62,
                "text": "The overvoltage controller protects against high DC-link voltage.",
            },
            "entails": _always_entail,
            "expect_path": "deberta",
            "note": "Partial paraphrase (≈34–50% overlap path) → entailment keep + cite.",
        },
        {
            "case_id": "drop-unsupported",
            "question": "What tools are needed?",
            "draft": "Always use a laser alignment tool that the page never mentions.",
            "entails": _never_entail,
            "expect_path": "withheld",
            "note": "Claim not on page and not entailed → drop; nothing left → withhold.",
        },
        {
            "case_id": "domain-refuse",
            "question": "What is ABB share price today?",
            "draft": "sorry — my domain knowledge is restricted to the indexed abb manuals",
            "entails": _never_entail,
            "expect_path": "withheld",
            "note": "Off-topic draft refusal → gate normalizes to domain-restricted WITHHOLD.",
        },
    ]
    rows: list[dict] = []
    for case in cases:
        passages = [case.get("passage", page)]
        released = gate(case["draft"], passages, case["entails"], case["question"])
        withheld = released.strip() == WITHHOLD or "domain knowledge is restricted" in released.lower()
        cited = "document " in released.lower() and "page " in released.lower()
        rows.append(
            {
                "case_id": case["case_id"],
                "question": case["question"],
                "expect_path": case["expect_path"],
                "released_preview": released[:180].replace("\n", " "),
                "cited": "yes" if cited and not withheld else "no",
                "withheld": "yes" if withheld else "no",
                "note": case["note"],
            }
        )
    return rows


def write_q8_summary(rows: list[dict]) -> None:
    lines = [
        "# Q8 — Cite vs drop (confidence gate)",
        "",
        "When a drafted sentence is **already on a retrieved page**, the gate keeps that page wording and attaches `document_id` + `page_number`.",
        "When the page does **not** support the claim (and DeBERTa does not entail it), the sentence is **dropped**.",
        "If nothing remains, the gate releases the domain-restricted withhold message.",
        "",
        "| case | path | cited | withheld |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['expect_path']} | {row['cited']} | {row['withheld']} |"
        )
    lines.extend(
        [
            "",
            "Live technician evidence: run the Streamlit samples `flange-tools` (cite) and `unrelated-stock` (withhold), and capture UI panels per `ui-evidence/CAPTURE.md`.",
            "",
        ]
    )
    (OUT / "q8_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_method_doc() -> None:
    text = """# Q10 / Q11 — How confidence is derived

## Q10 — Hybrid score + evaluation gate

1. **Hybrid retrieval confidence (ranking only)**  
   For the routed equipment family, BM25 scores and BGE cosine scores are each scaled by their own max, then **added**. The top 15 fused hits become the generation context. This decides *which pages* the writer may use; it is not the final answer confidence label.

2. **Answer confidence (released text)**  
   After Qwen drafts, the confidence gate labels the outcome:
   - `copied-sentence` — claim already stated on a retrieved page (quoted + cited)
   - `deberta` — partial paraphrase kept only after DeBERTa-v3 MNLI entailment
   - `withheld` — all claims dropped, or domain-restricted refusal

So “confidence that an answer is correct” in this project means: **hybrid chose supporting pages**, and the **gate verified each released sentence against those pages** (copy or entailment). Unsupported steps are not filled in.

## Q11 — When is the answer settled without DeBERTa?

| Situation | What happens | Label |
|---|---|---|
| Exact / high word overlap with a manual line (≥50%, or ≥4 novel words covered ≥60% by one page sentence) | Quote the page sentence; cite doc+page | `copied-sentence` |
| Best line overlap about 34–50% | Load DeBERTa once; keep only if entailment | `deberta` |
| Overlap &lt;34%, contradiction, or off-topic | Drop claim; withhold if nothing left | `withheld` |

DeBERTa is **not** used for every sentence — only for the weak-paraphrase band. Sample `acs860-ov-controller-protect` is the hand-checked DeBERTa path; `flange-tools` is the copied-sentence path.

## UI evidence

Streamlit shows the same labels, DeBERTa call/keep counts, and top-10 passages. Capture screenshots using `ui-evidence/CAPTURE.md`.
"""
    (OUT / "q10_q11_confidence_method.md").write_text(text, encoding="utf-8")


def live_samples() -> list[dict]:
    from chat_service import run_sample, sample_questions

    rows: list[dict] = []
    for item in sample_questions():
        print(f"live: {item['id']}", flush=True)
        result = run_sample(item)
        rows.append(
            {
                "case_id": item["id"],
                "question": item["question"],
                "expect_path": result["confidence"],
                "released_preview": result["answer"][:180].replace("\n", " "),
                "cited": "yes" if result["cited_pages"] else "no",
                "withheld": "yes" if result["withheld"] else "no",
                "note": (
                    f"deberta_calls={result['deberta_calls']}; "
                    f"deberta_keeps={result['deberta_keeps']}"
                ),
            }
        )
    return rows


def main() -> int:
    rows = local_demos()
    path = OUT / "q8_gate_cite_vs_drop.csv"
    fields = ["case_id", "question", "expect_path", "released_preview", "cited", "withheld", "note"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_q8_summary(rows)
    write_method_doc()
    print(f"wrote {path}", flush=True)
    print(f"wrote {OUT / 'q8_summary.md'}", flush=True)
    print(f"wrote {OUT / 'q10_q11_confidence_method.md'}", flush=True)

    if os.environ.get("LIVE", "").strip() == "1":
        live_rows = live_samples()
        live_path = OUT / "q8_q11_live_samples.csv"
        with live_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(live_rows)
        print(f"wrote {live_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
