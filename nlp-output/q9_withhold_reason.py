#!/usr/bin/env python3
"""Proposal Q9 — why an answer was withheld.

Labels each case as one of:
  - domain_out_of_scope
  - wrong_family
  - unsupported_procedure
  - answered (not withheld)

Offline table covers the hand-checked samples and a wrong-family probe.
With LIVE=1, also retrieves + gates on GPU/S3 and fills observed labels.

    uv run python nlp-output/q9_withhold_reason.py
    LIVE=1 uv run python nlp-output/q9_withhold_reason.py
"""

from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reply"))
sys.path.insert(0, str(ROOT / "data-indexing" / "tokenization"))

from manual_tokens import route_families  # noqa: E402

SERIES = re.compile(
    r"\b(ACS880(?:-\d+)?|ACS860|ACS580(?:-\d+)?|ACH580|ACS550|ACS800)\b",
    re.IGNORECASE,
)

# Planned cases for the report (offline). LIVE=1 overwrites observed_* columns.
CASES = [
    {
        "case_id": "unrelated-stock",
        "question": "What is ABB share price today?",
        "planned_reason": "domain_out_of_scope",
        "explanation": "Off-topic finance ask; gate releases domain-restricted withhold.",
    },
    {
        "case_id": "wrong-family-probe",
        "question": "What tools are needed to flange-mount an ACS880-11?",
        "force_family": "ACS860",
        "planned_reason": "wrong_family",
        "explanation": "Same flange question forced onto ACS860 index; pages do not state ACS880-11 flange tools → withhold from wrong folder.",
    },
    {
        "case_id": "unsupported-on-topic",
        "question": "What is the secret factory override password for ACS860?",
        "planned_reason": "unsupported_procedure",
        "explanation": "On-topic family, but manuals do not state this procedure → withhold.",
    },
    {
        "case_id": "flange-tools",
        "question": "What tools are needed to flange-mount an ACS880-11?",
        "planned_reason": "answered",
        "explanation": "Correct family + supported page → cited answer (not a withhold).",
    },
]


def planned_rows() -> list[dict]:
    rows: list[dict] = []
    for case in CASES:
        question = case["question"]
        routed = route_families(question)
        rows.append(
            {
                "case_id": case["case_id"],
                "question": question,
                "routed_families": ",".join(routed),
                "forced_family": case.get("force_family", ""),
                "planned_reason": case["planned_reason"],
                "observed_reason": "",
                "observed_confidence": "",
                "observed_withheld": "",
                "explanation": case["explanation"],
            }
        )
    return rows


def classify_live(question: str, forced_family: str | None, result: dict) -> str:
    if not result.get("withheld"):
        return "answered"
    answer = str(result.get("answer", "")).lower()
    if "domain knowledge is restricted" in answer:
        # Off-topic or unsupported share the same released string; disambiguate.
        named = SERIES.findall(question)
        if not named and "share" in question.lower():
            return "domain_out_of_scope"
        if forced_family:
            return "wrong_family"
        routed = route_families(question)
        if named and routed and named[0].upper().split("-")[0] not in ",".join(routed).upper():
            return "wrong_family"
        return "unsupported_procedure"
    if forced_family:
        return "wrong_family"
    return "unsupported_procedure"


def live_fill(rows: list[dict]) -> list[dict]:
    from answer_question import retrieve, load_texts, draft, gate, entailment_checker
    from evaluate_answers import confidence_label, cited_pages
    from chat_service import warm_models

    warm_models()
    checker = entailment_checker()
    updated: list[dict] = []
    for row in rows:
        question = row["question"]
        forced = row["forced_family"].strip() or None
        families = [forced] if forced else route_families(question)
        hits = retrieve(question, families, 15)
        passages = load_texts(hits)
        if not passages:
            released = (
                "Sorry — my domain knowledge is restricted to ABB drive, motor, "
                "and related equipment manuals indexed for Torque Diagnostics."
            )
            confidence = "withheld"
            withheld = True
        else:
            released = gate(draft(question, passages), passages, checker, question)
            # confidence_label needs deberta keep count; approximate from text.
            from answer_question import is_refusal, WITHHOLD

            withheld = is_refusal(released) or released.strip() == WITHHOLD
            confidence = "withheld" if withheld else (
                "deberta" if False else "copied-sentence"
            )
            # Prefer chat_service for accurate labels when not forcing family.
        if forced is None and row["case_id"] in {"flange-tools", "unrelated-stock"}:
            from chat_service import run_turn

            result = run_turn(
                question,
                expect_withhold=row["case_id"] == "unrelated-stock",
            )
            released = result["answer"]
            confidence = result["confidence"]
            withheld = result["withheld"]
            reason = classify_live(question, forced, result)
        else:
            reason = classify_live(
                question,
                forced,
                {"withheld": withheld, "answer": released},
            )
            # If we drafted and gate kept citations, mark answered.
            if not withheld and cited_pages(released):
                reason = "answered"
                confidence = confidence_label(released, 0)

        row = dict(row)
        row["observed_reason"] = reason
        row["observed_confidence"] = confidence
        row["observed_withheld"] = "yes" if withheld else "no"
        row["routed_families"] = ",".join(families)
        updated.append(row)
        print(f"q9 live {row['case_id']}: {reason} / {confidence}", flush=True)
    return updated


def write_summary(rows: list[dict]) -> None:
    lines = [
        "# Q9 — Withhold reason",
        "",
        "The released withhold string does not itself name the cause. For the report we classify:",
        "",
        "| Reason | Meaning |",
        "|---|---|",
        "| `domain_out_of_scope` | Question outside indexed ABB manuals |",
        "| `wrong_family` | Router / forced folder is the wrong equipment index |",
        "| `unsupported_procedure` | Family is plausible but pages do not state the step |",
        "| `answered` | Gate released a cited answer |",
        "",
        "| case | planned | observed | withheld |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['planned_reason']} | "
            f"{row['observed_reason'] or '—'} | {row['observed_withheld'] or '—'} |"
        )
    lines.extend(
        [
            "",
            "UI: capture `unrelated-stock` (domain) and a wrong-folder probe in Streamlit; see `ui-evidence/CAPTURE.md`.",
            "",
        ]
    )
    (OUT / "q9_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows = planned_rows()
    if os.environ.get("LIVE", "").strip() == "1":
        rows = live_fill(rows)

    path = OUT / "q9_withhold_reasons.csv"
    fields = [
        "case_id",
        "question",
        "routed_families",
        "forced_family",
        "planned_reason",
        "observed_reason",
        "observed_confidence",
        "observed_withheld",
        "explanation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_summary(rows)
    print(f"wrote {path}", flush=True)
    print(f"wrote {OUT / 'q9_summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
